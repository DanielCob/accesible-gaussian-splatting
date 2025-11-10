"""Benchmark harness for Accessible Gaussian Splatting

Goals (per variant):
 - Quality: PSNR / SSIM / LPIPS via render.py + metrics.py
 - Time: COLMAP (preprocess), Training (7k iters), Total
 - Resources: peak VRAM (MB) via nvidia-smi, peak RAM (MB), #images processed,
	 matcher (sequential/exhaustive), sampling FPS label (2fps/5fps)
 - Comparison: internal grid over matcher x sampling rate (2fps vs 5fps)
	 Optionally, compare against an external reference repo.

Assumptions:
 - Source dataset contains undistorted images (images/) and possibly sparse/0.
 - We do NOT do undistortion here. We optionally rebuild sparse with COLMAP.

Examples (Windows PowerShell shown; quoting handled internally):
	# Basic internal comparison, train and render at 7k
	python compare_matching.py -s D:/data/tandt_scene --run_training

	# Rebuild sparse using COLMAP .bat, and create resized image pyramids
	python compare_matching.py -s D:/data/tandt_scene --rebuild_sparse \
		--colmap_executable "C:/Program Files/COLMAP/COLMAP.bat" --resize --run_training

	# Provide original base fps to derive 2fps/5fps subsampling steps
	python compare_matching.py -s D:/data/tandt_scene --base_fps 30 --run_training

	# Optional: also run the original reference repo for comparison
	python compare_matching.py -s D:/data/tandt_scene --reference_repo "..\\gaussian-splatting" --run_training
"""

import os
import sys
import shutil
import logging
import time
import json
import threading
from argparse import ArgumentParser
from typing import Optional
from pathlib import Path

try:
	import psutil  # type: ignore
	PSUTIL_AVAILABLE = True
except Exception:
	PSUTIL_AVAILABLE = False

parser = ArgumentParser("Accessible 3DGS Benchmark: matcher x sampling (2fps vs 5fps)")
parser.add_argument("--source_path", "-s", required=True, type=str, help="Base dataset root containing images/")
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--magick_executable", default="", type=str)
parser.add_argument("--resize", action="store_true", help="Resize images (creates images_2/_4/_8) for each variant")
parser.add_argument("--no_gpu", action="store_true")
parser.add_argument("--rebuild_sparse", action="store_true", help="Force COLMAP mapping to rebuild sparse in each variant root")
parser.add_argument("--run_training", action="store_true", help="Run training + render + metrics for each variant")
parser.add_argument("--output_dir", default="benchmark_results", type=str, help="Directory to store results under source_path")
parser.add_argument("--iterations", type=int, default=7000, help="Training/render iteration (default 7000)")
parser.add_argument("--base_fps", type=int, default=30, help="Assumed original FPS of input sequence for subsampling stride computation")
parser.add_argument("--target_fps", nargs="+", type=int, default=[2, 5], help="Target FPS variants to simulate via subsampling")
parser.add_argument("--reference_repo", type=str, default="", help="Optional path to the reference repo (original gaussian-splatting) to run as baseline")
parser.add_argument("--extra_train_args", type=str, default="", help="Extra args to pass to train.py (e.g., --antialiasing)")
args = parser.parse_args()

colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0

IS_WINDOWS = os.name == "nt"

def _xvfb_prefix():
	# Only needed on headless Linux; avoid on Windows/Mac.
	if IS_WINDOWS:
		return ""
	# If DISPLAY present, don't wrap
	return "" if os.environ.get("DISPLAY") else "xvfb-run -a "

# Require undistorted images folder
base_images = os.path.join(args.source_path, "images")
if not os.path.isdir(base_images):
    raise FileNotFoundError(f"Required folder not found: {base_images}")

comparison_root = os.path.join(args.source_path, args.output_dir)
os.makedirs(comparison_root, exist_ok=True)

def _compute_stride(base_fps: int, target_fps: int) -> int:
	if target_fps <= 0:
		return 1
	stride = max(1, round(base_fps / target_fps))
	return stride

def _subsample_and_copy(src_dir: str, dst_dir: str, stride: int) -> int:
	"""Copy every `stride`-th file in lexicographic order; returns count."""
	files = sorted([f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))])
	copied = 0
	for i, f in enumerate(files):
		if (i % stride) == 0:
			shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))
			copied += 1
	return copied

def _run_subprocess_with_measure(cmd: str, cwd: Optional[str], label: str, track_gpu: bool = False):
	"""Run a subprocess while sampling peak RAM (MB) and optionally peak VRAM (MB)."""
	import subprocess
	start = time.time()
	proc = subprocess.Popen(cmd, cwd=cwd, shell=True)

	peak_ram_mb = 0
	stop_flag = threading.Event()

	def _sample_ram():
		nonlocal peak_ram_mb
		if not PSUTIL_AVAILABLE:
			return
		try:
			p = psutil.Process(proc.pid)
		except Exception:
			return
		while not stop_flag.is_set():
			try:
				# Sum memory across process tree
				mem = 0
				procs = [p] + p.children(recursive=True)
				for cp in procs:
					try:
						mi = cp.memory_info().rss
						mem += mi
					except Exception:
						pass
				peak_ram_mb = max(peak_ram_mb, int(mem / (1024 * 1024)))
			except Exception:
				pass
			time.sleep(0.5)

	ram_thread = threading.Thread(target=_sample_ram, daemon=True)
	if PSUTIL_AVAILABLE:
		ram_thread.start()

	# Optional GPU sampling via nvidia-smi
	gpu_proc = None
	mem_log = None
	if track_gpu:
		try:
			mem_log = os.path.join(comparison_root, f"{label}_gpu_mem_log.txt")
			fout = open(mem_log, 'w')
			gpu_proc = subprocess.Popen([
				'nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-l', '1'
			], stdout=fout, stderr=subprocess.DEVNULL)
		except Exception:
			gpu_proc = None
			mem_log = None

	exit_code = proc.wait()
	stop_flag.set()
	if PSUTIL_AVAILABLE:
		ram_thread.join(timeout=1.0)

	# Stop GPU sampling
	peak_vram_mb = None
	if gpu_proc is not None:
		try:
			gpu_proc.terminate()
		except Exception:
			pass
		if mem_log and os.path.isfile(mem_log):
			try:
				with open(mem_log, 'r') as f:
					vals = []
					for line in f:
						line = line.strip()
						if line:
							try:
								vals.append(int(line))
							except:
								pass
				if vals:
					peak_vram_mb = max(vals)
			except Exception:
				pass

	elapsed = time.time() - start
	if exit_code != 0:
		raise RuntimeError(f"Command failed ({label}) with exit code {exit_code}: {cmd}")

	return {
		'elapsed_s': elapsed,
		'peak_ram_mb': peak_ram_mb if PSUTIL_AVAILABLE else None,
		'peak_vram_mb': peak_vram_mb
	}

def prepare_variant_root(label: str, stride: int):
	"""Create a variant root directory and copy subsampled images."""
	variant_root = os.path.join(comparison_root, label)
	os.makedirs(variant_root, exist_ok=True)
	variant_images = os.path.join(variant_root, "images")
	if os.path.isdir(variant_images):
		shutil.rmtree(variant_images)
	os.makedirs(variant_images, exist_ok=True)
	total = _subsample_and_copy(base_images, variant_images, stride)
	return variant_root, total

def run_colmap(matcher: str, variant_root: str):
	distorted_dir = os.path.join(variant_root, "distorted")
	os.makedirs(distorted_dir, exist_ok=True)
	os.makedirs(os.path.join(distorted_dir, "sparse"), exist_ok=True)

	# Feature extraction (on variant_root/images)
	xvfb = _xvfb_prefix()
	feat_cmd = (
		f'{xvfb}{colmap_command} feature_extractor '
		f'--database_path "{distorted_dir}/database.db" '
		f'--image_path "{variant_root}/images" '
		f'--ImageReader.single_camera 1 '
		f'--ImageReader.camera_model {args.camera} '
		f'--SiftExtraction.use_gpu {use_gpu}'
	)
	_run_subprocess_with_measure(feat_cmd, cwd=None, label=f"{Path(variant_root).name}_feature_extractor", track_gpu=False)

	if matcher == "sequential":
		match_cmd = (
			f'{xvfb}{colmap_command} sequential_matcher '
			f'--database_path "{distorted_dir}/database.db" '
			f'--SiftMatching.use_gpu {use_gpu} '
			f'--SequentialMatching.overlap 5 '
			f'--SequentialMatching.quadratic_overlap 1'
		)
	else:  # exhaustive
		match_cmd = (
			f'{xvfb}{colmap_command} exhaustive_matcher '
			f'--database_path "{distorted_dir}/database.db" '
			f'--SiftMatching.use_gpu {use_gpu}'
		)
	_run_subprocess_with_measure(match_cmd, cwd=None, label=f"{Path(variant_root).name}_matcher_{matcher}", track_gpu=False)

	# Mapper
	mapper_cmd = (
		f'{colmap_command} mapper '
		f'--database_path "{distorted_dir}/database.db" '
		f'--image_path "{variant_root}/images" '
		f'--output_path "{distorted_dir}/sparse" '
		f'--Mapper.ba_global_function_tolerance=0.000001'
	)
	_run_subprocess_with_measure(mapper_cmd, cwd=None, label=f"{Path(variant_root).name}_mapper", track_gpu=False)

	# Create sparse/0 expected by training by copying results from distorted
	sparse_dir = os.path.join(variant_root, "sparse")
	if os.path.isdir(sparse_dir):
		shutil.rmtree(sparse_dir)
	os.makedirs(os.path.join(sparse_dir, "0"), exist_ok=True)
	src0 = os.path.join(distorted_dir, "sparse", "0")
	if os.path.isdir(src0):
		for file in os.listdir(src0):
			s = os.path.join(src0, file)
			d = os.path.join(sparse_dir, "0", file)
			shutil.copy2(s, d)

	# Optional resize
	if args.resize:
		for scale_dir, percent in [("images_2", 50), ("images_4", 25), ("images_8", 12.5)]:
			out_dir = os.path.join(variant_root, scale_dir)
			if os.path.isdir(out_dir):
				shutil.rmtree(out_dir)
			os.makedirs(out_dir, exist_ok=True)
			for file in os.listdir(os.path.join(variant_root, "images")):
				src = os.path.join(variant_root, "images", file)
				dst = os.path.join(out_dir, file)
				shutil.copy2(src, dst)
				_run_subprocess_with_measure(f"{magick_command} mogrify -resize {percent}% \"{dst}\"", cwd=None, label=f"{Path(variant_root).name}_resize_{percent}", track_gpu=False)

def gather_stats(variant_root: str):
	images_file = os.path.join(variant_root, "sparse", "0", "images.txt")
	points_file = os.path.join(variant_root, "sparse", "0", "points3D.txt")
	def count_valid(path):
		if not os.path.isfile(path):
			return 0
		c = 0
		with open(path, 'r') as f:
			for line in f:
				line = line.strip()
				if not line or line.startswith('#'):
					continue
				c += 1
		return c
	return {
		'registered_images': count_valid(images_file),
		'points3D': count_valid(points_file)
	}

def _run_training_pipeline(variant_root: str, model_dir: str, label: str):
	"""Run train->render->metrics at specified iteration, returning stats and metrics."""
	os.makedirs(model_dir, exist_ok=True)
	# Train to args.iterations only
	train_cmd = f"python train.py -s \"{variant_root}\" -m \"{model_dir}\" --eval --iterations {args.iterations} {args.extra_train_args}"
	train_meas = _run_subprocess_with_measure(train_cmd, cwd=os.path.dirname(__file__), label=f"{label}_train", track_gpu=True)

	# Render at that iteration
	render_cmd = f"python render.py --iteration {args.iterations} -s \"{variant_root}\" -m \"{model_dir}\" --eval --skip_train"
	_run_subprocess_with_measure(render_cmd, cwd=os.path.dirname(__file__), label=f"{label}_render", track_gpu=False)

	# Metrics
	metrics_cmd = f"python metrics.py -m \"{model_dir}\""
	_run_subprocess_with_measure(metrics_cmd, cwd=os.path.dirname(__file__), label=f"{label}_metrics", track_gpu=False)

	# Parse metrics file
	results_json = os.path.join(model_dir, "results.json")
	metrics = {}
	if os.path.isfile(results_json):
		try:
			with open(results_json, 'r') as f:
				data = json.load(f)
				# Expect key like 'ours_7000' under 'test'
				# file format: { model_path: { method: {SSIM, PSNR, LPIPS} } } if multiple scenes; accessible writes per scene path
				# Here metrics.py writes a dict per scene_dir; we invoked with single model_dir so top-level is a dict of methods
				# Actually metrics.py sets: full_dict[scene_dir][method] = {...}; then writes that dict to model_dir/results.json
				# So data is like { "ours_7000": {"SSIM":..., "PSNR":..., "LPIPS":...}, ... }
				key = f"ours_{args.iterations}"
				if key in data:
					metrics = data[key]
				else:
					# fallback: take first entry
					first_key = next(iter(data.keys())) if data else None
					metrics = data.get(first_key, {}) if first_key else {}
		except Exception:
			pass
	return train_meas, metrics

grid_results = {}

for target_fps in args.target_fps:
	stride = _compute_stride(args.base_fps, target_fps)
	for matcher in ["sequential", "exhaustive"]:
		label = f"{matcher}_{target_fps}fps"
		print(f"\n=== Variant: {label} (stride {stride}) ===")
		variant_root, total_images = prepare_variant_root(label, stride)

		# If rebuild requested OR sparse missing, run COLMAP; otherwise reuse existing sparse
		sparse0 = os.path.join(variant_root, "sparse", "0")
		colmap_time_s = 0.0
		peak_ram_colmap = None
		if args.rebuild_sparse or not os.path.isdir(sparse0):
			start = time.time()
			run_colmap(matcher, variant_root)
			colmap_time_s = time.time() - start
			# Note: run_colmap uses multiple processes; peak RAM during those commands is sampled per subcommand in logs
		else:
			print("Reusing existing sparse/0")

		stats = gather_stats(variant_root)
		stats['images_total'] = total_images
		stats['matcher'] = matcher
		stats['target_fps'] = target_fps
		stats['colmap_time_s'] = colmap_time_s

		if args.run_training:
			model_dir = os.path.join(variant_root, "model")
			train_meas, metrics_vals = _run_training_pipeline(variant_root, model_dir, label)
			stats['train_time_s'] = train_meas['elapsed_s']
			stats['peak_ram_mb'] = train_meas['peak_ram_mb']
			stats['peak_vram_mb'] = train_meas['peak_vram_mb']
			stats['psnr'] = metrics_vals.get('PSNR')
			stats['ssim'] = metrics_vals.get('SSIM')
			stats['lpips'] = metrics_vals.get('LPIPS')

			# Optional reference repo comparison
			if args.reference_repo:
				ref_model_dir = os.path.join(variant_root, "model_ref")
				# Train reference
				ref_train = f"python train.py -s \"{variant_root}\" -m \"{ref_model_dir}\" --eval --iterations {args.iterations}"
				ref_meas = _run_subprocess_with_measure(ref_train, cwd=args.reference_repo, label=f"{label}_train_ref", track_gpu=True)
				# Render + metrics using reference repo scripts
				_run_subprocess_with_measure(f"python render.py --iteration {args.iterations} -s \"{variant_root}\" -m \"{ref_model_dir}\" --eval --skip_train", cwd=args.reference_repo, label=f"{label}_render_ref", track_gpu=False)
				_run_subprocess_with_measure(f"python metrics.py -m \"{ref_model_dir}\"", cwd=args.reference_repo, label=f"{label}_metrics_ref", track_gpu=False)
				ref_json = os.path.join(ref_model_dir, "results.json")
				ref_metrics = {}
				if os.path.isfile(ref_json):
					try:
						with open(ref_json, 'r') as f:
							data = json.load(f)
							key = f"ours_{args.iterations}"
							if key in data:
								ref_metrics = data[key]
							else:
								first_key = next(iter(data.keys())) if data else None
								ref_metrics = data.get(first_key, {}) if first_key else {}
					except Exception:
						pass
				stats['ref'] = {
					'train_time_s': ref_meas['elapsed_s'],
					'peak_ram_mb': ref_meas['peak_ram_mb'],
					'peak_vram_mb': ref_meas['peak_vram_mb'],
					'psnr': ref_metrics.get('PSNR'),
					'ssim': ref_metrics.get('SSIM'),
					'lpips': ref_metrics.get('LPIPS'),
				}

		grid_results[label] = stats

# Persist results
summary_txt = os.path.join(comparison_root, "summary.txt")
summary_json = os.path.join(comparison_root, "summary.json")

with open(summary_txt, 'w') as f:
	f.write("Accessible 3DGS Benchmark Summary\n")
	for label, st in grid_results.items():
		f.write(f"\n[{label}]\n")
		f.write(f"Matcher: {st.get('matcher')} | Target FPS: {st.get('target_fps')} | Images used: {st.get('images_total')}\n")
		f.write(f"Registered images: {st.get('registered_images')} | Points3D: {st.get('points3D')}\n")
		if st.get('colmap_time_s') is not None:
			mm, ss = divmod(int(st.get('colmap_time_s', 0)), 60)
			f.write(f"COLMAP time: {mm:02d}:{ss:02d}\n")
		if st.get('train_time_s') is not None:
			mm, ss = divmod(int(st.get('train_time_s', 0)), 60)
			f.write(f"Training time: {mm:02d}:{ss:02d}\n")
		total_s = (st.get('colmap_time_s') or 0) + (st.get('train_time_s') or 0)
		mm, ss = divmod(int(total_s), 60)
		f.write(f"Total time: {mm:02d}:{ss:02d}\n")
		f.write(f"Peak VRAM (MB): {st.get('peak_vram_mb')} | Peak RAM (MB): {st.get('peak_ram_mb')}\n")
		if st.get('psnr') is not None:
			f.write(f"PSNR: {st.get('psnr'):.4f} | SSIM: {st.get('ssim'):.4f} | LPIPS: {st.get('lpips'):.4f}\n")
		if 'ref' in st:
			ref = st['ref']
			f.write("Reference -> ")
			mm, ss = divmod(int(ref.get('train_time_s') or 0), 60)
			f.write(f"Train: {mm:02d}:{ss:02d} | ")
			f.write(f"VRAM: {ref.get('peak_vram_mb')} | RAM: {ref.get('peak_ram_mb')} | ")
			if ref.get('psnr') is not None:
				f.write(f"PSNR: {ref.get('psnr'):.4f} | SSIM: {ref.get('ssim'):.4f} | LPIPS: {ref.get('lpips'):.4f}")
			f.write("\n")

with open(summary_json, 'w') as f:
	json.dump(grid_results, f, indent=2)

print(f"\nBenchmark complete. Summaries saved to:\n - {summary_txt}\n - {summary_json}")
