"""Compare COLMAP sequential vs exhaustive matching on the same input frames.

This script replicates the conversion logic with two separate temporary dataset
roots, runs feature extraction + matching + mapping + undistortion for each, and
collects simple statistics (#registered images, #3D points). Optionally it can
run training and rendering at 7K iterations for both variants for downstream
quality comparison.

Usage (basic stats only):
	python compare_matching.py -s /path/to/base_dataset

Optional training + rendering:
	python compare_matching.py -s /path/to/base_dataset --run_training

The base dataset directory must already contain an `input/` folder with the raw
video-extracted frames (as produced earlier in the pipeline).
"""

import os
import shutil
import logging
import time
from argparse import ArgumentParser

parser = ArgumentParser("Sequential vs Exhaustive COLMAP matching comparison")
parser.add_argument("--source_path", "-s", required=True, type=str, help="Base dataset root containing input/ frames")
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--magick_executable", default="", type=str)
parser.add_argument("--resize", action="store_true", help="Resize images (creates images_2/_4/_8) for each variant")
parser.add_argument("--no_gpu", action="store_true")
parser.add_argument("--run_training", action="store_true", help="Run training + render + metrics for each variant")
parser.add_argument("--output_dir", default="matching_comparison", type=str, help="Directory to store results under source_path")
parser.add_argument("--iterations", type=int, default=7000, help="Training/render iteration (default 7000)")
args = parser.parse_args()

colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "imagemagick"
use_gpu = 1 if not args.no_gpu else 0
xvfb_prefix = "xvfb-run -a "

base_input = os.path.join(args.source_path, "input")
if not os.path.isdir(base_input):
	raise FileNotFoundError(f"Input frames directory not found: {base_input}")

comparison_root = os.path.join(args.source_path, args.output_dir)
os.makedirs(comparison_root, exist_ok=True)

def prepare_variant_root(method: str):
	variant_root = os.path.join(comparison_root, method)
	os.makedirs(variant_root, exist_ok=True)
	variant_input = os.path.join(variant_root, "input")
	os.makedirs(variant_input, exist_ok=True)
	# Copy frames (could be large; assume manageable for educational use)
	for f in os.listdir(base_input):
		src = os.path.join(base_input, f)
		dst = os.path.join(variant_input, f)
		if os.path.isfile(src):
			shutil.copy2(src, dst)
	return variant_root

def run_colmap(method: str, variant_root: str):
	distorted_dir = os.path.join(variant_root, "distorted")
	os.makedirs(distorted_dir, exist_ok=True)
	os.makedirs(os.path.join(distorted_dir, "sparse"), exist_ok=True)

	# Feature extraction
	feat_cmd = (
		f'{xvfb_prefix}{colmap_command} feature_extractor '
		f'--database_path {distorted_dir}/database.db '
		f'--image_path {variant_root}/input '
		f'--ImageReader.single_camera 1 '
		f'--ImageReader.camera_model {args.camera} '
		f'--SiftExtraction.use_gpu {use_gpu}'
	)
	if os.system(feat_cmd) != 0:
		raise RuntimeError(f"Feature extraction failed for {method}")

	if method == "sequential":
		match_cmd = (
			f'{xvfb_prefix}{colmap_command} sequential_matcher '
			f'--database_path {distorted_dir}/database.db '
			f'--SiftMatching.use_gpu {use_gpu} '
			f'--SequentialMatching.overlap 5 '
			f'--SequentialMatching.quadratic_overlap 1'
		)
	else:  # exhaustive
		match_cmd = (
			f'{xvfb_prefix}{colmap_command} exhaustive_matcher '
			f'--database_path {distorted_dir}/database.db '
			f'--SiftMatching.use_gpu {use_gpu}'
		)
	if os.system(match_cmd) != 0:
		raise RuntimeError(f"Feature matching failed for {method}")

	# Mapper
	mapper_cmd = (
		f'{colmap_command} mapper '
		f'--database_path {distorted_dir}/database.db '
		f'--image_path {variant_root}/input '
		f'--output_path {distorted_dir}/sparse '
		f'--Mapper.ba_global_function_tolerance=0.000001'
	)
	if os.system(mapper_cmd) != 0:
		raise RuntimeError(f"Mapper failed for {method}")

	# Undistort
	undist_cmd = (
		f'{colmap_command} image_undistorter '
		f'--image_path {variant_root}/input '
		f'--input_path {distorted_dir}/sparse/0 '
		f'--output_path {variant_root} '
		f'--output_type COLMAP'
	)
	if os.system(undist_cmd) != 0:
		raise RuntimeError(f"Image undistorter failed for {method}")

	# Move sparse files into sparse/0 (consistency with convert.py)
	sparse_dir = os.path.join(variant_root, "sparse")
	os.makedirs(os.path.join(sparse_dir, "0"), exist_ok=True)
	for file in os.listdir(sparse_dir):
		if file == '0':
			continue
		src = os.path.join(sparse_dir, file)
		dst = os.path.join(sparse_dir, "0", file)
		if os.path.isfile(src):
			shutil.move(src, dst)

	# Optional resize
	if args.resize:
		for scale_dir, percent in [("images_2", 50), ("images_4", 25), ("images_8", 12.5)]:
			out_dir = os.path.join(variant_root, scale_dir)
			os.makedirs(out_dir, exist_ok=True)
			for file in os.listdir(os.path.join(variant_root, "images")):
				src = os.path.join(variant_root, "images", file)
				dst = os.path.join(out_dir, file)
				shutil.copy2(src, dst)
				if os.system(f"{magick_command} mogrify -resize {percent}% {dst}") != 0:
					raise RuntimeError(f"Resize {percent}% failed for {method}")

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

def maybe_train(method: str, variant_root: str):
	if not args.run_training:
		return None
	model_dir = os.path.join(variant_root, "model")
	os.makedirs(model_dir, exist_ok=True)
	start = time.time()
	# Train (7K iterations default from OptimizationParams unless overridden)
	exit_code = os.system(f"python train.py -s {variant_root} -m {model_dir} --eval")
	if exit_code != 0:
		logging.error(f"Training failed for {method}")
		return None
	train_minutes = (time.time() - start)/60.0
	# Render at specified iteration
	exit_code = os.system(f"python render.py --iteration {args.iterations} -s {variant_root} -m {model_dir} --eval --skip_train")
	if exit_code != 0:
		logging.error(f"Render failed for {method}")
	# Metrics
	exit_code = os.system(f"python metrics.py -m \"{model_dir}\"")
	if exit_code != 0:
		logging.error(f"Metrics failed for {method}")
	return train_minutes

results = {}
for method in ["sequential", "exhaustive"]:
	print(f"\n=== Processing {method} matching ===")
	root = prepare_variant_root(method)
	run_colmap(method, root)
	stats = gather_stats(root)
	minutes = maybe_train(method, root)
	stats['train_minutes'] = minutes
	results[method] = stats

summary_path = os.path.join(comparison_root, "comparison_results.txt")
with open(summary_path, 'w') as f:
	f.write("Sequential vs Exhaustive Matching Comparison\n")
	for method, stats in results.items():
		f.write(f"\n[{method}]\n")
		f.write(f"Registered images: {stats['registered_images']}\n")
		f.write(f"Points3D: {stats['points3D']}\n")
		if stats['train_minutes'] is not None:
			f.write(f"Training time (min): {stats['train_minutes']:.2f}\n")
		else:
			f.write("Training: skipped\n")

print(f"\nComparison complete. Summary written to {summary_path}")
