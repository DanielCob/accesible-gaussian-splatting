#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from argparse import ArgumentParser
import time

mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]  # Unused after simplification
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]  # Unused after simplification
tanks_and_temples_scenes = ["truck", "train"]
deep_blending_scenes = ["drjohnson", "playroom"]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval")
parser.add_argument("--use_depth", action="store_true")
parser.add_argument("--use_expcomp", action="store_true")
parser.add_argument("--fast", action="store_true")
parser.add_argument("--aa", action="store_true")




args, _ = parser.parse_known_args()

"""Evaluation script simplified to ONLY run Tanks & Temples and Deep Blending datasets
at 7K iterations (single render point). Other scene lists are kept for backward
compatibility but not used. This reduces overall compute for accessible evaluation.
"""

all_scenes = []
all_scenes.extend(tanks_and_temples_scenes)
all_scenes.extend(deep_blending_scenes)

if not args.skip_training or not args.skip_rendering:
    # Only require paths for the datasets we still evaluate.
    parser.add_argument("--tanksandtemples", "-tat", required=True, type=str)
    parser.add_argument("--deepblending", "-db", required=True, type=str)
    args = parser.parse_args()
if not args.skip_training:
    common_args = " --disable_viewer --quiet --eval --test_iterations -1 "

    if args.aa:
        common_args += " --antialiasing "
    if args.use_depth:
        common_args += " -d depths2/ "

    if args.use_expcomp:
        # Respect exposure LR configuration (adjusted earlier in __init__.py defaults)
        common_args += " --exposure_lr_init 0.001 --exposure_lr_final 0.0001 --exposure_lr_delay_steps 5000 --exposure_lr_delay_mult 0.001 --train_test_exp "

    if args.fast:
        common_args += " --optimizer_type sparse_adam "

    tandt_timing = None
    db_timing = None

    start_time = time.time()
    for scene in tanks_and_temples_scenes:
        source = args.tanksandtemples + "/" + scene
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)
    tandt_timing = (time.time() - start_time)/60.0

    start_time = time.time()
    for scene in deep_blending_scenes:
        source = args.deepblending + "/" + scene
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)
    db_timing = (time.time() - start_time)/60.0
else:
    tandt_timing = None
    db_timing = None

with open(os.path.join(args.output_path, "timing.txt"), 'w') as file:
    file.write("Evaluation timings (minutes)\n")
    file.write(f"tandt: {tandt_timing if tandt_timing is not None else 'skipped'}\n")
    file.write(f"db: {db_timing if db_timing is not None else 'skipped'}\n")

if not args.skip_rendering:
    all_sources = []
    for scene in tanks_and_temples_scenes:
        all_sources.append(args.tanksandtemples + "/" + scene)
    for scene in deep_blending_scenes:
        all_sources.append(args.deepblending + "/" + scene)

    common_args = " --quiet --eval --skip_train"

    if args.aa:
        common_args += " --antialiasing "
    if args.use_expcomp:
        common_args += " --train_test_exp "

    # Single render point at 7000 iterations as requested.
    for scene, source in zip(all_scenes, all_sources):
        os.system("python render.py --iteration 7000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)

if not args.skip_metrics:
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene + "\" "
    os.system("python metrics.py -m " + scenes_string)
