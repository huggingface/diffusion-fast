#!/bin/bash

# From diffusion-fast source directory.

# Run Diffusion benchmark on CPU platforms.

python run_benchmark.py --no_sdpa --no_bf16 --device=cpu
python run_benchmark.py --compile_unet --compile_vae --device=cpu
python run_benchmark.py --compile_unet --compile_vae --enable_fused_projections --device=cpu
