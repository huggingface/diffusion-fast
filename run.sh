#!/bin/bash

python run_benchmark.py --compile_unet --compile_mode=max-autotune --compile_vae && \
python run_benchmark.py --compile_unet --compile_mode=max-autotune --compile_vae --change_comp_config && \
python run_benchmark.py --compile_unet --compile_mode=max-autotune --compile_vae --change_comp_config --enable_fused_projections && \
python run_benchmark.py --compile_unet --compile_mode=max-autotune --compile_vae --enable_fused_projections && \
python run_benchmark.py --compile_unet --compile_mode=max-autotune --compile_vae --do_quant "int8dynamic" && \
python run_benchmark.py --compile_unet --compile_mode=max-autotune --compile_vae --do_quant "int8weightonly" && \
python run_benchmark.py --compile_unet --compile_mode=max-autotune --compile_vae --enable_fused_projections --do_quant "int8dynamic" && \
python run_benchmark.py --compile_unet --compile_mode=max-autotune --compile_vae --enable_fused_projections --do_quant "int8dynamic" --change_comp_config && \
python prepare_plot.py --final_csv_filename collated_results_peft.csv
