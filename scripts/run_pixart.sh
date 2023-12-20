#!/bin/bash

cd ..

python run_benchmark_pixart.py --no_sdpa --no_bf16 --compile_mode=max-autotune && \
python run_benchmark_pixart.py --compile_transformer --compile_mode=max-autotune --compile_vae --change_comp_config && \
python run_benchmark_pixart.py --compile_transformer --compile_mode=max-autotune --compile_vae --change_comp_config --enable_fused_projections && \
python run_benchmark_pixart.py --compile_transformer --compile_mode=max-autotune --compile_vae --enable_fused_projections --do_quant "int8dynamic" --change_comp_config && \
python prepare_results.py --plot_title "PixArt-Alpha, Batch Size: 1, Steps: 30" --final_csv_filename "collated_results.csv"
