import csv
import gc
from typing import Dict, List, Union

import torch
import torch.utils.benchmark as benchmark
import argparse


BENCHMARK_FIELDS = [
    "pipeline_cls",
    "ckpt_id",
    "fp16",
    "sdpa",
    "fused_qkv_projections",
    "upcast_vae",
    "batch_size",
    "num_inference_steps",
    "compile_unet",
    "compile_vae",
    "compile_mode",
    "change_comp_config",
    "do_quant",
    "time (secs)",
    "memory (gbs)",
    "actual_gpu_memory (gbs)",
]
TOTAL_GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / (1024**3)

def create_parser():
    """Creates CLI args parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--no_sdpa", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--enable_fused_projections", action="store_true")
    parser.add_argument("--upcast_vae", action="store_true")
    parser.add_argument("--compile_unet", action="store_true")
    parser.add_argument("--compile_vae", action="store_true")
    parser.add_argument(
        "--compile_mode", type=str, default="reduce-overhead", choices=["reduce-overhead", "max-autotune"]
    )
    parser.add_argument("--change_comp_config", action="store_true")
    parser.add_argument("--do_quant", action="store_true")
    return parser


def flush():
    """Wipes off memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()


def bytes_to_giga_bytes(bytes):
    return f"{(bytes / 1024 / 1024 / 1024):.3f}"


def benchmark_fn(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "f": f},
        num_threads=torch.get_num_threads(),
    )
    return f"{(t0.blocked_autorange().mean):.3f}"


def generate_csv_dict(
    pipeline_cls: str, ckpt: str, args, time: float, memory: float
) -> Dict[str, Union[str, bool, float]]:
    """Packs benchmarking data into a dictionary for latter serialization."""
    data_dict = {
        "pipeline_cls": pipeline_cls,
        "ckpt_id": ckpt,
        "fp16": args.use_fp16,
        "sdpa": args.use_sdpa,
        "fused_qkv_projections": args.enable_fused_projections,
        "upcast_vae": args.upcast_vae,
        "batch_size": args.batch_size,
        "num_inference_steps": args.num_inference_steps,
        "compile_unet": args.compile_unet,
        "compile_vae": args.compile_unet,
        "compile_mode": args.compile_mode,
        "change_comp_config": args.change_comp_config,
        "do_quant": args.do_quant,
        "time (secs)": time,
        "memory (gbs)": memory,
        "actual_gpu_memory (gbs)": f"{(TOTAL_GPU_MEMORY):.3f}",
    }
    return data_dict


def write_to_csv(file_name: str, data_dict: Dict[str, Union[str, bool, float]]):
    """Serializes a dictionary into a CSV file."""
    with open(file_name, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=BENCHMARK_FIELDS)
        writer.writeheader()
        writer.writerow(data_dict)


def collate_csv(input_files: List[str], output_file: str):
    """Collates multiple identically structured CSVs into a single CSV file."""
    with open(output_file, mode="w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=BENCHMARK_FIELDS)
        writer.writeheader()

        for file in input_files:
            with open(file, mode="r") as infile:
                reader = csv.DictReader(infile)
                for row in reader:
                    writer.writerow(row)
