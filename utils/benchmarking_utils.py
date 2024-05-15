import argparse
import copy
import csv
import gc
from typing import Dict, List, Union

import torch
import torch.utils.benchmark as benchmark


BENCHMARK_FIELDS = [
    "pipeline_cls",
    "ckpt_id",
    "bf16",
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
    "tag",
]


def create_parser(is_pixart=False):
    """Creates CLI args parser."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--prompt", type=str, default="ghibli style, a fantasy landscape with castles")
    parser.add_argument("--no_bf16", action="store_true")
    parser.add_argument("--no_sdpa", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--enable_fused_projections", action="store_true")

    if not is_pixart:
        parser.add_argument("--upcast_vae", action="store_true")

    if is_pixart:
        parser.add_argument("--compile_transformer", action="store_true")
    else:
        parser.add_argument("--compile_unet", action="store_true")

    parser.add_argument("--compile_vae", action="store_true")
    parser.add_argument("--compile_mode", type=str, default=None, choices=["reduce-overhead", "max-autotune"])
    parser.add_argument("--change_comp_config", action="store_true")
    parser.add_argument("--do_quant", type=str, default=None)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
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


def generate_csv_dict(pipeline_cls: str, args, time: float) -> Dict[str, Union[str, bool, float]]:
    """Packs benchmarking data into a dictionary for latter serialization."""
    data_dict = {
        "pipeline_cls": pipeline_cls,
        "ckpt_id": args.ckpt,
        "bf16": not args.no_bf16,
        "sdpa": not args.no_sdpa,
        "fused_qkv_projections": args.enable_fused_projections,
        "upcast_vae": "NA" if "PixArt" in pipeline_cls else args.upcast_vae,
        "batch_size": args.batch_size,
        "num_inference_steps": args.num_inference_steps,
        "compile_unet": args.compile_transformer if "PixArt" in pipeline_cls else args.compile_unet,
        "compile_vae": args.compile_vae,
        "compile_mode": args.compile_mode,
        "change_comp_config": args.change_comp_config,
        "do_quant": args.do_quant,
        "time (secs)": time,
        "tag": args.tag,
    }
    if args.device == "cuda":
        memory = bytes_to_giga_bytes(torch.cuda.max_memory_allocated())  # in GBs.
        TOTAL_GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        data_dict["memory (gbs)"] = memory
        data_dict["actual_gpu_memory (gbs)"] = f"{(TOTAL_GPU_MEMORY):.3f}"
    if "PixArt" in pipeline_cls:
        data_dict["compile_transformer"] = data_dict.pop("compile_unet")
    return data_dict


def write_to_csv(file_name: str, data_dict: Dict[str, Union[str, bool, float]], is_pixart=False):
    """Serializes a dictionary into a CSV file."""
    fields_copy = copy.deepcopy(BENCHMARK_FIELDS)
    fields = BENCHMARK_FIELDS
    if is_pixart:
        i = BENCHMARK_FIELDS.index("compile_unet")
        fields_copy[i] = "compile_transformer"
        fields = fields_copy
    with open(file_name, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerow(data_dict)


def collate_csv(input_files: List[str], output_file: str, is_pixart=False):
    """Collates multiple identically structured CSVs into a single CSV file."""
    fields_copy = copy.deepcopy(BENCHMARK_FIELDS)
    fields = BENCHMARK_FIELDS
    if is_pixart:
        i = BENCHMARK_FIELDS.index("compile_unet")
        fields_copy[i] = "compile_transformer"
        fields = fields_copy
    with open(output_file, mode="w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fields)
        writer.writeheader()

        for file in input_files:
            with open(file, mode="r") as infile:
                reader = csv.DictReader(infile)
                for row in reader:
                    writer.writerow(row)
