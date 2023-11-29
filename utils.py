import torch
import torch.utils.benchmark as benchmark
import gc
from typing import Union, Dict 
import csv

BENCHMARK_FIELDS = [
    "pipeline_cls",
    "ckpt_id",
    "batch_size",
    "num_inference_steps",
    "run_compile",
    "compile_mode",
    "time (secs)",
    "memory (gbs)",
    "actual_gpu_memory (gbs)",
]
TOTAL_GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / (1024**3)



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
    if not args.run_compile:
        compile_mode = "NA"
    else:
        compile_mode = args.compile_mode
    
    data_dict = {
        "pipeline_cls": pipeline_cls,
        "ckpt_id": ckpt,
        "batch_size": args.batch_size,
        "num_inference_steps": args.num_inference_steps,
        "run_compile": args.run_compile,
        "compile_mode": compile_mode,
        "time (secs)":time,
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