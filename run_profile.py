import torch


torch.set_float32_matmul_precision("high")

from torch._inductor import config as inductorconfig # noqa: E402
inductorconfig.triton.unique_kernel_names = True 

import functools # noqa: E402
import argparse  # noqa: E402
import sys  # noqa: E402

from diffusers import DiffusionPipeline  # noqa: E402


sys.path.append(".")
from run_benchmark import load_pipeline

CKPT_ID = "stabilityai/stable-diffusion-xl-base-1.0"
PROMPT = "ghibli style, a fantasy landscape with castles"


def profiler_runner(path, fn, *args, **kwargs):
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True) as prof:
        result = fn(*args, **kwargs)
    prof.export_chrome_trace(path)
    return result

def run_inference(pipe, args):
    _ = pipe(
        prompt=PROMPT,
        num_inference_steps=args.num_inference_steps,
        num_images_per_prompt=args.batch_size,
    )


def main(args) -> dict:
    pipeline = load_pipeline(args)
    
    # warmup.
    run_inference(pipeline, args)
    run_inference(pipeline, args)

    trace_path = (
        CKPT_ID.replace("/", "_")
        + f"-bs@{args.batch_size}-upcast_vae@{args.upcast_vae}-steps@{args.num_inference_steps}-unet@{args.compile_unet}-vae@{args.compile_vae}-mode@{args.compile_mode}-change_comp_config@{args.change_comp_config}-do_quant@{args.do_quant}.json"
    )    
    runner = functools.partial(profiler_runner, trace_path)
    with torch.autograd.profiler.record_function("sdxl-brrr"):
        runner(run_inference, pipeline, args)
    return trace_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--upcast_vae", action="store_true")
    parser.add_argument("--compile_unet", action="store_true")
    parser.add_argument("--compile_vae", action="store_true")
    parser.add_argument(
        "--compile_mode", type=str, default="reduce-overhead", choices=["reduce-overhead", "max-autotune"]
    )
    parser.add_argument("--change_comp_config", action="store_true")
    parser.add_argument("--do_quant", action="store_true")
    args = parser.parse_args()

    if not args.compile_unet:
        args.compile_mode = "NA"

    trace_path = main(args)
    print(f"Trace generated at: {trace_path}")
