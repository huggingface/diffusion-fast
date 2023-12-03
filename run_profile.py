import torch


torch.set_float32_matmul_precision("high")

from torch._inductor import config as inductorconfig # noqa: E402
inductorconfig.triton.unique_kernel_names = True 

import functools # noqa: E402
import sys  # noqa: E402

sys.path.append(".")
from utils import create_parser
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
        + f"fp16@{args.no_fp16}-sdpa@{args.no_sdpa}-bs@{args.batch_size}-fuse@{args.enable_fused_projections}-upcast_vae@{args.upcast_vae}-steps@{args.num_inference_steps}-unet@{args.compile_unet}-vae@{args.compile_vae}-mode@{args.compile_mode}-change_comp_config@{args.change_comp_config}-do_quant@{args.do_quant}.json"
    )    
    runner = functools.partial(profiler_runner, trace_path)
    with torch.autograd.profiler.record_function("sdxl-brrr"):
        runner(run_inference, pipeline, args)
    return trace_path


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    if not args.compile_unet:
        args.compile_mode = "NA"

    trace_path = main(args)
    print(f"Trace generated at: {trace_path}")
