import torch


torch.set_float32_matmul_precision("high")

from torch._inductor import config as inductorconfig  # noqa: E402


inductorconfig.triton.unique_kernel_names = True

import functools  # noqa: E402
import sys  # noqa: E402


sys.path.append(".")
from utils.benchmarking_utils import create_parser  # noqa: E402
from utils.pipeline_utils import load_pipeline  # noqa: E402


def profiler_runner(path, fn, *args, **kwargs):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True
    ) as prof:
        result = fn(*args, **kwargs)
    prof.export_chrome_trace(path)
    return result


def run_inference(pipe, args):
    _ = pipe(
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        num_images_per_prompt=args.batch_size,
    )


def main(args) -> dict:
    pipeline = load_pipeline(
        ckpt=args.ckpt,
        compile_unet=args.compile_unet,
        compile_vae=args.compile_vae,
        no_sdpa=args.no_sdpa,
        no_bf16=args.no_bf16,
        upcast_vae=args.upcast_vae,
        enable_fused_projections=args.enable_fused_projections,
        do_quant=args.do_quant,
        compile_mode=args.compile_mode,
        change_comp_config=args.change_comp_config,
        device=args.device,
    )

    # warmup.
    run_inference(pipeline, args)
    run_inference(pipeline, args)

    trace_path = (
        args.ckpt.replace("/", "_")
        + f"bf16@{not args.no_bf16}-sdpa@{not args.no_sdpa}-bs@{args.batch_size}-fuse@{args.enable_fused_projections}-upcast_vae@{args.upcast_vae}-steps@{args.num_inference_steps}-unet@{args.compile_unet}-vae@{args.compile_vae}-mode@{args.compile_mode}-change_comp_config@{args.change_comp_config}-do_quant@{args.do_quant}-device@{args.device}.json"
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
