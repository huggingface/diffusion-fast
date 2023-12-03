import torch


torch.set_float32_matmul_precision("high")

import sys  # noqa: E402


sys.path.append(".")
from utils.benchmarking_utils import (  # noqa: E402
    benchmark_fn,
    bytes_to_giga_bytes,
    create_parser,
    generate_csv_dict,
    write_to_csv,
)
from utils.pipeline_utils import CKPT_ID, PROMPT, load_pipeline  # noqa: E402


# def apply_dynamic_quant_fn(m):
#     from torchao.quantization.dynamic_quant import DynamicallyPerAxisQuantizedLinear
#     from torchao.quantization.quant_api import replace_with_custom_fn_if_matches_filter
#     from torchao.quantization.weight_only import WeightOnlyInt8QuantLinear

#     def from_float(mod):
#         if hasattr(mod, "lora_layer"):
#             assert mod.lora_layer is None
#         if mod.weight.size(1) == 1280 and mod.weight.size(0) == 1280:
#             return WeightOnlyInt8QuantLinear.from_float(mod)
#         if mod.weight.size(1) == 5120 and mod.weight.size(0) == 1280:
#             return DynamicallyPerAxisQuantizedLinear.from_float(mod)
#         return mod

#     replace_with_custom_fn_if_matches_filter(
#         m,
#         from_float,
#         lambda mod, fqn: isinstance(mod, torch.nn.Linear),
#     )


def run_inference(pipe, args):
    _ = pipe(
        prompt=PROMPT,
        num_inference_steps=args.num_inference_steps,
        num_images_per_prompt=args.batch_size,
    )


def main(args) -> dict:
    pipeline = load_pipeline(args)

    # Warmup.
    run_inference(pipeline, args)
    run_inference(pipeline, args)
    run_inference(pipeline, args)

    time = benchmark_fn(run_inference, pipeline, args)  # in seconds.
    memory = bytes_to_giga_bytes(torch.cuda.max_memory_allocated())  # in GBs.

    data_dict = generate_csv_dict(
        pipeline_cls=str(pipeline.__class__.__name__),
        ckpt=CKPT_ID,
        args=args,
        time=time,
        memory=memory,
    )
    return data_dict


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    if not args.compile_unet:
        args.compile_mode = "NA"

    data_dict = main(args)

    name = (
        CKPT_ID.replace("/", "_")
        + f"fp16@{not args.no_fp16}-sdpa@{not args.no_sdpa}-bs@{args.batch_size}-fuse@{args.enable_fused_projections}-upcast_vae@{args.upcast_vae}-steps@{args.num_inference_steps}-unet@{args.compile_unet}-vae@{args.compile_vae}-mode@{args.compile_mode}-change_comp_config@{args.change_comp_config}-do_quant@{args.do_quant}.csv"
    )
    write_to_csv(name, data_dict)
