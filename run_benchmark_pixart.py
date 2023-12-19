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
from utils.pipeline_utils_pixart import CKPT_ID, PROMPT, load_pipeline  # noqa: E402


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
    img = pipeline(
        prompt=PROMPT,
        num_inference_steps=args.num_inference_steps,
        num_images_per_prompt=args.batch_size,
    ).images[0]

    return data_dict, img


if __name__ == "__main__":
    parser = create_parser(is_pixart=True)
    args = parser.parse_args()
    print(args)

    data_dict, img = main(args)

    name = (
        CKPT_ID.replace("/", "_")
        + f"bf16@{not args.no_bf16}-sdpa@{not args.no_sdpa}-bs@{args.batch_size}-fuse@{args.enable_fused_projections}-upcast_vae@{args.upcast_vae}-steps@{args.num_inference_steps}-transformer@{args.compile_transformer}-vae@{args.compile_vae}-mode@{args.compile_mode}-change_comp_config@{args.change_comp_config}-do_quant@{args.do_quant}-tag@{args.tag}.csv"
    )
    img.save(f"{name}.jpeg")
    write_to_csv(name, data_dict, is_pixart=True)
