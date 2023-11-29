import torch 
torch.set_float32_matmul_precision("high")

from diffusers import DiffusionPipeline # noqa: E402
import argparse # noqa: E402

import sys # noqa: E402
sys.path.append(".")
from utils import benchmark_fn, bytes_to_giga_bytes, generate_csv_dict, write_to_csv # noqa: E402

CKPT_ID = "stabilityai/stable-diffusion-xl-base-1.0"
PROMPT = "ghibli style, a fantasy landscape with castles"


def load_pipeline(run_compile=False, compile_mode="reduce-overhead"):
    pipe = DiffusionPipeline.from_pretrained(CKPT_ID, torch_dtype=torch.float16, use_safetensors=True)
    pipe = pipe.to("cuda")

    if run_compile:
        pipe.unet.to(memory_format=torch.channels_last)
        print("Run torch compile")
        if compile_mode == "max-autotune":
            torch._inductor.config.conv_1x1_as_mm = True 
            torch._inductor.config.coordinate_descent_tuning = True 
        pipe.unet = torch.compile(pipe.unet, mode=compile_mode, fullgraph=True)

    pipe.set_progress_bar_config(disable=True)
    return pipe

def run_inference(pipe, args):
    _ = pipe(
        prompt=PROMPT,
        num_inference_steps=args.num_inference_steps,
        num_images_per_prompt=args.batch_size,
    )


def main(args) -> dict:
    pipeline = load_pipeline(run_compile=args.run_compile)

    time = benchmark_fn(run_inference, pipeline, args)  # in seconds.
    memory = bytes_to_giga_bytes(torch.cuda.max_memory_allocated())  # in GBs.
    
    data_dict = generate_csv_dict(
        pipeline_cls=str(pipeline.__class__.__name__), ckpt=CKPT_ID, args=args, time=time, memory=memory,
    )
    return data_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--run_compile", action="store_true")
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead", choices=["reduce-overhead", "max-autotune"])
    args = parser.parse_args()

    if not args.run_compile:
        args.compile_mode = "NA"
    
    data_dict = main(args)

    name = (
        CKPT_ID.replace("/", "_")
        + f"-bs@{args.batch_size}-steps@{args.num_inference_steps}-compile@{args.run_compile}-mode@{args.compile_mode}.csv"
    )
    write_to_csv(name, data_dict)
    