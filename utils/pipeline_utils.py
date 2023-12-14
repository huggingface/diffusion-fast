import torch
from torchao.quantization import (
    change_linear_weights_to_int4_woqtensors,
    change_linear_weights_to_int8_dqtensors,
    change_linear_weights_to_int8_woqtensors,
    swap_conv2d_1x1_to_linear,
)

from diffusers import AutoencoderKL, DiffusionPipeline


def dynamic_quant_filter_fn(mod, *args):
    return (
        isinstance(mod, torch.nn.Linear)
        and mod.in_features > 16
        and (mod.in_features, mod.out_features)
        not in [
            (320, 640),
            (320, 1280),
            (2816, 1280),
            (1280, 640),
            (1280, 320),
            (512, 512),
            (512, 1536),
            (2048, 2560),
            (2048, 1280),
        ]
    )


CKPT_ID = "stabilityai/stable-diffusion-xl-base-1.0"
PROMPT = "ghibli style, a fantasy landscape with castles"

# torch._inductor.config.fx_graph_cache = True # speeds up recompile, may reduce performance


def load_pipeline(args):
    """Loads the SDXL pipeline."""

    if args.do_quant and not args.compile_unet:
        raise ValueError("Compilation for UNet must be enabled when quantizing.")
    if args.do_quant and not args.compile_vae:
        raise ValueError("Compilation for VAE must be enabled when quantizing.")

    dtype = torch.float32 if args.no_bf16 else torch.bfloat16
    print(f"Using dtype: {dtype}")
    pipe = DiffusionPipeline.from_pretrained(CKPT_ID, torch_dtype=dtype, use_safetensors=True)

    if not args.upcast_vae:
        pipe.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype)

    if args.enable_fused_projections:
        print("Enabling fused QKV projections for both UNet and VAE.")
        pipe.fuse_qkv_projections()

    if args.upcast_vae:
        print("Upcasting VAE.")
        pipe.upcast_vae()

    if args.no_sdpa:
        pipe.unet.set_default_attn_processor()
        pipe.vae.set_default_attn_processor()

    pipe = pipe.to("cuda")

    if args.compile_unet:
        pipe.unet.to(memory_format=torch.channels_last)
        print("Compile UNet")
        swap_conv2d_1x1_to_linear(pipe.unet)
        if args.compile_mode == "max-autotune" and args.change_comp_config:
            torch._inductor.config.conv_1x1_as_mm = True
            torch._inductor.config.coordinate_descent_tuning = True
            torch._inductor.config.epilogue_fusion = False
            torch._inductor.config.coordinate_descent_check_all_directions = True

        if args.do_quant:
            print("Apply quantization to UNet")
            if args.do_quant == "int4weightonly":
                change_linear_weights_to_int4_woqtensors(pipe.unet)
            elif args.do_quant == "int8weightonly":
                change_linear_weights_to_int8_woqtensors(pipe.unet)
            elif args.do_quant == "int8dynamic":
                change_linear_weights_to_int8_dqtensors(pipe.unet, dynamic_quant_filter_fn)
            else:
                raise ValueError(f"Unknown do_quant value: {args.do_quant}.")
            torch._inductor.config.force_fuse_int_mm_with_mul = True
            torch._inductor.config.use_mixed_mm = True

        pipe.unet = torch.compile(pipe.unet, mode=args.compile_mode, fullgraph=True)

    if args.compile_vae:
        pipe.vae.to(memory_format=torch.channels_last)
        print("Compile VAE")
        swap_conv2d_1x1_to_linear(pipe.vae)

        if args.compile_mode == "max-autotune" and args.change_comp_config:
            torch._inductor.config.conv_1x1_as_mm = True
            torch._inductor.config.coordinate_descent_tuning = True
            torch._inductor.config.epilogue_fusion = False
            torch._inductor.config.coordinate_descent_check_all_directions = True

        if args.do_quant:
            print("Apply quantization to VAE")
            if args.do_quant == "int4weightonly":
                change_linear_weights_to_int4_woqtensors(pipe.vae)
            elif args.do_quant == "int8weightonly":
                change_linear_weights_to_int8_woqtensors(pipe.vae)
            elif args.do_quant == "int8dynamic":
                change_linear_weights_to_int8_dqtensors(pipe.vae, dynamic_quant_filter_fn)
            else:
                raise ValueError(f"Unknown do_quant value: {args.do_quant}.")
            torch._inductor.config.force_fuse_int_mm_with_mul = True
            torch._inductor.config.use_mixed_mm = True

        pipe.vae.decode = torch.compile(pipe.vae.decode, mode=args.compile_mode, fullgraph=True)

    pipe.set_progress_bar_config(disable=True)
    return pipe
