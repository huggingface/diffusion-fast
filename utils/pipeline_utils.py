import torch
from torchao.quantization import (
    apply_dynamic_quant,
    change_linear_weights_to_int4_woqtensors,
    change_linear_weights_to_int8_woqtensors,
    swap_conv2d_1x1_to_linear,
)

from diffusers import AutoencoderKL, DiffusionPipeline, DPMSolverMultistepScheduler


PROMPT = "ghibli style, a fantasy landscape with castles"


def dynamic_quant_filter_fn(mod, *args):
    return (
        isinstance(mod, torch.nn.Linear)
        and mod.in_features > 16
        and (mod.in_features, mod.out_features)
        not in [
            (1280, 640),
            (1920, 1280),
            (1920, 640),
            (2048, 1280),
            (2048, 2560),
            (2560, 1280),
            (256, 128),
            (2816, 1280),
            (320, 640),
            (512, 1536),
            (512, 256),
            (512, 512),
            (640, 1280),
            (640, 1920),
            (640, 320),
            (640, 5120),
            (640, 640),
            (960, 320),
            (960, 640),
        ]
    )


def conv_filter_fn(mod, *args):
    return (
        isinstance(mod, torch.nn.Conv2d) and mod.kernel_size == (1, 1) and 128 in [mod.in_channels, mod.out_channels]
    )


def load_pipeline(
    ckpt: str,
    compile_unet: bool,
    compile_vae: bool,
    no_sdpa: bool,
    no_bf16: bool,
    upcast_vae: bool,
    enable_fused_projections: bool,
    do_quant: bool,
    compile_mode: str,
    change_comp_config: bool,
):
    """Loads the SDXL pipeline."""

    if do_quant and not compile_unet:
        raise ValueError("Compilation for UNet must be enabled when quantizing.")
    if do_quant and not compile_vae:
        raise ValueError("Compilation for VAE must be enabled when quantizing.")

    dtype = torch.float32 if no_bf16 else torch.bfloat16
    print(f"Using dtype: {dtype}")

    if ckpt != "runwayml/stable-diffusion-v1-5":
        pipe = DiffusionPipeline.from_pretrained(ckpt, torch_dtype=dtype, use_safetensors=True)
    else:
        pipe = DiffusionPipeline.from_pretrained(ckpt, torch_dtype=dtype, use_safetensors=True, safety_checker=None)
        # As the default scheduler of SD v1-5 doesn't have sigmas device placement
        # (https://github.com/huggingface/diffusers/pull/6173)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if not upcast_vae and ckpt != "runwayml/stable-diffusion-v1-5":
        print("Using a more numerically stable VAE.")
        pipe.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype)

    if enable_fused_projections:
        print("Enabling fused QKV projections for both UNet and VAE.")
        pipe.fuse_qkv_projections()

    if upcast_vae and ckpt != "runwayml/stable-diffusion-v1-5":
        print("Upcasting VAE.")
        pipe.upcast_vae()

    if no_sdpa:
        print("Using vanilla attention.")
        pipe.unet.set_default_attn_processor()
        pipe.vae.set_default_attn_processor()

    pipe = pipe.to("cuda")

    if compile_unet:
        pipe.unet.to(memory_format=torch.channels_last)
        print("Compile UNet.")
        swap_conv2d_1x1_to_linear(pipe.unet, conv_filter_fn)
        if compile_mode == "max-autotune" and change_comp_config:
            torch._inductor.config.conv_1x1_as_mm = True
            torch._inductor.config.coordinate_descent_tuning = True
            torch._inductor.config.epilogue_fusion = False
            torch._inductor.config.coordinate_descent_check_all_directions = True

        if do_quant:
            print("Apply quantization to UNet.")
            if do_quant == "int4weightonly":
                change_linear_weights_to_int4_woqtensors(pipe.unet)
            elif do_quant == "int8weightonly":
                change_linear_weights_to_int8_woqtensors(pipe.unet)
            elif do_quant == "int8dynamic":
                apply_dynamic_quant(pipe.unet, dynamic_quant_filter_fn)
            else:
                raise ValueError(f"Unknown do_quant value: {do_quant}.")
            torch._inductor.config.force_fuse_int_mm_with_mul = True
            torch._inductor.config.use_mixed_mm = True

        pipe.unet = torch.compile(pipe.unet, mode=compile_mode, fullgraph=True)

    if compile_vae:
        pipe.vae.to(memory_format=torch.channels_last)
        print("Compile VAE.")
        swap_conv2d_1x1_to_linear(pipe.vae, conv_filter_fn)

        if compile_mode == "max-autotune" and change_comp_config:
            torch._inductor.config.conv_1x1_as_mm = True
            torch._inductor.config.coordinate_descent_tuning = True
            torch._inductor.config.epilogue_fusion = False
            torch._inductor.config.coordinate_descent_check_all_directions = True

        if do_quant:
            print("Apply quantization to VAE.")
            if do_quant == "int4weightonly":
                change_linear_weights_to_int4_woqtensors(pipe.vae)
            elif do_quant == "int8weightonly":
                change_linear_weights_to_int8_woqtensors(pipe.vae)
            elif do_quant == "int8dynamic":
                apply_dynamic_quant(pipe.vae, dynamic_quant_filter_fn)
            else:
                raise ValueError(f"Unknown do_quant value: {do_quant}.")
            torch._inductor.config.force_fuse_int_mm_with_mul = True
            torch._inductor.config.use_mixed_mm = True

        pipe.vae.decode = torch.compile(pipe.vae.decode, mode=compile_mode, fullgraph=True)

    pipe.set_progress_bar_config(disable=True)
    return pipe
