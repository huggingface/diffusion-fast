import torch

from diffusers import AutoencoderKL, DiffusionPipeline


CKPT_ID = "stabilityai/stable-diffusion-xl-base-1.0"
PROMPT = "ghibli style, a fantasy landscape with castles"


def apply_dynamic_quant_fn(m):
    """Applies weight-only and dynamic quantization in a selective manner."""
    from torchao.quantization.dynamic_quant import DynamicallyPerAxisQuantizedLinear
    from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter
    from torchao.quantization.weight_only import WeightOnlyInt8QuantLinear

    def from_float(mod):
        if hasattr(mod, "lora_layer"):
            assert mod.lora_layer is None
        # if mod.weight.size(1) == 1280 and mod.weight.size(0) == 1280:
        #     return WeightOnlyInt8QuantLinear.from_float(mod)
        # if mod.weight.size(1) == 640 and mod.weight.size(0) == 640:
        #     return WeightOnlyInt8QuantLinear.from_float(mod)
        if mod.weight.size(1) == 5120 and mod.weight.size(0) == 1280:
            return DynamicallyPerAxisQuantizedLinear.from_float(mod)
        # if mod.weight.size(1) == 2560 and mod.weight.size(0) == 640:
        #     return DynamicallyPerAxisQuantizedLinear.from_float(mod)
        return mod

    _replace_with_custom_fn_if_matches_filter(
        m,
        from_float,
        lambda mod, fqn: isinstance(mod, torch.nn.Linear),
    )


def load_pipeline(args):
    """Loads the SDXL pipeline."""
    dtype = torch.float32 if args.no_fp16 else torch.float16
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

        if args.compile_mode == "max-autotune" and args.change_comp_config:
            torch._inductor.config.conv_1x1_as_mm = True
            torch._inductor.config.coordinate_descent_tuning = True

        if args.do_quant:
            print("Apply quantization to UNet")
            apply_dynamic_quant_fn(pipe.unet)
            torch._inductor.config.force_fuse_int_mm_with_mul = True

        if args.compile_mode == "max-autotune":
            pipe.unet = torch.compile(pipe.unet, mode=args.compile_mode)
        else:
            pipe.unet = torch.compile(pipe.unet, mode=args.compile_mode, fullgraph=True)

    if args.compile_vae:
        pipe.vae.to(memory_format=torch.channels_last)
        print("Compile VAE")

        if args.compile_mode == "max-autotune" and args.change_comp_config:
            torch._inductor.config.conv_1x1_as_mm = True
            torch._inductor.config.coordinate_descent_tuning = True

        if args.do_quant:
            print("Apply quantization to VAE")
            apply_dynamic_quant_fn(pipe.vae)
            torch._inductor.config.force_fuse_int_mm_with_mul = True

        if args.compile_mode == "max-autotune":
            pipe.vae.decode = torch.compile(pipe.vae.decode, mode=args.compile_mode)
        else:
            pipe.vae.decode = torch.compile(pipe.vae.decode, mode=args.compile_mode, fullgraph=True)

    pipe.set_progress_bar_config(disable=True)
    return pipe
