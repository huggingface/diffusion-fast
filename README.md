# Diffusion, fast

Repository for the blog post: [**Accelerating Generative AI Part III: Diffusion, Fast**](https://pytorch.org/blog/accelerating-generative-ai-3/). You can find a run down of the techniques on the [🤗 Diffusers website](https://huggingface.co/docs/diffusers/main/en/tutorials/fast_diffusion) too. 

<div align="center">

<img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/final-results-diffusion-fast/SDXL%2C_Batch_Size%3A_1%2C_Steps%3A_30.png" width=500>

</div><br>

> [!WARNING]  
> This repository relies on the `torchao` package for all things quantization. Since the first version of this repo, the `torchao` package has changed its APIs significantly. More specifically, [this](https://github.com/huggingface/diffusion-fast/blob/f4fa861422d9819226eb2ceac247c85c3547130d/Dockerfile#L30) version was used to obtain the numbers in this repository. For more updated usage of `torchao`, please refer to the [`diffusers-torchao`](https://github.com/sayakpaul/diffusers-torchao) repository.

Summary of the optimizations:

* Running with the bfloat16 precision
* `scaled_dot_product_attention` (SDPA)
* `torch.compile`
* Combining q,k,v projections for attention computation
* Dynamic int8 quantization 

These techniques are fairly generalizable to other pipelines too, as we show below.

Table of contents:

* [Setup](#setup-🛠️)
* [Running benchmarking experiments](#running-a-benchmarking-experiment-🏎️)
* [Code](#improvements-progressively-📈-📊)
* [Results from other pipelines](#results-from-other-pipelines-🌋)

## Setup 🛠️

We rely on pure PyTorch for the optimizations. You can refer to the [Dockerfile](./Dockerfile) to get the complete development environment setup. 

For hardware, we used an 80GB 400W A100 GPU with its memory clock set to the maximum rate (1593 in our case).

Meanwhile, these optimizations (BFloat16, SDPA, torch.compile, Combining q,k,v projections) can run on CPU platforms as well, and bring 4x latency improvement to Stable Diffusion XL (SDXL) on 4th Gen Intel® Xeon® Scalable processors.

## Running a benchmarking experiment 🏎️

[`run_benchmark.py`](./run_benchmark.py) is the main script for benchmarking the different optimization techniques. After an experiment has been done, you should expect to see two files:

* A `.csv` file with all the benchmarking numbers.
* A `.jpeg` image file corresponding to the experiment. 

Refer to the [`experiment-scripts/run_sd.sh`](./experiment-scripts/run_sd.sh) for some reference experiment commands. 

**Notes on running PixArt-Alpha experiments**:

* Use the [`run_experiment_pixart.py`](./run_benchmark_pixart.py) for this.
* Uninstall the current installation of `diffusers` and re-install it again like so: `pip install git+https://github.com/huggingface/diffusers@fuse-projections-pixart`.
* Refer to the [`experiment-scripts/run_pixart.sh`](./experiment-scripts/run_pixart.sh) script for some reference experiment commands.

_(Support for PixArt-Alpha is experimental.)_

You can use the [`prepare_results.py`](./prepare_results.py) script to generate a consolidated CSV file and a plot to visualize the results from it. This is best used after you have run a couple of benchmarking experiments already and have their corresponding CSV files.

The script also supports CPU platforms, you can refer to the [`experiment-scripts/run_sd_cpu.sh`](./experiment-scripts/run_sd_cpu.sh) for some reference experiment commands. 

To run the script, you need the following dependencies:

* pandas
* matplotlib
* seaborn

## Improvements, progressively 📈 📊

<details>
  <summary>Baseline</summary>

```python
from diffusers import StableDiffusionXLPipeline

# Load the pipeline in full-precision and place its model components on CUDA.
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0"
).to("cuda")

# Run the attention ops without efficiency.
pipe.unet.set_default_attn_processor()
pipe.vae.set_default_attn_processor()

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt, num_inference_steps=30).images[0]
```

With this, we're at:

<div align="center">

<img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/progressive-acceleration-sdxl/SDXL%2C_Batch_Size%3A_1%2C_Steps%3A_30_0.png" width=500>

</div>

</details>

<details>
  <summary>Bfloat16</summary>

```python
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
	"stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16
).to("cuda")

# Run the attention ops without efficiency.
pipe.unet.set_default_attn_processor()
pipe.vae.set_default_attn_processor()

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt, num_inference_steps=30).images[0]
```

<div align="center">

<img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/progressive-acceleration-sdxl/SDXL%2C_Batch_Size%3A_1%2C_Steps%3A_30_1.png" width=500>

</div>

> 💡 We later ran the experiments in float16 and found out that the recent versions of `torchao` do not incur numerical problems from float16.

</details>

<details>
  <summary>scaled_dot_product_attention</summary>

```python
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
	"stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16
).to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt, num_inference_steps=30).images[0]
```

<div align="center">

<img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/progressive-acceleration-sdxl/SDXL%2C_Batch_Size%3A_1%2C_Steps%3A_30_2.png" width=500>

</div>

</details>

<details>
  <summary>torch.compile</summary><br>

First, configure some compiler flags:

```python
from diffusers import StableDiffusionXLPipeline
import torch

# Set the following compiler flags to make things go brrr.
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True
```

Then load the pipeline:

```python
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16
).to("cuda")
```

Compile and perform inference:

```python
# Compile the UNet and VAE.
pipe.unet.to(memory_format=torch.channels_last)
pipe.vae.to(memory_format=torch.channels_last)
pipe.unet = torch.compile(pipe.unet, mode="max-autotune", fullgraph=True)
pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# First call to `pipe` will be slow, subsequent ones will be faster.
image = pipe(prompt, num_inference_steps=30).images[0]
```

<div align="center">

<img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/progressive-acceleration-sdxl/SDXL%2C_Batch_Size%3A_1%2C_Steps%3A_30_3.png" width=500>

</div>

</details>

<details>
  <summary>Combining attention projection matrices</summary><br>

```python
from diffusers import StableDiffusionXLPipeline
import torch

# Configure the compiler flags.
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16
).to("cuda")

# Combine attention projection matrices.
pipe.fuse_qkv_projections()

# Compile the UNet and VAE.
pipe.unet.to(memory_format=torch.channels_last)
pipe.vae.to(memory_format=torch.channels_last)
pipe.unet = torch.compile(pipe.unet, mode="max-autotune", fullgraph=True)
pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# First call to `pipe` will be slow, subsequent ones will be faster.
image = pipe(prompt, num_inference_steps=30).images[0]
```

<div align="center">

<img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/progressive-acceleration-sdxl/SDXL%2C_Batch_Size%3A_1%2C_Steps%3A_30_4.png" width=500>

</div>

</details>

<details>
  <summary>Dynamic quantization</summary><br>

Start by setting the compiler flags (this time, we have two new):

```python
from diffusers import StableDiffusionXLPipeline
import torch

from torchao.quantization import apply_dynamic_quant, swap_conv2d_1x1_to_linear

# Compiler flags. There are two new.
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True
torch._inductor.config.force_fuse_int_mm_with_mul = True
torch._inductor.config.use_mixed_mm = True
```

Then write the filtering functions to apply dynamic quantization:

```python
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
```

Then we're rwady for inference:

```python
pipe = StableDiffusionXLPipeline.from_pretrained(
	"stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16
).to("cuda")

# Combine attention projection matrices.
pipe.fuse_qkv_projections()

# Change the memory layout.
pipe.unet.to(memory_format=torch.channels_last)
pipe.vae.to(memory_format=torch.channels_last)

# Swap the pointwise convs with linears.
swap_conv2d_1x1_to_linear(pipe.unet, conv_filter_fn)
swap_conv2d_1x1_to_linear(pipe.vae, conv_filter_fn)

# Apply dynamic quantization.
apply_dynamic_quant(pipe.unet, dynamic_quant_filter_fn)
apply_dynamic_quant(pipe.vae, dynamic_quant_filter_fn)

# Compile.
pipe.unet = torch.compile(pipe.unet, mode="max-autotune", fullgraph=True)
pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt, num_inference_steps=30).images[0]
```

<div align="center">

<img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/progressive-acceleration-sdxl/SDXL%2C_Batch_Size%3A_1%2C_Steps%3A_30_5.png" width=500>

</div>

</details>

## Results from other pipelines 🌋

<details>
  <summary>SSD-1B</summary>

<div align="center">

<img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/final-results-diffusion-fast/SSD-1B%2C_Batch_Size%3A_1%2C_Steps%3A_30.png" width=500>
<br><sup><a href="https://huggingface.co/segmind/SSD-1B">segmind/SSD-1B</a></sup>

</div>

</details>

<details>
  <summary>SD v1-5</summary>

<div align="center">

<img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/final-results-diffusion-fast/SD_v1-5%2C_Batch_Size%3A_1%2C_Steps%3A_30.png" width=500>
<br><sup><a href="https://huggingface.co/runwayml/stable-diffusion-v1-5">runwayml/stable-diffusion-v1-5</a></sup>

</div>

</details>

<details>
  <summary>Pixart-Alpha</summary>

<div align="center">

<img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/final-results-diffusion-fast/PixArt-%24%5Calpha%24%2C_Batch_Size%3A_1%2C_Steps%3A_30.png" width=500>
<br><sup><a href="https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS">PixArt-alpha/PixArt-XL-2-1024-MS</a></sup>

</div>

</details>
