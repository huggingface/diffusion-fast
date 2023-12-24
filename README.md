# Diffusion, fast

Repository for the blog post: **Accelerating Generative AI Part III: Diffusion, Fast** (TODO).

<div align="center">

<img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/final-results-diffusion-fast/SDXL%2C_Batch_Size%3A_1%2C_Steps%3A_30.png" width=500>

</div>

Summary of the optimizations:

* Running with the bfloat16 precision
* `scaled_dot_product_attention` (SPDA)
* `torch.compile`
* Combining q,k,v projections for attention computation
* Dynamic int8 quantization 

## Setup 🛠️

We rely on pure PyTorch for the optimizations. You can refer to the [Dockerfile](./Dockerfile) to get the complete development environment setup. 

For hardware, we used an 80GB 400W A100 GPU with its memory clock set to the maximum rate.

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

</details>

<details>
  <summary>scaled_dot_product_attention</summary>

```python
from diffusers import StableDiffusionXLPipeline

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
  <summary>torch.compile</summary>

```python
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16
).to("cuda")

# Compile the UNet and VAE.
pipe.unet = torch.compile(pipe.unet, mode="max-autotune", fullgraph=True)
pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# First call to `pipe` will be slow, subsequent ones will be faster.
image = pipe(prompt).images[0]
```

<div align="center">

<img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/progressive-acceleration-sdxl/SDXL%2C_Batch_Size%3A_1%2C_Steps%3A_30_3.png" width=500>

</div>

</details>