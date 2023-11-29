# sdxl-fast
Faster generation with SDXL.

Docker image: `spsayakpaul/sdxl-fast-compile:nightly`

Use it by running (assuming on your base machine you have CUDA, Docker, etc. configured):

```bash
nvidia-docker run -it --user root --rm spsayakpaul/sdxl-fast-compile:nightly
```

## Some commands

**For building and pushing the Docker image (assuming you're authenticated already via `docker login`)**:

```bash
docker build -f Dockerfile -t spsayakpaul/sdxl-fast-compile:nightly --compress .
docker push spsayakpaul/sdxl-fast-compile:nightly
```

_(Change `spsayakpaul/sdxl-fast-compile:nightly` accordingly)_

**For bulk-launching benchmark runs and pushing a nice plot**:

```bash
python run_benchmark.py && \
    python run_benchmark.py --run_compile && \
    python run_benchmark.py --run_compile --compile_mode=max-autotune && \
    python run_benchmark.py --run_compile --compile_mode=max-autotune --change_comp_config && \
    python prepare_plot.py --push_to_hub
```

Specifying `--push_to_hub` requires you to run `huggingface-cli login` before hand. 

_(Change the `REPO_ID` variable in the `prepare_plot.py` script accordingly)_

To run benchmarking with `--do_quant` install `torchao` first:

```bash
git clone https://github.com/pytorch-labs/ao
cd ao
python setup.py install
```

Then run:

```bash
python run_benchmark.py --run_compile --compile_mode=max-autotune --do_quant
```

**For bulk-profiling kernel traces**:

```bash
python run_profile.py && \
    python run_profile.py --run_compile && \
    python run_profile.py --run_compile --compile_mode=max-autotune && \
    python run_profile.py --run_compile --compile_mode=max-autotune --change_comp_config
```