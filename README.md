---
title: Z Image Turbo Local
emoji: ⚡
colorFrom: yellow
colorTo: yellow
sdk: gradio
sdk_version: 6.0.1
app_file: app.py
pinned: true
---

# Z-Image-Turbo Local Runner

This repository contains a local-friendly Gradio app for running [`Tongyi-MAI/Z-Image-Turbo`](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) on your PC.

The original Hugging Face Space uses ZeroGPU-specific helpers. This version removes those hosted-only requirements and adds the dependency and launch instructions needed for a local machine.

## Hardware requirements

- **Recommended:** NVIDIA GPU with at least **16 GB VRAM**.
- **Supported but slower:** CPU or Apple Silicon MPS. Use smaller image sizes such as `512x512` if you are not on CUDA.
- The first run downloads the model weights from Hugging Face, so you need a stable internet connection and enough disk space for the model cache.

## Setup on Windows, Linux, or macOS

> Use Python 3.10 or 3.11. Python 3.12 may work, but many AI packages still have the best compatibility on 3.10/3.11.

```bash
python -m venv .venv
```

### Activate the virtual environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you have a different CUDA version or want CPU-only PyTorch, install the correct PyTorch wheel from <https://pytorch.org/get-started/locally/> before running `pip install -r requirements.txt`.

## Run locally

```bash
python app.py
```

Open <http://127.0.0.1:7860> in your browser.

## Useful environment variables

You can customize the runtime without editing code:

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL_ID` | `Tongyi-MAI/Z-Image-Turbo` | Hugging Face model repo to load. |
| `DEVICE` | `cuda` when available, otherwise `cpu` | Device passed to PyTorch and Diffusers. Use `cpu`, `cuda`, or `mps`. |
| `TORCH_DTYPE` | `bfloat16` on CUDA, otherwise `float32` | Torch dtype name. Use `float16` if your GPU does not support bfloat16. |
| `GRADIO_SERVER_NAME` | `127.0.0.1` | Set to `0.0.0.0` to expose on your LAN. |
| `GRADIO_SERVER_PORT` | `7860` | Local web server port. |

Examples:

```bash
DEVICE=cpu TORCH_DTYPE=float32 python app.py
```

```powershell
$env:DEVICE="cuda"; $env:TORCH_DTYPE="float16"; python app.py
```

## Notes

- Generation uses `guidance_scale=0.0`, matching the Turbo model's recommended classifier-free-guidance-free setup.
- The app outputs PNG images for better quality than JPEG.
- If you run out of memory, reduce height/width to `512` or `768` and restart the app.
