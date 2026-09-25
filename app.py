"""Qwen-Image-2.1 demo for Hugging Face ZeroGPU.
Text-to-image, multi-reference image editing, and native RGBA generation with a
single `QwenImage21Pipeline`. The pipeline is placed on `cuda` at import time,
which is how ZeroGPU makes the weights resident: CUDA is emulated outside
`@spaces.GPU` and the runtime moves the tensors onto the real GPU when a
decorated function is called.
"""

import os
import gc
import random
import base64
import json
from io import BytesIO
from typing import List, Tuple

import numpy as np
import spaces
import torch
from PIL import Image
import gradio as gr
from gradio import Server
from fastapi.responses import HTMLResponse
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen-Image-2.1")
TURBO_REPO = os.environ.get("TURBO_REPO", "Viggle/Qwen-Image-2.1-viggle-turbo")
TURBO_LORA_FILE = "Qwen-Image-2.1-viggle-turbo-4step-lora-r64.safetensors"
TURBO_ADAPTER_NAME = "turbo"
TURBO_STEPS = 4
HF_TOKEN = os.environ.get("HF_TOKEN")

OBJECT_MOVER_REPO = "prithivMLmods/Qwen-Image-2.1-Object-Mover-Bbox-Preview"
OBJECT_MOVER_FILE = "Qwen-Image-2.1-Object-Mover-Bbox-Preview-5000.safetensors"
OBJECT_MOVER_ADAPTER = "object_mover"

OBJECT_REMOVER_REPO = "prithivMLmods/Qwen-Image-2.1-Object-Remover-Bbox-Preview"
OBJECT_REMOVER_FILE = "Qwen-Image-2.1-Object-Remover-Bbox-Preview-5000.safetensors"
OBJECT_REMOVER_ADAPTER = "object_remover"

NATURAL_EXPOSURE_REPO = "prithivMLmods/Qwen-Image-2.1-Natural-Exposure-LoRA"
NATURAL_EXPOSURE_FILE = "Qwen-Image-2.1-Natural-Exposure-LoRA-4000.safetensors"
NATURAL_EXPOSURE_ADAPTER = "natural_exposure"

MAX_SEED = np.iinfo(np.int32).max
MAX_REF_IMAGES = 10

DEFAULT_MODE = "None (no LoRA)"
TURBO_MODE = "4-Step Turbo (Viggle)"
OBJECT_MOVER_MODE = "Object Mover (prithivMLmods)"
OBJECT_REMOVER_MODE = "Object Remover (prithivMLmods)"
NATURAL_EXPOSURE_MODE = "Natural Exposure (prithivMLmods)"
CUSTOM_MODE = "Custom LoRA (enter repo below)"

MODE_CHOICES = [DEFAULT_MODE, TURBO_MODE, OBJECT_MOVER_MODE, OBJECT_REMOVER_MODE, NATURAL_EXPOSURE_MODE, CUSTOM_MODE]

LORA_PRESETS = {
    TURBO_MODE: (TURBO_REPO, TURBO_LORA_FILE, TURBO_ADAPTER_NAME),
    OBJECT_MOVER_MODE: (OBJECT_MOVER_REPO, OBJECT_MOVER_FILE, OBJECT_MOVER_ADAPTER),
    OBJECT_REMOVER_MODE: (OBJECT_REMOVER_REPO, OBJECT_REMOVER_FILE, OBJECT_REMOVER_ADAPTER),
    NATURAL_EXPOSURE_MODE: (NATURAL_EXPOSURE_REPO, NATURAL_EXPOSURE_FILE, NATURAL_EXPOSURE_ADAPTER),
}

TIERS = {
    "1K (fast)": {
        "1:1": (1024, 1024), "4:3": (1184, 864), "3:4": (864, 1184),
        "3:2": (1248, 832), "2:3": (832, 1248), "16:9": (1344, 768), "9:16": (768, 1344),
    },
    "2K (native)": {
        "1:1": (2048, 2048), "4:3": (2400, 1792), "3:4": (1792, 2400),
        "3:2": (2528, 1696), "2:3": (1696, 2528), "16:9": (2752, 1536), "9:16": (1536, 2752),
    },
}
TIER_RESOLUTION = {"1K (fast)": 1024, "2K (native)": 2048}
ASPECTS = ["auto (reference or 1:1)", "1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16"]

RGBA_PREFIX = "This is an RGBA image with transparency. "
RGBA_SUFFIX = " The image has alpha channel and the background is transparent."

print(f"Loading {MODEL_ID} ...", flush=True)
pipe = QwenImage21Pipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
print("Pipeline loaded.", flush=True)

print(f"Loading turbo LoRA from {TURBO_REPO} ...", flush=True)
pipe.load_lora_weights(
    TURBO_REPO, weight_name=TURBO_LORA_FILE, adapter_name=TURBO_ADAPTER_NAME, token=HF_TOKEN,
)
pipe.disable_lora()
print("Turbo LoRA loaded (inactive).", flush=True)

_LOADED_ADAPTERS = {TURBO_ADAPTER_NAME}

default_scheduler = pipe.scheduler
turbo_scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config, shift_terminal=None)

patch_embed = pipe.text_encoder.model.visual.patch_embed

def _patch_embed_forward(hidden_states):
    proj = patch_embed.proj
    weight = proj.weight.reshape(proj.weight.shape[0], -1)
    return hidden_states.to(weight.dtype).flatten(1) @ weight.T + proj.bias

patch_embed.forward = _patch_embed_forward

def _custom_adapter_name(repo, weight_name):
    key = f"{repo}::{weight_name or ''}"
    return "custom_" + str(abs(hash(key)) % 10_000_000)

def _ensure_lora_loaded(adapter_name, repo, weight_name):
    if adapter_name in _LOADED_ADAPTERS:
        return
    print(f"Lazily loading LoRA '{adapter_name}' from {repo} ...", flush=True)
    try:
        pipe.load_lora_weights(
            repo, weight_name=weight_name or None, adapter_name=adapter_name, token=HF_TOKEN,
        )
    except Exception as exc:
        raise gr.Error(f"Could not load LoRA from '{repo}': {exc}") from exc
    _LOADED_ADAPTERS.add(adapter_name)
    print(f"LoRA '{adapter_name}' loaded.", flush=True)

def _needs_full_gpu(output_resolution, references):
    return int(output_resolution) >= 2048 and bool(references)

_TEXT_ENCODER_EVICTION = {"armed": False}

def _vram_note(stage):
    free, total = torch.cuda.mem_get_info()
    print(
        f"[vram] {stage}: driver free={free / 2**30:.1f}G of {total / 2**30:.1f}G, "
        f"allocated={torch.cuda.memory_allocated() / 2**30:.1f}G, "
        f"peak={torch.cuda.max_memory_allocated() / 2**30:.1f}G",
        flush=True,
    )

def _release_text_encoder(module, args):
    if _TEXT_ENCODER_EVICTION["armed"]:
        _TEXT_ENCODER_EVICTION["armed"] = False
        pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        _vram_note("after text encoder eviction")

for component in (pipe.vae, pipe.transformer):
    component.register_forward_pre_hook(_release_text_encoder)

def _run_pipeline(
    prompt, pil_references, width, height, output_resolution, num_inference_steps,
    negative_prompt, guidance_scale, seed, turbo, lora
):
    if lora:
        _ensure_lora_loaded(lora["adapter_name"], lora["repo"], lora.get("weight_name"))
        pipe.enable_lora()
        pipe.set_adapters([lora["adapter_name"]], adapter_weights=[1.0])
    else:
        pipe.disable_lora()
    pipe.scheduler = turbo_scheduler if turbo else default_scheduler

    encoder_device = next(pipe.text_encoder.parameters()).device
    if encoder_device.type != "cuda":
        pipe.text_encoder.to("cuda")
    _TEXT_ENCODER_EVICTION["armed"] = True
    torch.cuda.reset_peak_memory_stats()
    lora_tag = lora["adapter_name"] if lora else "none"
    _vram_note(
        f"start {output_resolution}px, lora={lora_tag}, turbo={turbo}, "
        f"text encoder was on {encoder_device}"
    )

    generator = torch.Generator("cuda").manual_seed(int(seed))
    kwargs = {
        "prompt": prompt,
        "num_inference_steps": int(num_inference_steps),
        "generator": generator,
        "output_resolution": int(output_resolution),
    }
    if width:
        kwargs["width"] = int(width)
        kwargs["height"] = int(height)
    if pil_references:
        images = [img.convert("RGBA") for img in pil_references]
        kwargs["image"] = images if len(images) > 1 else images[0]
    if negative_prompt and guidance_scale > 1:
        kwargs["negative_prompt"] = negative_prompt
        kwargs["true_cfg_scale"] = float(guidance_scale)

    try:
        image = pipe(**kwargs).images[0]
        _vram_note("end of pipeline call")
        return image
    finally:
        _TEXT_ENCODER_EVICTION["armed"] = False

def _gpu_duration_func(
    prompt, pil_references, width, height, output_resolution, num_inference_steps,
    negative_prompt, guidance_scale, seed, turbo, lora, gpu_duration
):
    return int(gpu_duration)

@spaces.GPU(duration=_gpu_duration_func)
def _generate(
    prompt, pil_references, width, height, output_resolution, num_inference_steps,
    negative_prompt, guidance_scale, seed, turbo, lora, gpu_duration
):
    return _run_pipeline(
        prompt, pil_references, width, height, output_resolution, num_inference_steps,
        negative_prompt, guidance_scale, seed, turbo, lora
    )

@spaces.GPU(duration=_gpu_duration_func, size="xlarge")
def _generate_full_gpu(
    prompt, pil_references, width, height, output_resolution, num_inference_steps,
    negative_prompt, guidance_scale, seed, turbo, lora, gpu_duration
):
    return _run_pipeline(
        prompt, pil_references, width, height, output_resolution, num_inference_steps,
        negative_prompt, guidance_scale, seed, turbo, lora
    )

def b64_to_pil_list(b64_json_str):
    if not b64_json_str or b64_json_str.strip() in ("", "[]"):
        return []
    try:
        b64_list = json.loads(b64_json_str)
    except Exception:
        return []
    pil_images = []
    for b64_str in b64_list:
        if not b64_str or not isinstance(b64_str, str):
            continue
        try:
            if b64_str.startswith("data:image"):
                _, data = b64_str.split(",", 1)
            else:
                data = b64_str
            image_data = base64.b64decode(data)
            pil_images.append(Image.open(BytesIO(image_data)).convert("RGBA"))
        except Exception as e:
            print(f"Error decoding image: {e}")
    return pil_images

def pil_to_b64_png(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

def generate(
    prompt, negative_prompt, images_b64_json, tier, aspect_ratio, num_inference_steps,
    guidance_scale, transparent, seed, randomize_seed, mode, custom_lora_repo, custom_lora_weight, gpu_duration
):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Enter a prompt.")

    turbo = mode == TURBO_MODE
    if turbo:
        num_inference_steps = TURBO_STEPS
        guidance_scale = 1.0
        negative_prompt = ""

    lora_config = None
    if mode in LORA_PRESETS:
        repo, weight_name, adapter_name = LORA_PRESETS[mode]
        lora_config = {"repo": repo, "weight_name": weight_name, "adapter_name": adapter_name}
    elif mode == CUSTOM_MODE:
        repo = (custom_lora_repo or "").strip()
        if not repo:
            raise gr.Error("Enter a custom LoRA repo id.")
        weight_name = (custom_lora_weight or "").strip() or None
        lora_config = {
            "repo": repo, "weight_name": weight_name,
            "adapter_name": _custom_adapter_name(repo, weight_name),
        }

    pil_refs = b64_to_pil_list(images_b64_json)[:MAX_REF_IMAGES]
    if len(pil_refs) > MAX_REF_IMAGES:
        gr.Warning(f"Using the first {MAX_REF_IMAGES} reference images.")

    if transparent:
        if RGBA_PREFIX.strip() not in prompt:
            prompt = RGBA_PREFIX + prompt
        if "alpha channel" not in prompt:
            prompt = prompt + RGBA_SUFFIX

    width, height = None, None
    if aspect_ratio in TIERS.get(tier, {}):
        width, height = TIERS[tier][aspect_ratio]
    elif not pil_refs:
        width, height = TIERS[tier]["1:1"]
    output_resolution = TIER_RESOLUTION[tier]

    seed = int(seed) if seed is not None else 0
    if randomize_seed or seed < 0:
        seed = random.randint(0, MAX_SEED)
    seed = seed % (MAX_SEED + 1)

    runner = _generate_full_gpu if _needs_full_gpu(output_resolution, pil_refs) else _generate
    image = runner(
        prompt, pil_refs, width, height, output_resolution, int(num_inference_steps),
        (negative_prompt or "").strip(), float(guidance_scale), seed, turbo, lora_config, int(gpu_duration)
    )

    run_mode = "edit" if pil_refs else "text-to-image"
    size = f"{width}x{height}" if width else f"{image.width}x{image.height} (from reference)"
    gpu = "full GPU" if runner is _generate_full_gpu else "half GPU"
    has_alpha = "RGBA" if image.mode == "RGBA" else "RGB"
    lora_tag = lora_config["adapter_name"] if lora_config else "none"
    info = (
        f"{run_mode} | {size} | {int(num_inference_steps)} steps | seed {seed} | "
        f"cfg {float(guidance_scale):.1f} | {gpu} | {has_alpha} | mode: {mode} | lora: {lora_tag} | "
        f"gpu_duration: {gpu_duration}s\n{prompt}"
    )
    return image, info, seed

app = Server(title="Qwen-Image-2.1-LoRAs-PnP")

@app.mcp.tool(name="generate_image")
@app.api(name="generate_image")
def infer(
    prompt: str,
    negative_prompt: str,
    images_b64_json: str,
    tier: str,
    aspect_ratio: str,
    num_inference_steps: int,
    guidance_scale: float,
    transparent: bool,
    seed: int,
    randomize_seed: bool,
    mode: str,
    custom_lora_repo: str,
    custom_lora_weight: str,
    gpu_duration: int,
) -> dict:
    """Generates an image using Qwen-Image-2.1 with PnP LoRAs."""
    gc.collect()
    torch.cuda.empty_cache()

    image, info, used_seed = generate(
        prompt, negative_prompt, images_b64_json, tier, aspect_ratio, num_inference_steps,
        guidance_scale, transparent, seed, randomize_seed, mode, custom_lora_repo, custom_lora_weight, gpu_duration
    )
    
    return {
        "image": pil_to_b64_png(image),
        "info": info,
        "seed": used_seed
    }

@app.get("/api/config")
def client_config():
    return {"app_name": "Qwen-Image-2.1-LoRAs-PnP"}

@app.get("/", response_class=HTMLResponse)
async def homepage():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    app.launch(show_error=True, mcp_server=True)