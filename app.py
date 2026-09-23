import os
import random
import base64
import json
from io import BytesIO
from typing import List, Optional

import gradio as gr
from gradio import Server
from fastapi.responses import HTMLResponse
import numpy as np
import spaces
import torch
from PIL import Image
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

MAX_SEED = np.iinfo(np.int32).max
MAX_REF_IMAGES = 10
MAX_GPU_SECONDS = 120
MIN_GPU_SECONDS = 30

DEFAULT_MODE = "None"
TURBO_MODE = "4-Step Turbo (Viggle)"
OBJECT_MOVER_MODE = "Object Mover"
OBJECT_REMOVER_MODE = "Object Remover"
CUSTOM_MODE = "Custom LoRA"

MODE_CHOICES = [DEFAULT_MODE, TURBO_MODE, OBJECT_MOVER_MODE, OBJECT_REMOVER_MODE, CUSTOM_MODE]

LORA_PRESETS = {
    TURBO_MODE: (TURBO_REPO, TURBO_LORA_FILE, TURBO_ADAPTER_NAME),
    OBJECT_MOVER_MODE: (OBJECT_MOVER_REPO, OBJECT_MOVER_FILE, OBJECT_MOVER_ADAPTER),
    OBJECT_REMOVER_MODE: (OBJECT_REMOVER_REPO, OBJECT_REMOVER_FILE, OBJECT_REMOVER_ADAPTER),
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
ASPECTS = ["auto", "1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16"]

RGBA_PREFIX = "This is an RGBA image with transparency. "
RGBA_SUFFIX = " The image has alpha channel and the background is transparent."

print(f"Loading {MODEL_ID} ...", flush=True)
pipe = QwenImage21Pipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
print("Pipeline loaded.", flush=True)

print(f"Loading turbo LoRA from {TURBO_REPO} ...", flush=True)
pipe.load_lora_weights(TURBO_REPO, weight_name=TURBO_LORA_FILE, adapter_name=TURBO_ADAPTER_NAME, token=HF_TOKEN)
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

_TEXT_ENCODER_EVICTION = {"armed": False}

def _vram_note(stage):
    free, total = torch.cuda.mem_get_info()
    print(f"[vram] {stage}: driver free={free / 2**30:.1f}G of {total / 2**30:.1f}G", flush=True)

def _release_text_encoder(module, args):
    if _TEXT_ENCODER_EVICTION["armed"]:
        _TEXT_ENCODER_EVICTION["armed"] = False
        pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        _vram_note("after text encoder eviction")

for component in (pipe.vae, pipe.transformer):
    component.register_forward_pre_hook(_release_text_encoder)

def _custom_adapter_name(repo, weight_name):
    key = f"{repo}::{weight_name or ''}"
    return "custom_" + str(abs(hash(key)) % 10_000_000)

def _ensure_lora_loaded(adapter_name, repo, weight_name):
    if adapter_name in _LOADED_ADAPTERS:
        return
    print(f"Lazily loading LoRA '{adapter_name}' from {repo} ...", flush=True)
    try:
        pipe.load_lora_weights(repo, weight_name=weight_name or None, adapter_name=adapter_name, token=HF_TOKEN)
    except Exception as exc:
        raise gr.Error(f"Could not load LoRA from '{repo}': {exc}") from exc
    _LOADED_ADAPTERS.add(adapter_name)
    print(f"LoRA '{adapter_name}' loaded.", flush=True)

def _gpu_duration(output_resolution, references, num_inference_steps):
    steps = int(num_inference_steps)
    if int(output_resolution) >= 2048 and bool(references):
        seconds = 25 + 1.6 * steps
    elif int(output_resolution) >= 2048:
        seconds = 18 + 2.4 * steps
    else:
        seconds = 15 + 1.2 * steps
    return int(min(MAX_GPU_SECONDS, max(MIN_GPU_SECONDS, seconds)))

def _run_pipeline(prompt, refs_pil, width, height, output_resolution, num_inference_steps, negative_prompt, guidance_scale, seed, turbo, lora):
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
    
    generator = torch.Generator("cuda").manual_seed(int(seed))
    kwargs = {
        "prompt": prompt,
        "num_inference_steps": int(num_inference_steps),
        "generator": generator,
        "output_resolution": int(output_resolution),
    }
    if width and height:
        kwargs["width"] = int(width)
        kwargs["height"] = int(height)
    if refs_pil:
        kwargs["image"] = refs_pil if len(refs_pil) > 1 else refs_pil[0]
    if negative_prompt and guidance_scale > 1:
        kwargs["negative_prompt"] = negative_prompt
        kwargs["true_cfg_scale"] = float(guidance_scale)

    try:
        image = pipe(**kwargs).images[0]
        return image
    finally:
        _TEXT_ENCODER_EVICTION["armed"] = False

@spaces.GPU(duration=_gpu_duration)
def _generate_standard(prompt, refs_pil, width, height, output_resolution, num_inference_steps, negative_prompt, guidance_scale, seed, turbo, lora):
    return _run_pipeline(prompt, refs_pil, width, height, output_resolution, num_inference_steps, negative_prompt, guidance_scale, seed, turbo, lora)

@spaces.GPU(duration=_gpu_duration, size="xlarge")
def _generate_full_gpu(prompt, refs_pil, width, height, output_resolution, num_inference_steps, negative_prompt, guidance_scale, seed, turbo, lora):
    return _run_pipeline(prompt, refs_pil, width, height, output_resolution, num_inference_steps, negative_prompt, guidance_scale, seed, turbo, lora)

def b64_to_pil_list(b64_json_str: str) -> List[Image.Image]:
    if not b64_json_str or b64_json_str.strip() in ("", "[]"):
        return []
    try:
        b64_list = json.loads(b64_json_str)
    except Exception:
        return []
    pil_images = []
    for b64_str in b64_list:
        if not b64_str or not isinstance(b64_str, str): continue
        try:
            if b64_str.startswith("data:image"):
                _, data = b64_str.split(",", 1)
            else:
                data = b64_str
            image_data = base64.b64decode(data)
            pil_images.append(Image.open(BytesIO(image_data)).convert("RGBA"))
        except Exception:
            pass
    return pil_images

def pil_to_b64_png(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

# ── Gradio Server (Server mode) ────────────
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
) -> dict:
    """Generates or edits an image using Qwen-Image-2.1"""
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
        lora_config = {"repo": repo, "weight_name": weight_name, "adapter_name": _custom_adapter_name(repo, weight_name)}

    refs_pil = b64_to_pil_list(images_b64_json)[:MAX_REF_IMAGES]

    if transparent:
        if RGBA_PREFIX.strip() not in prompt:
            prompt = RGBA_PREFIX + prompt
        if "alpha channel" not in prompt:
            prompt = prompt + RGBA_SUFFIX

    width, height = None, None
    if aspect_ratio in TIERS.get(tier, {}):
        width, height = TIERS[tier][aspect_ratio]
    elif not refs_pil:
        width, height = TIERS[tier]["1:1"]
        
    output_resolution = TIER_RESOLUTION.get(tier, 1024)

    seed = int(seed) if seed is not None else 0
    if randomize_seed or seed < 0:
        seed = random.randint(0, MAX_SEED)
    seed = seed % (MAX_SEED + 1)

    runner = _generate_full_gpu if (int(output_resolution) >= 2048 and bool(refs_pil)) else _generate_standard
    image = runner(
        prompt, refs_pil, width, height, output_resolution,
        int(num_inference_steps), (negative_prompt or "").strip(),
        float(guidance_scale), seed, turbo, lora_config
    )

    run_mode = "edit" if refs_pil else "text-to-image"
    size = f"{width}x{height}" if width else f"{image.width}x{image.height} (from reference)"
    gpu = "full GPU" if runner is _generate_full_gpu else "half GPU"
    has_alpha = "RGBA" if image.mode == "RGBA" else "RGB"
    lora_tag = lora_config["adapter_name"] if lora_config else "none"
    info = (
        f"{run_mode} | {size} | {int(num_inference_steps)} steps | seed {seed} | "
        f"cfg {float(guidance_scale):.1f} | {gpu} | {has_alpha} | mode: {mode} | lora: {lora_tag}\n{prompt}"
    )
    return {"image": pil_to_b64_png(image), "seed": seed, "info": info}

@app.get("/api/config")
def client_config():
    return {
        "lora_modes": MODE_CHOICES,
        "tiers": list(TIERS.keys()),
        "aspects": ASPECTS,
        "default_mode": DEFAULT_MODE,
        "default_tier": "1K (fast)",
        "default_aspect": "auto",
        "default_steps": 40,
        "default_cfg": 1.0
    }

@app.get("/", response_class=HTMLResponse)
async def homepage():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    app.launch(show_error=True, mcp_server=True)