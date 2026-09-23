"""Qwen-Image-2.1 demo for Hugging Face ZeroGPU.

Text-to-image, multi-reference image editing, and native RGBA generation with a
single `QwenImage21Pipeline`. The pipeline is placed on `cuda` at import time,
which is how ZeroGPU makes the weights resident: CUDA is emulated outside
`@spaces.GPU` and the runtime moves the tensors onto the real GPU when a
decorated function is called.

Generation mode: "None (no LoRA)" runs the base Qwen-Image-2.1 checkpoint at
whatever step count the user picks. "4-Step Turbo" activates the
Viggle/Qwen-Image-2.1-viggle-turbo LoRA (r64) on top of the same transformer
and forces the distilled 4-step schedule with CFG off; it is loaded once at
startup (eagerly) and left disabled. "Object Mover" and "Object Remover"
(both prithivMLmods bbox-preview LoRAs) and "Custom LoRA" (any repo the user
types in) are loaded lazily: the weights are only downloaded and injected
into the pipeline the first time that mode is actually used, inside the
`@spaces.GPU`-decorated call, since that's the only place the pipeline's
modules live on a real GPU. Once loaded, an adapter stays resident for the
rest of the session and later requests just flip which adapter is active.
"""

import os
import random

import gradio as gr
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
HF_TOKEN = os.environ.get("HF_TOKEN")  # only needed if a LoRA repo is private/gated

OBJECT_MOVER_REPO = "prithivMLmods/Qwen-Image-2.1-Object-Mover-Bbox-Preview"
OBJECT_MOVER_FILE = "Qwen-Image-2.1-Object-Mover-Bbox-Preview-5000.safetensors"
OBJECT_MOVER_ADAPTER = "object_mover"

OBJECT_REMOVER_REPO = "prithivMLmods/Qwen-Image-2.1-Object-Remover-Bbox-Preview"
OBJECT_REMOVER_FILE = "Qwen-Image-2.1-Object-Remover-Bbox-Preview-5000.safetensors"
OBJECT_REMOVER_ADAPTER = "object_remover"

MAX_SEED = np.iinfo(np.int32).max
MAX_REF_IMAGES = 10
# ZeroGPU rejects any request above 120s of GPU time.
MAX_GPU_SECONDS = 120
MIN_GPU_SECONDS = 30

DEFAULT_MODE = "None (no LoRA)"
TURBO_MODE = "4-Step Turbo (Viggle)"
OBJECT_MOVER_MODE = "Object Mover (prithivMLmods)"
OBJECT_REMOVER_MODE = "Object Remover (prithivMLmods)"
CUSTOM_MODE = "Custom LoRA (enter repo below)"

MODE_CHOICES = [DEFAULT_MODE, TURBO_MODE, OBJECT_MOVER_MODE, OBJECT_REMOVER_MODE, CUSTOM_MODE]

# Fixed presets: repo, weight file, adapter name. Custom mode builds its own tuple
# from the textboxes at request time, so it isn't listed here.
LORA_PRESETS = {
    TURBO_MODE: (TURBO_REPO, TURBO_LORA_FILE, TURBO_ADAPTER_NAME),
    OBJECT_MOVER_MODE: (OBJECT_MOVER_REPO, OBJECT_MOVER_FILE, OBJECT_MOVER_ADAPTER),
    OBJECT_REMOVER_MODE: (OBJECT_REMOVER_REPO, OBJECT_REMOVER_FILE, OBJECT_REMOVER_ADAPTER),
}

# Output sizes. Every value is a multiple of 32 because the pipeline packs
# 2x2 latent patches on top of a 16x spatial compression factor.
TIERS = {
    "1K (fast)": {
        "1:1": (1024, 1024),
        "4:3": (1184, 864),
        "3:4": (864, 1184),
        "3:2": (1248, 832),
        "2:3": (832, 1248),
        "16:9": (1344, 768),
        "9:16": (768, 1344),
    },
    "2K (native)": {
        "1:1": (2048, 2048),
        "4:3": (2400, 1792),
        "3:4": (1792, 2400),
        "3:2": (2528, 1696),
        "2:3": (1696, 2528),
        "16:9": (2752, 1536),
        "9:16": (1536, 2752),
    },
}
TIER_RESOLUTION = {"1K (fast)": 1024, "2K (native)": 2048}
ASPECTS = ["auto (reference or 1:1)", "1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16"]

# The checkpoint reads the alpha channel from these phrases, so transparency is
# requested through the prompt instead of a pipeline flag.
RGBA_PREFIX = "This is an RGBA image with transparency. "
RGBA_SUFFIX = " The image has alpha channel and the background is transparent."

print(f"Loading {MODEL_ID} ...", flush=True)
pipe = QwenImage21Pipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
print("Pipeline loaded.", flush=True)

print(f"Loading turbo LoRA from {TURBO_REPO} ...", flush=True)
pipe.load_lora_weights(
    TURBO_REPO,
    weight_name=TURBO_LORA_FILE,
    adapter_name=TURBO_ADAPTER_NAME,
    token=HF_TOKEN,
)
pipe.disable_lora()  # inactive by default; _run_pipeline enables it per request when turbo mode is picked
print("Turbo LoRA loaded (inactive).", flush=True)

# Turbo is the only adapter loaded eagerly. Object Mover, Object Remover, and any
# custom repo the user types in are loaded on first use inside a GPU-decorated call
# (see _ensure_lora_loaded) and then stay resident for the rest of the session.
_LOADED_ADAPTERS = {TURBO_ADAPTER_NAME}

# The stock Qwen-Image-2.1 scheduler config carries shift_terminal=0.02, which stretches the
# last sigma node to 0.02 instead of 0 and silently costs the final step. The turbo student was
# distilled against the unstretched schedule, so keep a second scheduler for turbo mode and swap
# between the two per request rather than mutating one shared config.
default_scheduler = pipe.scheduler
turbo_scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config, shift_terminal=None)

# transformers' Qwen3VLVisionPatchEmbed runs an nn.Conv3d whose kernel_size equals its stride, which is
# exactly a linear map over each flattened patch. cuDNN has no usable bf16 Conv3d kernel for this shape and
# falls back to one that costs ~30 s per reference image (measured; 356 ms even with memory to spare, versus
# 2.9 ms for the same conv in fp32). The matmul below is 0.8 ms and its output is bitwise equal to the conv's.
patch_embed = pipe.text_encoder.model.visual.patch_embed


def _patch_embed_forward(hidden_states):
    proj = patch_embed.proj
    weight = proj.weight.reshape(proj.weight.shape[0], -1)
    return hidden_states.to(weight.dtype).flatten(1) @ weight.T + proj.bias


patch_embed.forward = _patch_embed_forward


def _custom_adapter_name(repo, weight_name):
    """A stable, unique adapter name per (repo, weight_name) pair, so trying a second
    custom repo doesn't collide with (or silently replace) the first one loaded."""
    key = f"{repo}::{weight_name or ''}"
    return "custom_" + str(abs(hash(key)) % 10_000_000)


def _ensure_lora_loaded(adapter_name, repo, weight_name):
    """Downloads and injects a LoRA the first time it's selected; a no-op after that.

    Must run inside a `@spaces.GPU`-decorated call: outside of one, `cuda` is emulated
    and the pipeline's real weights aren't there yet, so PEFT would have nothing real
    to attach the new adapter's layers to.
    """
    if adapter_name in _LOADED_ADAPTERS:
        return
    print(f"Lazily loading LoRA '{adapter_name}' from {repo} ...", flush=True)
    try:
        pipe.load_lora_weights(
            repo,
            weight_name=weight_name or None,
            adapter_name=adapter_name,
            token=HF_TOKEN,
        )
    except Exception as exc:  # noqa: BLE001 - surface hub/format errors to the UI, not a traceback
        raise gr.Error(f"Could not load LoRA from '{repo}': {exc}") from exc
    _LOADED_ADAPTERS.add(adapter_name)
    print(f"LoRA '{adapter_name}' loaded.", flush=True)


def _gpu_duration(
    prompt,
    references,
    width,
    height,
    output_resolution,
    num_inference_steps,
    negative_prompt,
    guidance_scale,
    seed,
    turbo,
    lora,
):
    """Seconds to reserve, from the cost of the request.

    ZeroGPU charges quota for the reserved window rather than the wall clock
    used, so ask for close to the real cost and never exceed the 120s ceiling.
    Turbo mode already arrives here with num_inference_steps forced to 4, so
    the formula below needs no separate turbo branch. `lora` only changes
    which weights run, not the token/step counts, so it isn't priced either;
    a lazily-loaded LoRA's one-time download happens inside the reserved
    window on its first use, same as any other cold-start cost.
    """
    steps = int(num_inference_steps)
    if _needs_full_gpu(output_resolution, references):
        # Twice the SMs, but the reference tokens ride along with the 2K target.
        seconds = 25 + 1.6 * steps
    elif int(output_resolution) >= 2048:
        # 4x the latent tokens at half the SMs, plus a 4MP VAE decode.
        seconds = 18 + 2.4 * steps
    else:
        seconds = 15 + 1.2 * steps
    return int(min(MAX_GPU_SECONDS, max(MIN_GPU_SECONDS, seconds)))


# The text encoder is idle once the prompt is encoded: it is 17GB of the 30GB of
# resident weights, and nothing else needs it until the next run's prompt. Park it
# in host RAM for every run. It is released on the first component forward after
# encoding, which is the reference VAE encode for an edit and the denoiser for
# text-to-image, so the release lands before the memory-heavy phases either way.
_TEXT_ENCODER_EVICTION = {"armed": False}


def _vram_note(stage):
    free, total = torch.cuda.mem_get_info()
    print(
        f"[vram] {stage}: driver free={free / 2**30:.1f}G of {total / 2**30:.1f}G, "
        f"allocated={torch.cuda.memory_allocated() / 2**30:.1f}G, "
        f"reserved={torch.cuda.memory_reserved() / 2**30:.1f}G, "
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


def _needs_full_gpu(output_resolution, references):
    """A 2K edit carries the reference tokens alongside the 2K target, and that does
    not fit the 47GB slice a ZeroGPU `large` request gets even with the text encoder
    parked. Everything else runs on `large`."""
    return int(output_resolution) >= 2048 and bool(references)


def _run_pipeline(
    prompt,
    references,
    width,
    height,
    output_resolution,
    num_inference_steps,
    negative_prompt,
    guidance_scale,
    seed,
    turbo,
    lora,
):
    # `lora` is None for the no-LoRA mode, or a dict describing which adapter should be
    # active for this request. It is (lazily) loaded if needed and then made the only
    # active adapter; every other mode just disables all adapters. Turbo additionally
    # swaps in the unstretched scheduler it was distilled against. This runs inside the
    # ZeroGPU-decorated call, same as the text encoder move below, so both the adapter
    # toggle and any first-time weight load land before the transformer forward.
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
    if references:
        images = [Image.open(path).convert("RGBA") for path in references]
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


@spaces.GPU(duration=_gpu_duration)
def _generate(
    prompt,
    references,
    width,
    height,
    output_resolution,
    num_inference_steps,
    negative_prompt,
    guidance_scale,
    seed,
    turbo,
    lora,
):
    return _run_pipeline(
        prompt,
        references,
        width,
        height,
        output_resolution,
        num_inference_steps,
        negative_prompt,
        guidance_scale,
        seed,
        turbo,
        lora,
    )


@spaces.GPU(duration=_gpu_duration, size="xlarge")
def _generate_full_gpu(
    prompt,
    references,
    width,
    height,
    output_resolution,
    num_inference_steps,
    negative_prompt,
    guidance_scale,
    seed,
    turbo,
    lora,
):
    return _run_pipeline(
        prompt,
        references,
        width,
        height,
        output_resolution,
        num_inference_steps,
        negative_prompt,
        guidance_scale,
        seed,
        turbo,
        lora,
    )


def _reference_path(item):
    """Gallery items reach the callback as a path, a (path, caption) pair, or FileData."""
    if isinstance(item, (tuple, list)):
        item = item[0] if item else None
    if isinstance(item, dict):
        item = item.get("path")
    return getattr(item, "path", item)


def generate(
    prompt,
    negative_prompt,
    references,
    tier,
    aspect_ratio,
    num_inference_steps,
    guidance_scale,
    transparent,
    seed,
    randomize_seed,
    mode,
    custom_lora_repo,
    custom_lora_weight,
    progress=gr.Progress(track_tqdm=True),
):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Enter a prompt.")

    turbo = mode == TURBO_MODE
    if turbo:
        # Steps and CFG are fixed by the distillation recipe; ignore whatever the
        # (disabled) sliders currently hold so a stale value can't sneak through.
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
            raise gr.Error(
                "Enter a custom LoRA repo id, e.g. prithivMLmods/Qwen-Image-2.1-Natural-Exposure-LoRA."
            )
        weight_name = (custom_lora_weight or "").strip() or None
        lora_config = {
            "repo": repo,
            "weight_name": weight_name,
            "adapter_name": _custom_adapter_name(repo, weight_name),
        }
    # DEFAULT_MODE (or anything unrecognized) leaves lora_config as None: no adapter active.

    refs = [_reference_path(item) for item in (references or [])]
    refs = [path for path in refs if path][:MAX_REF_IMAGES]
    if references and len(references) > MAX_REF_IMAGES:
        gr.Warning(f"Using the first {MAX_REF_IMAGES} reference images.")
    # Uploads live in a temp directory that ZeroGPU wipes when the app restarts, so a
    # Gallery can outlive its files. Fail before the GPU reservation instead of in it.
    missing = [path for path in refs if not os.path.exists(path)]
    if missing:
        raise gr.Error("A reference image is no longer on the server. Upload it again and retry.")

    if transparent:
        if RGBA_PREFIX.strip() not in prompt:
            prompt = RGBA_PREFIX + prompt
        if "alpha channel" not in prompt:
            prompt = prompt + RGBA_SUFFIX

    width, height = None, None
    if aspect_ratio in TIERS[tier]:
        width, height = TIERS[tier][aspect_ratio]
    elif not refs:
        # Nothing to derive an aspect ratio from, so fall back to the square preset.
        width, height = TIERS[tier]["1:1"]
    output_resolution = TIER_RESOLUTION[tier]

    seed = int(seed) if seed is not None else 0
    if randomize_seed or seed < 0:
        seed = random.randint(0, MAX_SEED)
    seed = seed % (MAX_SEED + 1)

    progress(0.01, desc="Encoding prompt")
    runner = _generate_full_gpu if _needs_full_gpu(output_resolution, refs) else _generate
    image = runner(
        prompt,
        refs,
        width,
        height,
        output_resolution,
        int(num_inference_steps),
        (negative_prompt or "").strip(),
        float(guidance_scale),
        seed,
        turbo,
        lora_config,
    )

    run_mode = "edit" if refs else "text-to-image"
    size = f"{width}x{height}" if width else f"{image.width}x{image.height} (from reference)"
    gpu = "full GPU" if runner is _generate_full_gpu else "half GPU"
    has_alpha = "RGBA" if image.mode == "RGBA" else "RGB"
    lora_tag = lora_config["adapter_name"] if lora_config else "none"
    info = (
        f"{run_mode} | {size} | {int(num_inference_steps)} steps | seed {seed} | "
        f"cfg {float(guidance_scale):.1f} | {gpu} | {has_alpha} | mode: {mode} | lora: {lora_tag}\n{prompt}"
    )
    return image, info


def _on_mode_change(mode):
    turbo = mode == TURBO_MODE
    is_custom = mode == CUSTOM_MODE
    if turbo:
        steps_update = gr.update(value=TURBO_STEPS, interactive=False, info="Fixed at 4 steps for the turbo student.")
        guidance_update = gr.update(value=1.0, interactive=False, info="Turbo is distilled without CFG.")
        negative_update = gr.update(interactive=False)
    else:
        steps_update = gr.update(value=40, interactive=True, info="40 is the official default.")
        guidance_update = gr.update(
            value=1.0, interactive=True, info="1.0 disables classifier-free guidance, as Qwen-Image-2.1 recommends."
        )
        negative_update = gr.update(interactive=True)
    custom_group_update = gr.update(visible=is_custom)
    return steps_update, guidance_update, negative_update, custom_group_update


EXAMPLES = [
    ['A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, reflections on wet pavement'],
    ["A cute cartoon dragon sticker, thick outlines, flat pastel colors"],
    ["Change the background to a sunset beach, keep the subject unchanged"],
    [
        "These three characters are sitting around a campfire in a forest, "
        "cinematic lighting, shallow depth of field"
    ],
]

with gr.Blocks(title="Qwen-Image-2.1-LoRAs-PnP") as demo:
    gr.Markdown(
        "# Qwen-Image-2.1-LoRAs-PnP\n"
        "Unified text-to-image generation and image editing, running on ZeroGPU. "
        "Add reference images to switch to editing mode; leave them empty for text-to-image.\n\n"
        "[Model](https://huggingface.co/Qwen/Qwen-Image-2.1) | "
        "[Blog](https://qwen.ai/blog?id=qwen-image-2.1) | "
        "[GitHub](https://github.com/QwenLM/Qwen-Image-2.1) | "
        "[Turbo LoRA](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo) | "
        "[Object Mover LoRA](https://huggingface.co/prithivMLmods/Qwen-Image-2.1-Object-Mover-Bbox-Preview) | "
        "[Object Remover LoRA](https://huggingface.co/prithivMLmods/Qwen-Image-2.1-Object-Remover-Bbox-Preview)"
    )
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt",
                lines=4,
                value=EXAMPLES[0][0],
                placeholder="Describe the image, or how to edit the references.",
            )
            references = gr.Gallery(
                label=f"Reference images (up to {MAX_REF_IMAGES}, empty for text-to-image)",
                type="filepath",
                columns=5,
                height=180,
            )
            negative_prompt = gr.Textbox(
                label="Negative prompt (used only when guidance > 1, disabled in turbo mode)",
                lines=2,
            )
            with gr.Row():
                tier = gr.Dropdown(
                    label="Resolution",
                    choices=list(TIERS),
                    value="1K (fast)",
                    info="2K with reference images needs the full GPU and costs double the quota.",
                )
                aspect_ratio = gr.Dropdown(
                    label="Aspect ratio",
                    choices=ASPECTS,
                    value=ASPECTS[0],
                )
            with gr.Row():
                num_inference_steps = gr.Slider(
                    label="Steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=40,
                    info="40 is the official default.",
                )
                guidance_scale = gr.Slider(
                    label="True CFG scale",
                    minimum=1.0,
                    maximum=4.0,
                    step=0.1,
                    value=1.0,
                    info="1.0 disables classifier-free guidance, as Qwen-Image-2.1 recommends.",
                )
            with gr.Row():
                seed = gr.Number(label="Seed", value=0, precision=0)
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            transparent = gr.Checkbox(
                label="Transparent background (RGBA)",
                value=False,
                info="Wraps the prompt in the format the checkpoint expects for alpha output.",
            )
            with gr.Group():
                gr.Markdown("**LoRA / Settings**")
                mode = gr.Radio(
                    label="Generation mode",
                    choices=MODE_CHOICES,
                    value=DEFAULT_MODE,
                    info=(
                        "None runs the base checkpoint with no LoRA. Turbo forces steps=4, CFG off. "
                        "Object Mover / Object Remover and Custom LoRA are downloaded on first use only."
                    ),
                )
                with gr.Group(visible=False) as custom_lora_group:
                    custom_lora_repo = gr.Textbox(
                        label="Custom LoRA repo id",
                        placeholder="e.g. prithivMLmods/Qwen-Image-2.1-Natural-Exposure-LoRA",
                    )
                    custom_lora_weight = gr.Textbox(
                        label="Weight file name (optional)",
                        placeholder="Leave blank to auto-detect a single .safetensors file in the repo",
                    )
            run_button = gr.Button("Generate", variant="primary")
            gr.Examples(examples=EXAMPLES, inputs=[prompt], cache_examples=False)
        with gr.Column(scale=1):
            result = gr.Image(
                label="Result",
                type="pil",
                format="png",
                height=640,
            )
            status = gr.Textbox(label="Run info", lines=3)

    mode.change(
        fn=_on_mode_change,
        inputs=[mode],
        outputs=[num_inference_steps, guidance_scale, negative_prompt, custom_lora_group],
    )

    run_button.click(
        fn=generate,
        inputs=[
            prompt,
            negative_prompt,
            references,
            tier,
            aspect_ratio,
            num_inference_steps,
            guidance_scale,
            transparent,
            seed,
            randomize_seed,
            mode,
            custom_lora_repo,
            custom_lora_weight,
        ],
        outputs=[result, status],
    )

    gr.Markdown(
        "ZeroGPU is shared: without an account you get 2 minutes of GPU time per day, "
        "with a free account 5 minutes, with PRO 40 minutes. Every run reserves its window up "
        "front and ZeroGPU charges that reserved time times 1.5, so a 1K image at 40 steps costs "
        "about 95 seconds of quota and the same image at 8 steps costs 45; 4-Step Turbo runs at a "
        "fixed 4 steps and reserves close to the 30s floor. A 2K edit with reference images runs "
        "on the full GPU and is charged double regardless of mode. LoRAs other than Turbo are "
        "downloaded the first time you select them, which adds a one-time delay to that run's "
        "GPU window; every later run is as fast as usual. Nothing is stored and the Space keeps "
        "nothing between sessions."
    )

demo.queue().launch()