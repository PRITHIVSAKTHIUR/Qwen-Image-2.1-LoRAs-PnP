"""Qwen-Image-2.1 demo for Hugging Face ZeroGPU.

Text-to-image, multi-reference image editing, and native RGBA generation with a
single `QwenImage21Pipeline`. The pipeline is placed on `cuda` at import time,
which is how ZeroGPU makes the weights resident: CUDA is emulated outside
`@spaces.GPU` and the runtime moves the tensors onto the real GPU when a
decorated function is called.
"""

import os
import random

import gradio as gr
import numpy as np
import spaces
import torch
from PIL import Image
from diffusers import QwenImage21Pipeline

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen-Image-2.1")
MAX_SEED = np.iinfo(np.int32).max
MAX_REF_IMAGES = 10
# ZeroGPU rejects any request above 120s of GPU time.
MAX_GPU_SECONDS = 120
MIN_GPU_SECONDS = 30

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
):
    """Seconds to reserve, from the cost of the request.

    ZeroGPU charges quota for the reserved window rather than the wall clock
    used, so ask for close to the real cost and never exceed the 120s ceiling.
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
):
    encoder_device = next(pipe.text_encoder.parameters()).device
    if encoder_device.type != "cuda":
        pipe.text_encoder.to("cuda")
    _TEXT_ENCODER_EVICTION["armed"] = True
    torch.cuda.reset_peak_memory_stats()
    _vram_note(f"start {output_resolution}px, text encoder was on {encoder_device}")

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
    progress=gr.Progress(track_tqdm=True),
):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Enter a prompt.")

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
    )

    mode = "edit" if refs else "text-to-image"
    size = f"{width}x{height}" if width else f"{image.width}x{image.height} (from reference)"
    gpu = "full GPU" if runner is _generate_full_gpu else "half GPU"
    has_alpha = "RGBA" if image.mode == "RGBA" else "RGB"
    info = (
        f"{mode} | {size} | {int(num_inference_steps)} steps | seed {seed} | "
        f"cfg {float(guidance_scale):.1f} | {gpu} | {has_alpha}\n{prompt}"
    )
    return image, info


EXAMPLES = [
    ['A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, reflections on wet pavement'],
    ["A cute cartoon dragon sticker, thick outlines, flat pastel colors"],
    ["Change the background to a sunset beach, keep the subject unchanged"],
    [
        "These three characters are sitting around a campfire in a forest, "
        "cinematic lighting, shallow depth of field"
    ],
]

with gr.Blocks(title="Qwen-Image-2.1") as demo:
    gr.Markdown(
        "# Qwen-Image-2.1\n"
        "Unified text-to-image generation and image editing, running on ZeroGPU. "
        "Add reference images to switch to editing mode; leave them empty for text-to-image.\n\n"
        "[Model](https://huggingface.co/Qwen/Qwen-Image-2.1) | "
        "[Blog](https://qwen.ai/blog?id=qwen-image-2.1) | "
        "[GitHub](https://github.com/QwenLM/Qwen-Image-2.1)"
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
                label="Negative prompt (used only when guidance > 1)",
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
        ],
        outputs=[result, status],
    )

    gr.Markdown(
        "ZeroGPU is shared: without an account you get 2 minutes of GPU time per day, "
        "with a free account 5 minutes, with PRO 40 minutes. Every run reserves its window up "
        "front and ZeroGPU charges that reserved time times 1.5, so a 1K image at 40 steps costs "
        "about 95 seconds of quota and the same image at 8 steps costs 45. A 2K edit with "
        "reference images runs on the full GPU and is charged double. Lower the step count to "
        "get more runs. Nothing is stored and the Space keeps nothing between sessions."
    )

demo.queue().launch()
