# **Qwen-Image-2.1-LoRAs-PnP**

Qwen-Image-2.1-LoRAs-PnP is a modular image generation, multi-reference editing, and plug-and-play (PnP) LoRA execution studio powered by the `Qwen/Qwen-Image-2.1` base pipeline (`QwenImage21Pipeline`). The platform handles both text-to-image and complex image editing workflows (including object movement, object removal, natural exposure correction, head/face swapping, and native RGBA transparent generation) at 1K and 2K resolution tiers.

The system features dynamic lazy loading of pre-registered LoRAs (such as Viggle 4-Step Turbo, Object Mover, Object Remover, Natural Exposure, and Best Face Swap), as well as direct arbitrary LoRA loading from Hugging Face Hub repositories. It is served through a single-page web application (SPA) built with a FastAPI backend (`gradio.Server`) and a dark-themed client interface featuring an interactive bounding box annotator, history management, and a dedicated LoRA Explorer tray.

<img width="1920" height="896" alt="Screenshot From 2026-09-27 17-25-07" src="https://github.com/user-attachments/assets/775da3ac-1f5f-406f-8553-7c37ebb26464" />

### **Key Features**

* **Unified Qwen-Image-2.1 Execution:** Handles text-to-image, multi-reference image composition (up to 10 images), and native alpha-channel RGBA transparency using `QwenImage21Pipeline`.
* **Plug-and-Play (PnP) LoRA Hub:** Supports built-in presets alongside on-the-fly custom LoRA loading by providing any Hugging Face model repository ID and target `.safetensors` weight file.
* **4-Step Turbo Mode:** Integrates Viggle's step-distilled turbo LoRA paired with `FlowMatchEulerDiscreteScheduler` to deliver high-quality outputs in 4 sampling steps.
* **Canvas Bounding Box Annotator:** Includes client-side vector bounding box drawing tools directly over input reference images to guide spatial tasks like object movement and targeted object removal.
* **Dynamic VRAM Management:** Incorporates forward pre-hooks to automatically evict the heavy vision-language text encoder to CPU after text encoding, freeing GPU memory for high-resolution 2K VAE decoding.
* **Interactive Studio SPA:** A dark-mode single-page application featuring input filmstrips, an A/B result viewer, quick preset selectors, and an integrated LoRA visual explorer grid.

### **Repository Structure**

```text
├── assets/
│   ├── Screenshot From 2026-09-24 09-58-52.png
│   ├── Screenshot From 2026-09-24 10-17-31.png
│   ├── Screenshot From 2026-09-24 10-19-56.png
│   ├── Screenshot From 2026-09-24 10-38-22.png
│   └── Screenshot From 2026-09-24 10-40-29.png
├── examples/
│   ├── anime_input.jpg
│   ├── exposure_input.jpg
│   ├── faceswap_input_1.jpeg
│   ├── faceswap_input_2.jpeg
│   ├── objmv_input.jpg
│   ├── objmvT_input.jpg
│   ├── objrm_input.jpg
│   ├── objrmT_input.jpg
│   └── outpaint_input.jpg
├── LoRA_Cover/
│   ├── anime_cover.png
│   ├── exposure_cover.png
│   ├── faceswap_cover.png
│   ├── objmv_cover.jpg
│   ├── objmvT_cover.jpg
│   ├── objrm_cover.jpg
│   ├── objrmT_cover.jpg
│   └── outpaint_cover.jpg
├── app.py
├── index.html
├── LICENSE
├── pre-requirements.txt
├── pyproject.toml
├── README.md
├── requirements.txt
└── uv.lock
```

### **Installation and Requirements**

To set up the Qwen-Image-2.1-LoRAs-PnP environment locally, configure your system according to the specifications below. A modern CUDA-enabled GPU (with bfloat16 support) is required.

* **Python Version:** Minimum Python **3.10.13** is required; Python **3.14** is recommended.
* **PyTorch Version:** `torch==2.11.0` or above is required for optimal system compatibility.
* **CUDA Version:** **CUDA 13.0** is recommended (`--extra-index-url [https://download.pytorch.org/whl/cu130](https://download.pytorch.org/whl/cu130)`), matching the environment used on the live Hugging Face demo.

#### **Running with `uv` (Recommended)**

`uv` is an ultra-fast Python package and project manager written in Rust. It ensures rapid virtual environment setup and exact dependency synchronization based on the `uv.lock` file.

**Step 1 — Install `uv`**

* **macOS / Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
* **Windows:** `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

**Step 2 — Clone the repository**

```bash
git clone https://github.com/PRITHIVSAKTHIUR/Qwen-Image-2.1-LoRAs-PnP.git
cd Qwen-Image-2.1-LoRAs-PnP
```

**Step 3 — Initialize the project and install dependencies**

```bash
uv sync
```

**Step 4 — Run the script**

```bash
uv run app.py
```

#### **Standard PIP Implementation**

**1. Update Package Manager**
Upgrade your local package manager:

```bash
pip install "pip>=26.2.1"
```

**2. Install Core Dependencies**
Install the primary deep learning stack, transformer libraries, and core computing utilities listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

#### **Core Requirements List (`requirements.txt`)**

```text
--extra-index-url https://download.pytorch.org/whl/cu130

git+https://github.com/huggingface/diffusers.git
torch==2.11.0
torchvision==0.26.0
transformers==5.17.0
accelerate==1.14.0
peft==0.19.1
gradio==6.28.0
av==17.1.0
spaces>=0.51.1
huggingface-hub>=1.24.0
pillow>=12.3.0
```

### **Usage**

Once the FastAPI web server initializes, open your browser to the local address output in your terminal (typically `http://127.0.0.1:7860/`).

1. **Upload References (Optional):** Drag and drop one or more images into the main stage or use the rail upload button. Leave empty for text-to-image mode.
2. **Select Mode & LoRA:**
* Select a built-in preset from the **Mode & Resolution** dropdown (e.g., *4-Step Turbo*, *Object Mover*, *Object Remover*, *Natural Exposure*, or *Face Swap*).
* Or choose **Custom LoRA (enter repo below)** to enter any public Hugging Face repository ID (e.g., `user/repo`) and weight filename to load it on the fly.

3. **Bounding Box Annotation:** When performing localized tasks like object movement or removal, click the **BBox** tool in the left rail to draw target red bounding boxes directly over the input image.
4. **Configure Parameters:** Adjust resolution tier (`1K` or `2K`), aspect ratio, inference steps, and guidance scale. Toggle **Transparent background (RGBA)** if alpha-channel output is required.
5. **Execute:** Click **Generate Image** or press ⌘/Ctrl + Enter. The result will render in the primary canvas and display generation metadata in the inspector panel.

### **License and Source**

* **License:** [Qwen Research License Agreement](https://github.com/PRITHIVSAKTHIUR/Qwen-Image-2.1-LoRAs-PnP/blob/main/LICENSE?utm_source=gemini)
* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/Qwen-Image-2.1-LoRAs-PnP.git](https://github.com/PRITHIVSAKTHIUR/Qwen-Image-2.1-LoRAs-PnP.git?utm_source=gemini)
* **Hugging Face Live Space:** [https://huggingface.co/spaces/prithivMLmods/Qwen-Image-2.1-LoRAs-PnP](https://huggingface.co/spaces/prithivMLmods/Qwen-Image-2.1-LoRAs-PnP?utm_source=gemini)
