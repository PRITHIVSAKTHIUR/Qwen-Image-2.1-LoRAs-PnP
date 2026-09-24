# **[Qwen-Image-2.1-LoRAs-PnP](https://huggingface.co/spaces/prithivMLmods/Qwen-Image-2.1-LoRAs-PnP)**

Qwen-Image-2.1-LoRAs-PnP is a flexible, plug-and-play image synthesis and editing platform built on top of the `Qwen/Qwen-Image-2.1` diffusion pipeline. Operating natively in `bfloat16`, the application consolidates text-to-image synthesis, multi-reference image editing, and transparent background (RGBA) generation inside a single unified framework.

The platform integrates a dynamic LoRA loading system supporting pre-configured community checkpoints—including 4-Step Turbo distillation (`Viggle/Qwen-Image-2.1-viggle-turbo`), spatial object manipulation (`Object Mover` and `Object Remover`), and `Natural Exposure`—along with support for arbitrary Hugging Face LoRA repositories. Built with a FastAPI backend server (`gradio.Server`) and a dark-mode frontend workspace, it includes image filmstrips, multi-reference queues, and an inspection engine.

<img width="1911" height="891" alt="Screenshot From 2026-09-24 10-17-31" src="https://github.com/user-attachments/assets/baac2b84-457f-4350-aa3a-124d5996134a" />

<img width="1911" height="891" alt="Screenshot From 2026-09-24 10-40-29" src="https://github.com/user-attachments/assets/c41baabf-a89b-489a-a500-431a1355232a" />

### **Key Features**

* **Unified T2I, I2I & Multi-Reference Workflows:** Execute prompt-based generations or supply up to 10 visual references simultaneously for guided scene modifications, asset transpositions, and character editing.
* **Plug-and-Play LoRA Architecture:** Features an on-demand adapter loader supporting:
* `4-Step Turbo (Viggle)`: Distilled high-speed inference in 4 sampling steps.
* `Object Mover (Bbox Preview)` & `Object Remover`: Precision object adjustments and removals.
* `Natural Exposure`: Lighting and dynamic range refinement.
* `Custom LoRA`: Load any compatible Qwen-Image-2.1 LoRA directly by entering a Hugging Face repository ID and optional weight filename.


* **Native RGBA Transparency:** Generates isolated subjects with native alpha channel transparency using automatic prompt-formatting guards.
* **Dynamic Resolution Tiers:** Supports standard aspect ratios (`1:1`, `4:3`, `3:4`, `3:2`, `2:3`, `16:9`, `9:16`) mapped across **1K (fast)** and **2K (native)** resolution tiers.
* **VRAM Optimization & Text-Encoder Eviction:** Implements forward pre-hooks to temporarily evict the text encoder off GPU memory during VAE and transformer decoding stages, minimizing CUDA overhead.

### **Repository Structure**

```text
├── assets/
│   ├── Screenshot From 2026-09-24 09-58-52.png
│   ├── Screenshot From 2026-09-24 10-17-31.png
│   ├── Screenshot From 2026-09-24 10-19-56.png
│   ├── Screenshot From 2026-09-24 10-38-22.png
│   └── Screenshot From 2026-09-24 10-40-29.png
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

To set up the environment locally, configure your system according to the specifications below. A dedicated CUDA-capable GPU is required.

* **Python Version:** Minimum Python **3.10.13** is required; Python **3.14** is recommended.
* **PyTorch Version:** `torch==2.11.0` or above for optimal compatibility.
* **CUDA Version:** **CUDA 13.0** is recommended (`--extra-index-url https://download.pytorch.org/whl/cu130`), matching the live Hugging Face demo.

#### **Running with `uv` (Recommended)**

`uv` is an ultra-fast Python package and project manager written in Rust. It ensures rapid virtual environment setup and exact dependency synchronization based on `uv.lock`.

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

**Step 4 — Run the application**

```bash
uv run app.py
```

#### **Standard PIP Implementation**

**1. Update Package Manager**

```bash
pip install "pip>=26.2.1"
```

**2. Install Core Dependencies**
Install the primary deep learning stack and bleeding-edge diffusers dependencies:

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

Once initialized, open your browser to the local server address output in your terminal (typically `http://127.0.0.1:7860/`).

1. **Input Composition:**
* Enter a descriptive generation or modification directive in the **Prompt** field.
* *(Optional)* Drop or upload reference images into the canvas or filmstrip (supports up to 10 images for multi-reference editing).


2. **Select Mode & Resolution:**
* Choose an adapter mode (e.g., `4-Step Turbo`, `Object Mover`, or select `Custom LoRA` to input your own Hugging Face model repository).
* Set your target **Tier** (`1K (fast)` or `2K (native)`) and **Aspect Ratio**.


3. **Configure Advanced Settings:**
* Adjust inference steps (automatically locked to 4 when in Turbo mode).
* Adjust Guidance Scale (CFG), Seed, or toggle **Transparent background (RGBA)**.


4. **Execute:** Click **Generate Image** or press ⌘/Ctrl + Enter. The resulting output will populate the central interactive canvas with one-click download access.

### **License and Source**

* **License:** [Qwen RESEARCH LICENSE](https://github.com/PRITHIVSAKTHIUR/Qwen-Image-2.1-LoRAs-PnP/blob/main/LICENSE?utm_source=gemini)
* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/Qwen-Image-2.1-LoRAs-PnP.git](https://github.com/PRITHIVSAKTHIUR/Qwen-Image-2.1-LoRAs-PnP.git?utm_source=gemini)
* **Hugging Face Live Space:** [https://huggingface.co/spaces/prithivMLmods/Qwen-Image-2.1-LoRAs-PnP](https://huggingface.co/spaces/prithivMLmods/Qwen-Image-2.1-LoRAs-PnP?utm_source=gemini)
