import os
import torch
import gradio as gr
from PIL import Image
from pathlib import Path

from trellis2.pipelines import Trellis2ImageTo3DPipeline

# ------------------------------------------------------------
# Global config
# ------------------------------------------------------------

MODEL_NAME = "microsoft/TRELLIS.2-4B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

pipeline = None


# ------------------------------------------------------------
# Model loading
# ------------------------------------------------------------

def load_model(
    model_name: str,
    backend: str,
    low_vram: bool,
):
    """
    Loads TRELLIS-2 once and keeps it in memory.

    This replaces:
      - Trellis2LoadModel
      - ComfyUI model management
    """
    global pipeline

    os.environ["ATTN_BACKEND"] = backend
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
        model_name,
        keep_models_loaded=True,
        use_fp8=("FP8" in model_name),
    )

    pipeline.low_vram = low_vram

    if DEVICE == "cuda":
        pipeline.cuda()
    else:
        pipeline.to("cpu")

    return f"Model loaded: {model_name} ({DEVICE})"


# ------------------------------------------------------------
# Inference
# ------------------------------------------------------------

def image_to_3d(
    image: Image.Image,
    seed: int,
    pipeline_type: str,
    sparse_steps: int,
    shape_steps: int,
    texture_steps: int,
    max_views: int,
):
    """
    Runs the TRELLIS-2 image-to-3D pipeline.

    Replaces:
      - Trellis2MeshWithVoxelGenerator
    """

    if pipeline is None:
        raise RuntimeError("Model not loaded")

    torch.manual_seed(seed)

    mesh, = pipeline.run(
        image=image,
        seed=seed,
        pipeline_type=pipeline_type,
        sparse_structure_sampler_params={"steps": sparse_steps},
        shape_slat_sampler_params={"steps": shape_steps},
        tex_slat_sampler_params={"steps": texture_steps},
        max_views=max_views,
        generate_texture_slat=True,
        use_tiled=True,
    )

    output_path = OUTPUT_DIR / f"trellis_{seed}.glb"
    mesh.export(output_path)

    return str(output_path)


# ------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------

with gr.Blocks(title="TRELLIS-2 Image → 3D") as demo:
    gr.Markdown("# TRELLIS-2 Image to 3D")

    with gr.Row():
        model_name = gr.Dropdown(
            ["microsoft/TRELLIS.2-4B", "visualbruno/TRELLIS.2-4B-FP8"],
            value=MODEL_NAME,
            label="Model",
        )
        backend = gr.Dropdown(
            ["flash_attn", "xformers", "sdpa", "flash_attn_3"],
            value="flash_attn",
            label="Attention Backend",
        )
        low_vram = gr.Checkbox(value=True, label="Low VRAM")

    load_btn = gr.Button("Load Model")
    load_status = gr.Textbox(label="Status")

    load_btn.click(
        load_model,
        inputs=[model_name, backend, low_vram],
        outputs=load_status,
    )

    gr.Markdown("## Image to 3D")

    image = gr.Image(type="pil", label="Input Image")

    with gr.Row():
        seed = gr.Number(value=12345, label="Seed", precision=0)
        pipeline_type = gr.Dropdown(
            ["512", "1024", "1024_cascade", "1536_cascade"],
            value="1024_cascade",
            label="Pipeline Type",
        )

    with gr.Row():
        sparse_steps = gr.Slider(1, 50, value=12, label="Sparse Steps")
        shape_steps = gr.Slider(1, 50, value=12, label="Shape Steps")
        texture_steps = gr.Slider(1, 50, value=12, label="Texture Steps")
        max_views = gr.Slider(1, 8, value=4, step=1, label="Views")

    run_btn = gr.Button("Generate 3D Mesh")
    output_file = gr.File(label="Output GLB")

    run_btn.click(
        image_to_3d,
        inputs=[
            image,
            seed,
            pipeline_type,
            sparse_steps,
            shape_steps,
            texture_steps,
            max_views,
        ],
        outputs=output_file,
    )

demo.launch()
