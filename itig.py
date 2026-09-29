import os
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("torch").setLevel(logging.ERROR)

import torch
from diffusers import FluxKontextPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
from diffusers.utils import load_image

device = "cuda" if torch.cuda.is_available() else "cpu"

# FLUX.1-Kontext-dev is an instruction-based image editor. We load only the
# transformer from the Q2_K GGUF quant (~4 GB) and pull the CLIP/T5 text
# encoders + VAE from the full black-forest-labs/FLUX.1-Kontext-dev repo
# (already in the HF cache; its transformer weights were never downloaded).
gguf_path = (
    "https://huggingface.co/QuantStack/FLUX.1-Kontext-dev-GGUF/"
    "blob/main/flux1-kontext-dev-Q4_K_M.gguf"
)
base_repo = "black-forest-labs/FLUX.1-Kontext-dev"

transformer = FluxTransformer2DModel.from_single_file(
    gguf_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
    config=base_repo,
    subfolder="transformer",
)

pipe = FluxKontextPipeline.from_pretrained(
    base_repo,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
# 12 GB VRAM can't hold the Q2_K transformer + T5-XXL at once, so stream
# modules on/off the GPU as they're needed.
pipe.enable_model_cpu_offload()

# ---------------------------------------------------------------------------
in_path = "Textures/0.png"
out_path = "Output/0_enhanced.png"

prompt = (
    "Fix the image, make Shadow the Hedgehog look like a real character with furry coat, remove artifacts"
)
guidance_scale = 10.0
num_inference_steps = 10
seed = 136232

image = load_image(in_path)

generator = torch.Generator(device=device).manual_seed(seed)
result = pipe(
    image=image,
    prompt=prompt,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    generator=generator,
).images[0]

os.makedirs(os.path.dirname(out_path), exist_ok=True)
result.save(out_path)
print(f"saved enhanced image to {out_path}")
