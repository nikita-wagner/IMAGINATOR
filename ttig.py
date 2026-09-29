import os
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("torch").setLevel(logging.ERROR)

import torch
from diffusers import StableDiffusion3Pipeline, StableDiffusion3Img2ImgPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

model_id = "stabilityai/stable-diffusion-3.5-medium"  # 3.5-medium is the smallest SD3 model that still has the new text encoder and tokenizer

# Full-precision transformer (no NF4/FP4 quantization) — this is the biggest
# quality lever now that VRAM allows it. T5-XXL stays excluded as requested.
pipe = StableDiffusion3Pipeline.from_pretrained(
	model_id,
	text_encoder_3=None,
	tokenizer_3=None,
	dtype=dtype,
).to(device)  # no CPU offload — everything stays resident on the GPU

# Img2img pipeline reuses pipe's already-loaded weights (no extra VRAM/download).
img2img_pipe = StableDiffusion3Img2ImgPipeline(**pipe.components).to(device)

# If you hit a CUDA out-of-memory error, lower height/width, num_images_per_prompt,
# or num_inference_steps before reaching for offload.

prompt = "Shadow the Hedgehog, realistic 3d render, cinematic lighting, photorealistic, 8k, high detail"
directory = "Textures"
height = 1024
width = 1024
guidance_scale = 7.5
num_inference_steps = 60
seed = 251217772  # set to any integer for reproducible results; change it to get a different image


def sanitize_filename(text, max_length=60):
	"""Turn arbitrary prompt text into a short, filesystem-safe slug.
	Long prompts otherwise blow past Windows' ~260-char path limit and
	cause a confusing FileNotFoundError on save()."""
	keep = "".join(c if c.isalnum() or c in (" ", "-", "_") else "" for c in text)
	slug = "_".join(keep.split())  # collapse whitespace runs to single underscores
	return slug[:max_length].rstrip("_")


def test_inference_steps(prompt, min_steps=0, max_steps=60, out_dir=directory):
	"""Generate one image per inference-step count in [min_steps, max_steps]
	(inclusive), all from the same seed, so only the step count varies."""
	os.makedirs(out_dir, exist_ok=True)
	base_name = sanitize_filename(prompt)

	for steps in range(min_steps, max_steps + 1):
		generator = torch.Generator(device=device).manual_seed(seed)
		try:
			image = pipe(
				prompt,
				height=height,
				width=width,
				num_inference_steps=steps,
				guidance_scale=guidance_scale,
				num_images_per_prompt=1,
				generator=generator,
			).images[0]
		except Exception as e:
			# num_inference_steps=0 (and possibly 1) isn't valid for the
			# scheduler and will raise — skip it and keep going.
			print(f"skipped steps={steps}: {e}")
			continue

		filename = f"{base_name}_{height}_{width}_{guidance_scale}_steps{steps}_seed{seed}.png"
		image.save(f"{out_dir}/{filename}")
		print(f"saved {prompt} (steps={steps}) to {filename}")


def just_one_image(prompt, out_dir=directory):
	"""Generate one image from the prompt and save it to disk."""
	os.makedirs(out_dir, exist_ok=True)
	base_name = sanitize_filename(prompt)

	generator = torch.Generator(device=device).manual_seed(seed)
	image = pipe(
		prompt,
		height=height,
		width=width,
		num_inference_steps=num_inference_steps,
		guidance_scale=guidance_scale,
		num_images_per_prompt=1,
		generator=generator,
	).images[0]

	filename = f"{base_name}_{height}_{width}_{guidance_scale}_steps{num_inference_steps}_seed{seed}.png"
	image.save(f"{out_dir}/{filename}")
	print(f"saved {prompt} to {filename}")

just_one_image(prompt, out_dir=directory)

def test_img2img_modification(base_prompt, edit_prompt, strength=0.6, out_dir=directory):
	"""Generate a base image, then run it back through img2img with a
	different prompt to modify it (e.g. plain asphalt -> asphalt with trash
	on it). `strength` controls how much the edit is allowed to change the
	base image: 0.0 = untouched, 1.0 = almost a fresh generation that
	ignores the original."""
	os.makedirs(out_dir, exist_ok=True)

	# Step 1: generate the base image with the normal text-to-image pipe.
	generator = torch.Generator(device=device).manual_seed(seed)
	base_image = pipe(
		base_prompt,
		height=height,
		width=width,
		num_inference_steps=num_inference_steps,
		guidance_scale=guidance_scale,
		generator=generator,
	).images[0]
	base_filename = f"{sanitize_filename(base_prompt)}_base_seed{seed}.png"
	base_image.save(f"{out_dir}/{base_filename}")
	print(f"saved base image to {base_filename}")

	# Step 2: feed the base image + a new prompt into img2img to modify it.
	generator = torch.Generator(device=device).manual_seed(seed)
	modified_image = img2img_pipe(
		prompt=edit_prompt,
		image=base_image,
		strength=strength,
		num_inference_steps=20,
		guidance_scale=10.0,
		generator=generator,
	).images[0]
	edit_filename = f"{sanitize_filename(edit_prompt)}_strength{strength}_seed{seed}.png"
	modified_image.save(f"{out_dir}/{edit_filename}")
	print(f"saved modified image to {edit_filename}")

	return base_image, modified_image

#
#test_img2img_modification(
#	base_prompt="Dark asphalt wall texture, realistic, high detail, 8k, photorealistic",
#	edit_prompt="Add scattered trash and bottles on the asphalt wall, realistic, high detail, 8k, photorealistic",
#	strength=1,
#)