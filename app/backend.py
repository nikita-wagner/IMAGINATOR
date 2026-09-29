"""Pipeline loading and job execution.

Everything in here is GUI-agnostic: jobs report progress through a small
callback object, so the same code could just as well be driven from a CLI.
"""

import gc
import os
import random
import warnings
import logging
from dataclasses import dataclass, field
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("torch").setLevel(logging.ERROR)

from . import models as registry
from . import formats
from .models import TXT2IMG, IMG2IMG


# DDS and TGA included: Pillow decodes them in memory, so a game texture can be
# fed in directly without being converted to PNG on disk first.
IMAGE_EXTS = formats.READABLE_EXTS

MODE_TXT2IMG = "txt2img"
MODE_IMG2IMG = "img2img"
MODE_BATCH = "batch"
MODE_SWEEP = "sweep"
MODE_CHAIN = "chain"

MODE_LABELS = {
	MODE_TXT2IMG: "Text → Image",
	MODE_IMG2IMG: "Image → Image (one file)",
	MODE_BATCH: "Image → Image (batch of files)",
	MODE_SWEEP: "Step sweep (one image per step count)",
	MODE_CHAIN: "Text → Image → Image (generate, then edit)",
}
MODE_ORDER = (MODE_TXT2IMG, MODE_IMG2IMG, MODE_BATCH, MODE_SWEEP, MODE_CHAIN)

# Which pipeline family each mode needs from the selected model.
MODE_TASK = {
	MODE_TXT2IMG: TXT2IMG,
	MODE_SWEEP: TXT2IMG,
	MODE_IMG2IMG: IMG2IMG,
	MODE_BATCH: IMG2IMG,
	MODE_CHAIN: IMG2IMG,   # chain also needs a txt2img model, picked separately
}


class Cancelled(Exception):
	"""Raised inside the denoising callback when the user presses Cancel."""


@dataclass
class JobConfig:
	mode: str = MODE_TXT2IMG
	model_key: str = "sd35-medium"
	quant: str = "Q4_K_M"
	# Only used by MODE_CHAIN: the model that makes the base image.
	base_model_key: str = "sd35-medium"

	prompt: str = ""
	negative_prompt: str = ""
	# MODE_CHAIN: prompt for the first (text-to-image) stage.
	base_prompt: str = ""

	inputs: list = field(default_factory=list)   # input image paths
	out_dir: str = "Output"

	width: int = 1024
	height: int = 1024
	steps: int = 30
	guidance: float = 7.5
	strength: float = 0.6
	seed: int = 0
	random_seed: bool = True
	num_images: int = 1

	sweep_min: int = 1
	sweep_max: int = 30

	memory_mode: str = "auto"   # auto | none | cpu_offload | sequential_offload


class Progress:
	"""Sink for job output. The GUI subclasses this to marshal onto the UI thread."""

	def log(self, text):
		print(text)

	def status(self, text):
		pass

	def steps(self, done, total):
		pass

	def items(self, done, total):
		pass

	def saved(self, path, source=None):
		pass

	def cancelled(self):
		return False


def sanitize_filename(text, max_length=60):
	"""Turn arbitrary prompt text into a short, filesystem-safe slug.

	Long prompts otherwise blow past the ~260-char Windows path limit and cause
	a confusing FileNotFoundError on save().
	"""
	keep = "".join(c if c.isalnum() or c in (" ", "-", "_") else "" for c in text)
	slug = "_".join(keep.split())          # collapse whitespace runs to underscores
	slug = slug[:max_length].rstrip("_")
	return slug or "image"


def unique_path(directory, filename):
	"""Never silently overwrite an earlier render: foo.png -> foo_1.png -> ..."""
	path = Path(directory) / filename
	if not path.exists():
		return path
	stem, suffix = path.stem, path.suffix
	n = 1
	while True:
		candidate = path.with_name(f"{stem}_{n}{suffix}")
		if not candidate.exists():
			return candidate
		n += 1


# Windows' path limit is ~260 characters total; keep some margin below it since
# unique_path() may still need to append "_1", "_2", ... on a collision.
MAX_PATH_BUDGET = 240


def fit_filename(out_dir, filename):
	"""Shorten filename (keeping its extension) so out_dir/filename stays under
	Windows' path limit, no matter how long out_dir itself is.

	Each call site already truncates the pieces it controls, but img2img
	filenames also fold in the *input* file's name - and when that input is an
	earlier output from this same app, its name already carries the previous
	round's prompt and seed. Chaining edits onto their own output (edit, then
	edit the result, then edit that) adds another ~50 characters every round
	with nothing capping the total, so this is the backstop that keeps that
	growth from ever reaching save() as a crash instead of a shorter name.
	"""
	budget = max(20, MAX_PATH_BUDGET - len(str(Path(out_dir).resolve())) - 1)
	if len(filename) <= budget:
		return filename
	stem, suffix = os.path.splitext(filename)
	return stem[:max(1, budget - len(suffix))].rstrip("_") + suffix


def round_to(value, multiple=16):
	"""VAEs need dimensions divisible by 16; round down, keep at least one tile."""
	return max(multiple, int(value) // multiple * multiple)


def collect_images(paths):
	"""Expand a mixed list of files and folders into a sorted list of image files."""
	found = []
	for raw in paths:
		p = Path(raw)
		if p.is_dir():
			found.extend(sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXTS))
		elif p.is_file() and p.suffix.lower() in IMAGE_EXTS:
			found.append(p)
	seen, out = set(), []
	for p in found:
		key = str(p.resolve()).lower()
		if key not in seen:
			seen.add(key)
			out.append(p)
	return out


# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------

class PipelineCache:
	"""Holds one loaded model at a time and reuses it across runs.

	Loading SD3.5 or FLUX takes tens of seconds, so keeping the pipeline alive
	between clicks is the difference between an interactive tool and a batch
	script. Switching models frees the old one first - 12 GB does not fit two.
	"""

	def __init__(self):
		self._key = None
		self._bundle = {}

	@staticmethod
	def device():
		import torch

		return "cuda" if torch.cuda.is_available() else "cpu"

	@staticmethod
	def dtype():
		import torch

		return torch.bfloat16 if torch.cuda.is_available() else torch.float32

	def loaded_label(self):
		return self._key[0] if self._key else None

	def release(self, cb=None):
		if self._bundle and cb:
			cb.log("Freeing the previously loaded model...")
		self._bundle = {}
		self._key = None
		gc.collect()
		try:
			import torch

			if torch.cuda.is_available():
				torch.cuda.empty_cache()
		except Exception:
			pass

	def _offload_choice(self, spec, memory_mode):
		"""Decide whether weights stay resident on the GPU or stream on demand."""
		if memory_mode in ("none", "cpu_offload", "sequential_offload"):
			return memory_mode
		# auto: these cannot hold their transformer and text encoders on 12 GB.
		if spec.kind in OFFLOAD_KINDS or spec.key == "sd35-large":
			return "cpu_offload"
		return "none"

	def get(self, spec, task, cfg, cb):
		"""Return a ready pipeline for (model, task), loading it if needed."""
		offload = self._offload_choice(spec, cfg.memory_mode)
		quant = registry.quant_for(spec, cfg.quant)
		key = (spec.key, quant, offload)

		if key != self._key:
			self.release(cb)
			cb.status(f"Loading {spec.label}...")
			cb.log(f"Loading {spec.label} ({spec.repo}) on {self.device()}...")
			if not registry.is_cached(spec):
				cb.log("  Not in the local cache yet - downloading. The first run of "
				       "a new model can take a long time.")
			self._bundle = self._load(spec, quant, offload, cb)
			self._key = key
			cb.log("Model ready.")

		if task in self._bundle:
			return self._bundle[task]

		# The sibling pipeline reuses the already-loaded weights, so it costs no
		# extra VRAM and no extra download.
		pipe = self._derive(spec, task, cb)
		self._bundle[task] = pipe
		return pipe

	def _apply_memory(self, pipe, offload, cb):
		if offload == "cpu_offload":
			cb.log("  Memory: model CPU offload (modules stream on/off the GPU).")
			pipe.enable_model_cpu_offload()
		elif offload == "sequential_offload":
			cb.log("  Memory: sequential CPU offload (lowest VRAM, slowest).")
			pipe.enable_sequential_cpu_offload()
		else:
			cb.log("  Memory: everything resident on the GPU.")
			pipe.to(self.device())
		return pipe

	@staticmethod
	def _gguf_transformer(cls, spec, quant, cb):
		"""Load just the transformer from a GGUF quant; its config comes from
		the full repo, as do the text encoders and VAE later on."""
		import torch
		from diffusers import GGUFQuantizationConfig

		url = (f"https://huggingface.co/{spec.gguf_repo}/blob/main/"
		       f"{spec.gguf_file.format(quant=quant)}")
		cb.log(f"  Transformer quant: {quant}")
		model = cls.from_single_file(
			url,
			quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
			torch_dtype=torch.bfloat16,
			config=spec.repo,
			subfolder="transformer",
		)
		PipelineCache._unpack_loose_gguf(model)
		return model

	@staticmethod
	def _unpack_loose_gguf(model):
		"""Dequantise GGUF weights that don't live in a Linear layer.

		diffusers only unpacks GGUF weights inside Linear layers, on the fly.
		Anything else - e.g. Qwen-Image 2.1's text_norm, stored as BF16 in its
		GGUF - would reach the maths as raw bytes (twice the expected width)
		and crash with a size mismatch. They're tiny, so unpack them once.
		"""
		import torch
		from diffusers.quantizers.gguf.utils import GGUFParameter, dequantize_gguf_tensor

		for module in model.modules():
			if isinstance(module, torch.nn.Linear):
				continue
			for name, param in list(module.named_parameters(recurse=False)):
				if isinstance(param, GGUFParameter):
					plain = dequantize_gguf_tensor(param).to(torch.bfloat16)
					setattr(module, name, torch.nn.Parameter(plain, requires_grad=False))

	@staticmethod
	def _nf4_config(spec, cb):
		"""from_pretrained kwargs that quantise spec.quantize_4bit on load."""
		if not spec.quantize_4bit:
			return {}
		import torch
		from diffusers import PipelineQuantizationConfig

		cb.log(f"  4-bit on load: {', '.join(spec.quantize_4bit)}")
		return {"quantization_config": PipelineQuantizationConfig(
			quant_backend="bitsandbytes_4bit",
			quant_kwargs={
				"load_in_4bit": True,
				"bnb_4bit_quant_type": "nf4",
				"bnb_4bit_compute_dtype": torch.bfloat16,
			},
			components_to_quantize=list(spec.quantize_4bit),
		)}

	def _load(self, spec, quant, offload, cb):
		import torch

		dtype = self.dtype()

		if spec.kind == "sd3":
			from diffusers import StableDiffusion3Pipeline

			# text_encoder_3 / tokenizer_3 are the T5-XXL pair: ~10 GB that buys
			# little for short prompts, so it stays out.
			pipe = StableDiffusion3Pipeline.from_pretrained(
				spec.repo,
				text_encoder_3=None,
				tokenizer_3=None,
				torch_dtype=dtype,
			)
			self._apply_memory(pipe, offload, cb)
			return {TXT2IMG: pipe}

		if spec.kind == "flux-kontext-gguf":
			from diffusers import FluxKontextPipeline, FluxTransformer2DModel

			# Only the transformer comes from the GGUF quant; the CLIP/T5 text
			# encoders and the VAE come from the full repo.
			transformer = self._gguf_transformer(FluxTransformer2DModel, spec, quant, cb)
			pipe = FluxKontextPipeline.from_pretrained(
				spec.repo,
				transformer=transformer,
				torch_dtype=torch.bfloat16,
			)
			self._apply_memory(pipe, offload, cb)
			return {IMG2IMG: pipe}

		if spec.kind == "flux-gguf":
			# FLUX.1-dev and finetunes of it (SRPO) that only swap the transformer.
			from diffusers import FluxPipeline, FluxTransformer2DModel

			transformer = self._gguf_transformer(FluxTransformer2DModel, spec, quant, cb)
			pipe = FluxPipeline.from_pretrained(
				spec.repo, transformer=transformer, torch_dtype=torch.bfloat16,
			)
			self._apply_memory(pipe, offload, cb)
			return {TXT2IMG: pipe}

		if spec.kind == "zimage-gguf":
			from diffusers import ZImagePipeline, ZImageTransformer2DModel

			transformer = self._gguf_transformer(ZImageTransformer2DModel, spec, quant, cb)
			pipe = ZImagePipeline.from_pretrained(
				spec.repo, transformer=transformer, torch_dtype=torch.bfloat16,
			)
			self._apply_memory(pipe, offload, cb)
			return {TXT2IMG: pipe}

		if spec.kind == "qwen21-gguf":
			from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel

			transformer = self._gguf_transformer(
				QwenImage21Transformer2DModel, spec, quant, cb)
			pipe = QwenImage21Pipeline.from_pretrained(
				spec.repo, transformer=transformer, torch_dtype=torch.bfloat16,
				**self._nf4_config(spec, cb),
			)
			self._apply_memory(pipe, offload, cb)
			# The same pipeline edits when handed an image, so no sibling needed.
			return {TXT2IMG: pipe, IMG2IMG: pipe}

		if spec.kind == "krea2":
			from diffusers import Krea2Pipeline

			pipe = Krea2Pipeline.from_pretrained(
				spec.repo, torch_dtype=torch.bfloat16, **self._nf4_config(spec, cb),
			)
			if spec.lora_repo:
				cb.log(f"  LoRA: {spec.lora_repo}")
				pipe.load_lora_weights(spec.lora_repo)
			self._apply_memory(pipe, offload, cb)
			return {TXT2IMG: pipe}

		if spec.kind == "instruct-pix2pix":
			# A dedicated architecture, not a text2img model with an img2img
			# sibling - AutoPipelineForText2Image doesn't recognise its class
			# at all, so it has to be loaded directly.
			from diffusers import StableDiffusionInstructPix2PixPipeline

			pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
				spec.repo, torch_dtype=dtype,
			)
			self._apply_memory(pipe, offload, cb)
			return {IMG2IMG: pipe}

		# Everything else (SDXL, SD1.5, ...) goes through the auto pipelines.
		from diffusers import AutoPipelineForText2Image

		pipe = AutoPipelineForText2Image.from_pretrained(spec.repo, torch_dtype=dtype)
		self._apply_memory(pipe, offload, cb)
		return {TXT2IMG: pipe}

	def _derive(self, spec, task, cb):
		"""Build the missing half of a model pair from already-loaded components."""
		if task == IMG2IMG and TXT2IMG in self._bundle:
			base = self._bundle[TXT2IMG]
			cb.log("  Reusing the loaded weights for image-to-image.")
			if spec.kind == "sd3":
				from diffusers import StableDiffusion3Img2ImgPipeline

				return StableDiffusion3Img2ImgPipeline(**base.components)
			from diffusers import AutoPipelineForImage2Image

			return AutoPipelineForImage2Image.from_pipe(base)

		if task == TXT2IMG and IMG2IMG in self._bundle:
			raise RuntimeError(
				f"{spec.label} is an image editor - it cannot do text-to-image. "
				"Pick a different model for this mode."
			)

		raise RuntimeError(f"{spec.label} does not support this mode.")


PIPELINES = PipelineCache()


# ---------------------------------------------------------------------------
# Running a job
# ---------------------------------------------------------------------------

def _make_callback(cb, total_steps):
	"""diffusers step hook: drives the progress bar and honours Cancel."""

	def on_step_end(pipe, step_index, timestep, kwargs):
		if cb.cancelled():
			raise Cancelled()
		cb.steps(step_index + 1, total_steps)
		return kwargs

	return on_step_end


def _resolve_seed(cfg, cb, offset=0):
	if cfg.random_seed:
		seed = random.randint(0, 2**31 - 1)
		cb.log(f"  Random seed: {seed}")
		return seed
	return int(cfg.seed) + offset


def _generator(seed):
	import torch

	return torch.Generator(device=PipelineCache.device()).manual_seed(int(seed))


def _neg(cfg, spec):
	"""FLUX has no classifier-free negative prompt; skip the kwarg for it."""
	if spec.kind in ("flux-kontext-gguf", "flux-gguf"):
		return {}
	text = cfg.negative_prompt.strip()
	return {"negative_prompt": text} if text else {}


def _guidance(spec, value):
	"""Qwen-Image 2.1 only has true CFG, under a different name."""
	if spec.kind == "qwen21-gguf":
		return {"true_cfg_scale": float(value)}
	return {"guidance_scale": float(value)}


# Models that condition on the whole input image (an instruction edit, or a
# concatenated-latent conditioning) rather than partially noising it, so a
# `strength` value would be meaningless for them.
NO_STRENGTH_KINDS = ("flux-kontext-gguf", "instruct-pix2pix", "qwen21-gguf")

# Models whose transformer and text encoders together overflow 12 GB, so
# "auto" memory mode streams them on and off the GPU.
OFFLOAD_KINDS = ("flux-kontext-gguf", "flux-gguf", "zimage-gguf", "qwen21-gguf", "krea2")


def _load_input(path, cfg, cb):
	"""Open an input image and size it for the VAE.

	Anything Pillow can decode works, DDS and TGA included - a compressed game
	texture is unpacked in memory here, so nothing has to be converted to PNG
	on disk beforehand. The conversion to RGB only happens when the source is
	not already RGB, and says so in the log when it does.
	"""
	from PIL import Image

	image = formats.open_image(path)
	if image.mode != "RGB":
		cb.log(f"  Decoded {formats.describe(image)} -> RGB")
		image = image.convert("RGB")
	if cfg.width > 0 and cfg.height > 0:
		target = (round_to(cfg.width), round_to(cfg.height))
	else:
		target = (round_to(image.width), round_to(image.height))
	if target != image.size:
		cb.log(f"  Resizing {image.width}x{image.height} -> {target[0]}x{target[1]}")
		image = image.resize(target, Image.LANCZOS)
	return image


def _save(image, out_dir, filename, cb, source=None):
	"""`source`, when given, is the pre-existing input file this result was
	edited from - carried through to the GUI so "Replace original" knows
	where the picture it's looking at actually came from."""
	os.makedirs(out_dir, exist_ok=True)
	filename = fit_filename(out_dir, filename)
	path = unique_path(out_dir, filename)
	image.save(path)
	cb.log(f"  Saved {path}")
	cb.saved(str(path), str(source) if source else None)
	return path


def run_job(cfg, cb):
	"""Execute one job. Raises Cancelled if the user aborts."""
	spec = registry.BY_KEY[cfg.model_key]
	os.makedirs(cfg.out_dir, exist_ok=True)

	if cfg.mode == MODE_TXT2IMG:
		_run_txt2img(cfg, spec, cb)
	elif cfg.mode == MODE_SWEEP:
		_run_sweep(cfg, spec, cb)
	elif cfg.mode in (MODE_IMG2IMG, MODE_BATCH):
		_run_img2img(cfg, spec, cb)
	elif cfg.mode == MODE_CHAIN:
		_run_chain(cfg, spec, cb)
	else:
		raise ValueError(f"Unknown mode {cfg.mode!r}")


def _run_txt2img(cfg, spec, cb):
	pipe = PIPELINES.get(spec, TXT2IMG, cfg, cb)
	width, height = round_to(cfg.width or 1024), round_to(cfg.height or 1024)
	total = max(1, cfg.num_images)
	base_name = sanitize_filename(cfg.prompt)

	for i in range(total):
		if cb.cancelled():
			raise Cancelled()
		cb.items(i, total)
		cb.status(f"Generating image {i + 1} of {total}...")
		seed = _resolve_seed(cfg, cb, offset=i)
		cb.log(f"Image {i + 1}/{total}  seed={seed}  {width}x{height}  "
		       f"steps={cfg.steps}  guidance={cfg.guidance}")
		image = pipe(
			cfg.prompt,
			height=height,
			width=width,
			num_inference_steps=int(cfg.steps),
			**_guidance(spec, cfg.guidance),
			num_images_per_prompt=1,
			generator=_generator(seed),
			callback_on_step_end=_make_callback(cb, int(cfg.steps)),
			**_neg(cfg, spec),
		).images[0]
		_save(image, cfg.out_dir,
		      f"{base_name}_{width}x{height}_g{cfg.guidance}_s{cfg.steps}_seed{seed}.png",
		      cb)
		cb.items(i + 1, total)


def _run_sweep(cfg, spec, cb):
	"""One image per inference-step count, all from the same seed, so the step
	count is the only thing that varies."""
	pipe = PIPELINES.get(spec, TXT2IMG, cfg, cb)
	width, height = round_to(cfg.width or 1024), round_to(cfg.height or 1024)
	lo, hi = int(cfg.sweep_min), int(cfg.sweep_max)
	if hi < lo:
		lo, hi = hi, lo
	seed = _resolve_seed(cfg, cb)
	base_name = sanitize_filename(cfg.prompt)
	counts = list(range(lo, hi + 1))
	cb.log(f"Step sweep {lo}..{hi} ({len(counts)} images) at seed {seed}")

	for i, steps in enumerate(counts):
		if cb.cancelled():
			raise Cancelled()
		cb.items(i, len(counts))
		cb.status(f"Sweep: {steps} steps ({i + 1}/{len(counts)})")
		try:
			image = pipe(
				cfg.prompt,
				height=height,
				width=width,
				num_inference_steps=int(steps),
				**_guidance(spec, cfg.guidance),
				num_images_per_prompt=1,
				generator=_generator(seed),
				callback_on_step_end=_make_callback(cb, max(int(steps), 1)),
				**_neg(cfg, spec),
			).images[0]
		except Cancelled:
			raise
		except Exception as exc:
			# steps=0 (and sometimes 1) is not valid for the scheduler.
			cb.log(f"  Skipped steps={steps}: {exc}")
			continue
		_save(image, cfg.out_dir,
		      f"{base_name}_{width}x{height}_g{cfg.guidance}_steps{steps}_seed{seed}.png",
		      cb)
		cb.items(i + 1, len(counts))


def _img2img_call(pipe, spec, cfg, image, prompt, steps, seed, cb):
	"""One img2img invocation, papering over FLUX vs SD parameter differences."""
	kwargs = dict(
		prompt=prompt,
		image=image,
		num_inference_steps=int(steps),
		**_guidance(spec, cfg.guidance),
		generator=_generator(seed),
		callback_on_step_end=_make_callback(cb, int(steps)),
		**_neg(cfg, spec),
	)
	if spec.kind in ("flux-kontext-gguf", "qwen21-gguf"):
		# Kontext and Qwen-Image 2.1 rewrite the whole image from an
		# instruction; no strength knob, but they do want the output size
		# stated (Qwen would otherwise resample to a 1024x1024-area canvas).
		kwargs["height"], kwargs["width"] = image.height, image.width
	elif spec.kind == "instruct-pix2pix":
		# InstructPix2Pix conditions on the whole input image via its own
		# image_guidance_scale (left at the diffusers default here), not by
		# partially noising it, so it has no strength knob either.
		pass
	else:
		kwargs["strength"] = float(cfg.strength)
	return pipe(**kwargs).images[0]


def _run_img2img(cfg, spec, cb):
	files = collect_images(cfg.inputs)
	if not files:
		raise ValueError("No input images selected.")
	if cfg.mode == MODE_IMG2IMG:
		files = files[:1]

	pipe = PIPELINES.get(spec, IMG2IMG, cfg, cb)
	variants = max(1, cfg.num_images)
	total = len(files) * variants
	done = 0
	cb.log(f"Editing {len(files)} image(s), {variants} variant(s) each.")

	for path in files:
		for v in range(variants):
			if cb.cancelled():
				raise Cancelled()
			cb.items(done, total)
			cb.status(f"Editing {path.name} ({done + 1}/{total})...")
			seed = _resolve_seed(cfg, cb, offset=v)
			image = _load_input(path, cfg, cb)
			extra = "" if spec.kind in NO_STRENGTH_KINDS else f"  strength={cfg.strength}"
			cb.log(f"{path.name}  seed={seed}  steps={cfg.steps}  "
			       f"guidance={cfg.guidance}{extra}")
			result = _img2img_call(pipe, spec, cfg, image, cfg.prompt, cfg.steps, seed, cb)
			# path.stem is capped here, not just the prompt: when the input is
			# an earlier output of this same app, its name already carries the
			# previous round's prompt and seed. Editing a result, then editing
			# that result again, would otherwise fold every past round's full
			# name into the next one and grow without bound.
			_save(result, cfg.out_dir,
			      f"{path.stem[:60]}_{sanitize_filename(cfg.prompt, 32)}_seed{seed}.png",
			      cb, source=path)
			done += 1
			cb.items(done, total)


def _run_chain(cfg, spec, cb):
	"""Generate a base image from text, then run it back through an editor with a
	second prompt (e.g. plain asphalt -> asphalt with trash on it)."""
	base_spec = registry.BY_KEY[cfg.base_model_key]
	total = max(1, cfg.num_images)
	width, height = round_to(cfg.width or 1024), round_to(cfg.height or 1024)
	swaps = base_spec.key != spec.key
	if swaps:
		cb.log("Note: the base and edit models differ, so each image reloads both "
		       "models. Using one model for both stages is much faster.")

	for i in range(total):
		if cb.cancelled():
			raise Cancelled()
		cb.items(i, total)
		seed = _resolve_seed(cfg, cb, offset=i)

		# Stage 1 - text to image.
		cb.status(f"Stage 1/2: base image ({i + 1}/{total})...")
		base_pipe = PIPELINES.get(base_spec, TXT2IMG, cfg, cb)
		cb.log(f"Base image {i + 1}/{total}  seed={seed}  {width}x{height}")
		base_image = base_pipe(
			cfg.base_prompt or cfg.prompt,
			height=height,
			width=width,
			num_inference_steps=int(cfg.steps),
			**_guidance(base_spec, cfg.guidance),
			generator=_generator(seed),
			callback_on_step_end=_make_callback(cb, int(cfg.steps)),
			**_neg(cfg, base_spec),
		).images[0]
		_save(base_image, cfg.out_dir,
		      f"{sanitize_filename(cfg.base_prompt or cfg.prompt)}_base_seed{seed}.png", cb)

		# Stage 2 - edit it.
		if cb.cancelled():
			raise Cancelled()
		cb.status(f"Stage 2/2: applying the edit ({i + 1}/{total})...")
		edit_pipe = PIPELINES.get(spec, IMG2IMG, cfg, cb)
		cb.log(f"Edit pass  seed={seed}  steps={cfg.steps}")
		edited = _img2img_call(edit_pipe, spec, cfg, base_image, cfg.prompt,
		                       cfg.steps, seed, cb)
		_save(edited, cfg.out_dir,
		      f"{sanitize_filename(cfg.prompt)}_edited_seed{seed}.png", cb)
		cb.items(i + 1, total)
