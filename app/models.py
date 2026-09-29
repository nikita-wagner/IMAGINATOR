"""Registry of the diffusion models the app can run.

Each entry says which task it serves (text->image or image->image), which
HuggingFace repo it comes from, and which loader `backend.py` should use.
Adding a new model is a matter of appending a ModelSpec here.
"""

from dataclasses import dataclass, field
from pathlib import Path


TXT2IMG = "txt2img"
IMG2IMG = "img2img"


@dataclass(frozen=True)
class ModelSpec:
	key: str          # stable id used in saved settings
	label: str        # what the dropdown shows
	repo: str         # HuggingFace repo id
	kind: str         # which loader branch in backend.py handles it
	tasks: tuple      # (TXT2IMG,), (IMG2IMG,) or both
	vram_note: str = ""
	# Extra quantisation choices, e.g. the FLUX GGUF files. Empty = no choice.
	quants: tuple = field(default_factory=tuple)
	default_quant: str = ""
	# Where the quantised transformer lives, when it isn't in `repo` itself.
	# gguf_file is formatted with the chosen quant key.
	gguf_repo: str = ""
	gguf_file: str = ""
	# Components squeezed to 4-bit (bitsandbytes NF4) while loading, for the
	# ones that would not fit a 12 GB card even one module at a time.
	quantize_4bit: tuple = field(default_factory=tuple)
	# A LoRA applied on top of the base weights after loading.
	lora_repo: str = ""
	# Sensible starting values, merged over the global defaults.
	defaults: dict = field(default_factory=dict)


# FLUX.1-Kontext-dev transformer quants published by QuantStack. Smaller = less
# VRAM and faster, but progressively lossier. Q4_K_M is the usual sweet spot.
FLUX_QUANTS = (
	("Q2_K", "Q2_K  (~4.0 GB, fastest / lowest quality)"),
	("Q3_K_M", "Q3_K_M  (~5.4 GB)"),
	("Q4_K_S", "Q4_K_S  (~6.7 GB)"),
	("Q4_K_M", "Q4_K_M  (~7.0 GB, recommended)"),
	("Q5_K_M", "Q5_K_M  (~8.4 GB, best quality / slowest)"),
)

# FLUX.1-dev transformer quants published by city96.
FLUX_DEV_QUANTS = (
	("Q2_K", "Q2_K  (~4.0 GB, fastest / lowest quality)"),
	("Q3_K_S", "Q3_K_S  (~5.2 GB)"),
	("Q4_K_S", "Q4_K_S  (~6.8 GB, recommended)"),
	("Q5_K_S", "Q5_K_S  (~8.3 GB)"),
	("Q6_K", "Q6_K  (~9.9 GB)"),
	("Q8_0", "Q8_0  (~12.7 GB, best quality / slowest)"),
)

# Tencent SRPO (a FLUX.1-dev finetune) quants published by befox. The
# official repo only has a 47 GB fp32 file, far too big to load here.
SRPO_QUANTS = (
	("Q2_K", "Q2_K  (~4.0 GB, fastest / lowest quality)"),
	("Q3_K", "Q3_K  (~5.4 GB)"),
	("Q4_K", "Q4_K  (~6.9 GB, recommended)"),
	("Q5_K", "Q5_K  (~8.4 GB)"),
	("Q6_K", "Q6_K  (~9.9 GB)"),
	("Q8_0", "Q8_0  (~12.7 GB, best quality / slowest)"),
)

# Z-Image-Turbo transformer quants published by unsloth.
ZIMAGE_QUANTS = (
	("Q2_K", "Q2_K  (~3.6 GB, fastest / lowest quality)"),
	("Q3_K_S", "Q3_K_S  (~4.0 GB)"),
	("Q3_K_M", "Q3_K_M  (~4.2 GB)"),
	("Q4_0", "Q4_0  (~4.6 GB)"),
	("Q4_K_S", "Q4_K_S  (~4.7 GB)"),
	("Q4_1", "Q4_1  (~4.9 GB)"),
	("Q4_K_M", "Q4_K_M  (~5.0 GB, recommended)"),
	("Q5_K_S", "Q5_K_S  (~5.2 GB)"),
	("Q5_0", "Q5_0  (~5.3 GB)"),
	("Q5_1", "Q5_1  (~5.5 GB)"),
	("Q5_K_M", "Q5_K_M  (~5.6 GB)"),
	("Q6_K", "Q6_K  (~5.9 GB)"),
	("Q8_0", "Q8_0  (~7.2 GB)"),
	("BF16", "BF16  (~12.3 GB, unquantised / slowest)"),
	("F16", "F16  (~12.3 GB, unquantised / slowest)"),
)

# Qwen-Image-2.1 Uncensored quants published by abenzerps. Only the GGUFs:
# the fp8 / int8-ConvRot files are ComfyUI-specific, MLX is Apple-only, and
# NVFP4 needs an RTX 50-series card.
QWEN21_QUANTS = (
	("Q4_0", "Q4_0  (~4.2 GB, fastest / lowest quality)"),
	("Q4_K_M", "Q4_K_M  (~4.6 GB, recommended)"),
	("Q5_K_M", "Q5_K_M  (~5.2 GB)"),
	("Q6_K", "Q6_K  (~5.9 GB)"),
	("Q8_0", "Q8_0  (~7.6 GB)"),
	("BF16", "BF16  (~14.2 GB, unquantised / slowest)"),
)


MODELS = (
	ModelSpec(
		key="sd35-medium",
		label="Stable Diffusion 3.5 Medium",
		repo="stabilityai/stable-diffusion-3.5-medium",
		kind="sd3",
		tasks=(TXT2IMG, IMG2IMG),
		vram_note="~8 GB VRAM (T5-XXL text encoder skipped)",
		defaults={"width": 1024, "height": 1024, "steps": 60, "guidance": 7.5},
	),
	ModelSpec(
		key="sd35-large",
		label="Stable Diffusion 3.5 Large",
		repo="stabilityai/stable-diffusion-3.5-large",
		kind="sd3",
		tasks=(TXT2IMG, IMG2IMG),
		vram_note="~16 GB VRAM - will offload to CPU on a 12 GB card (slow)",
		defaults={"width": 1024, "height": 1024, "steps": 40, "guidance": 4.5},
	),
	ModelSpec(
		key="sd35-large-turbo",
		label="Stable Diffusion 3.5 Large Turbo",
		repo="stabilityai/stable-diffusion-3.5-large-turbo",
		kind="sd3",
		tasks=(TXT2IMG, IMG2IMG),
		vram_note="Distilled: use ~4 steps and guidance 0.0",
		defaults={"width": 1024, "height": 1024, "steps": 4, "guidance": 0.0},
	),
	ModelSpec(
		key="flux-kontext-gguf",
		label="FLUX.1 Kontext dev (GGUF)",
		repo="black-forest-labs/FLUX.1-Kontext-dev",
		kind="flux-kontext-gguf",
		tasks=(IMG2IMG,),
		vram_note="Instruction-based editor. CPU offload on for 12 GB cards.",
		quants=FLUX_QUANTS,
		default_quant="Q4_K_M",
		gguf_repo="QuantStack/FLUX.1-Kontext-dev-GGUF",
		gguf_file="flux1-kontext-dev-{quant}.gguf",
		# Kontext follows an instruction ("make the sky red") rather than
		# re-describing the scene, and ignores `strength` entirely.
		defaults={"steps": 28, "guidance": 2.5, "width": 0, "height": 0},
	),
	ModelSpec(
		key="instruct-pix2pix",
		label="InstructPix2Pix",
		repo="timbrooks/instruct-pix2pix",
		kind="instruct-pix2pix",
		tasks=(IMG2IMG,),
		vram_note="~4 GB VRAM. Instruction-based editor - say what to change.",
		# Follows an instruction ("make it night") rather than re-describing
		# the scene, and (like FLUX Kontext) ignores `strength` entirely.
		defaults={"steps": 20, "guidance": 7.5, "width": 0, "height": 0},
	),
	ModelSpec(
		key="sdxl-base",
		label="Stable Diffusion XL 1.0 Base",
		repo="stabilityai/stable-diffusion-xl-base-1.0",
		kind="auto",
		tasks=(TXT2IMG, IMG2IMG),
		vram_note="~7 GB VRAM. Downloads on first use.",
		defaults={"width": 1024, "height": 1024, "steps": 30, "guidance": 7.0},
	),
	ModelSpec(
		key="sd15",
		label="Stable Diffusion 1.5",
		repo="stable-diffusion-v1-5/stable-diffusion-v1-5",
		kind="auto",
		tasks=(TXT2IMG, IMG2IMG),
		vram_note="~4 GB VRAM, 512x512 native. Downloads on first use.",
		defaults={"width": 512, "height": 512, "steps": 30, "guidance": 7.5},
	),
	ModelSpec(
		key="flux-dev-gguf",
		label="FLUX.1 dev (GGUF)",
		repo="black-forest-labs/FLUX.1-dev",
		kind="flux-gguf",
		tasks=(TXT2IMG, IMG2IMG),
		vram_note="CPU offload on for 12 GB cards. No negative prompt.",
		quants=FLUX_DEV_QUANTS,
		default_quant="Q4_K_S",
		gguf_repo="city96/FLUX.1-dev-gguf",
		gguf_file="flux1-dev-{quant}.gguf",
		# FLUX.1-dev is guidance-distilled: guidance is an input to the model,
		# not classifier-free guidance, and ~3.5 is its trained sweet spot.
		defaults={"width": 1024, "height": 1024, "steps": 28, "guidance": 3.5},
	),
	ModelSpec(
		key="srpo-gguf",
		label="Tencent SRPO (FLUX.1 dev finetune, GGUF)",
		# Only the transformer differs; text encoders and VAE are FLUX.1-dev's.
		repo="black-forest-labs/FLUX.1-dev",
		kind="flux-gguf",
		tasks=(TXT2IMG, IMG2IMG),
		vram_note="More realistic FLUX.1 dev. CPU offload on. No negative prompt.",
		quants=SRPO_QUANTS,
		default_quant="Q4_K",
		gguf_repo="befox/SRPO-GGUF",
		gguf_file="srpo-{quant}.gguf",
		defaults={"width": 1024, "height": 1024, "steps": 30, "guidance": 3.5},
	),
	ModelSpec(
		key="zimage-turbo-gguf",
		label="Z-Image Turbo (GGUF)",
		repo="Tongyi-MAI/Z-Image-Turbo",
		kind="zimage-gguf",
		tasks=(TXT2IMG, IMG2IMG),
		vram_note="Distilled: ~9 steps and guidance 0.0. CPU offload on.",
		quants=ZIMAGE_QUANTS,
		default_quant="Q4_K_M",
		gguf_repo="unsloth/Z-Image-Turbo-GGUF",
		gguf_file="z-image-turbo-{quant}.gguf",
		defaults={"width": 1024, "height": 1024, "steps": 9, "guidance": 0.0},
	),
	ModelSpec(
		key="qwen-image-21-uc-gguf",
		label="Qwen-Image 2.1 Uncensored (GGUF)",
		repo="Qwen/Qwen-Image-2.1",
		kind="qwen21-gguf",
		# One pipeline does both: pass an image and it edits it instead.
		tasks=(TXT2IMG, IMG2IMG),
		vram_note=("Text encoder squeezed to 4-bit. Also an instruction editor. "
		           "Guidance >1 needs a negative prompt."),
		quants=QWEN21_QUANTS,
		default_quant="Q4_K_M",
		gguf_repo="abenzerps/Qwen-Image-2.1-Uncensored-GGUF",
		gguf_file="qwen-image-2.1-UC-{quant}.gguf",
		# The 17.5 GB Qwen3-VL text encoder cannot be streamed onto a 12 GB
		# card whole, even with model offload.
		quantize_4bit=("text_encoder",),
		# Guidance maps to true CFG here, which doubles the cost of every step;
		# 1.0 leaves it off, as the model card's own examples do.
		defaults={"width": 1024, "height": 1024, "steps": 40, "guidance": 1.0},
	),
	ModelSpec(
		key="krea2-raven",
		label="Krea 2 Turbo + raven LoRA",
		repo="krea/Krea-2-Turbo",
		kind="krea2",
		tasks=(TXT2IMG,),
		vram_note=("Trigger word: 'vzx woman'. Gated - accept the licence on "
		           "huggingface.co/krea/Krea-2-Turbo first."),
		# Krea 2 has no GGUF path in diffusers, so the 26 GB transformer is
		# quantised on load instead.
		quantize_4bit=("transformer", "text_encoder"),
		lora_repo="September6969/raven",
		defaults={"width": 1024, "height": 1024, "steps": 8, "guidance": 0.0},
	),
)


BY_KEY = {m.key: m for m in MODELS}


def models_for(task):
	"""Every model that can serve the given task."""
	return [m for m in MODELS if task in m.tasks]


def hf_cache_dir():
	"""Where huggingface_hub keeps downloaded repos."""
	import os

	env = os.environ.get("HF_HOME")
	if env:
		return Path(env) / "hub"
	return Path.home() / ".cache" / "huggingface" / "hub"


def quant_for(spec, requested):
	"""The quant to load: the requested one if this model has it, else its default.

	The chain mode loads two models with one quant setting, and the quant
	names differ between repos (SRPO has Q4_K, FLUX.1-dev has Q4_K_S).
	"""
	if not spec.quants:
		return None
	if requested in {k for k, _ in spec.quants}:
		return requested
	return spec.default_quant or spec.quants[0][0]


def _repo_cached(repo):
	snapshots = hf_cache_dir() / ("models--" + repo.replace("/", "--")) / "snapshots"
	return snapshots.is_dir() and any(snapshots.iterdir())


def is_cached(spec):
	"""True if every repo the model needs already has a local snapshot, so no
	multi-GB download.

	Purely advisory - used to mark models in the dropdown, never to block them.
	"""
	repos = [spec.repo, spec.gguf_repo, spec.lora_repo]
	return all(_repo_cached(r) for r in repos if r)
