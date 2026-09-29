"""Describing a picture with a local vision-language model, through Ollama.

A VLM rather than a plain captioner, because the point is a rich description
- palette, material, lighting, art style - that can be fed straight back into
a generation prompt, not a one-line "a cat on a table".

The Ollama client itself (host resolution, availability, pull, unload,
response cleaning) lives in textgen.py and is reused from here rather than
duplicated. Both features talk to the same server and therefore share its
VRAM, so the Unload button and the automatic unload before Generate already
cover a vision model too.

Nothing here imports torch - it is all HTTP to localhost, plus Pillow to get
the picture into a format Ollama accepts.
"""

from dataclasses import dataclass

from . import textgen
from .textgen import INSTALL_URL, available, host, is_installed, pull, setup_hint


# A vision model has to load its image encoder as well as the language model,
# and answering about a picture is slower than rewriting a line of text, so
# this gets longer than textgen's 180 s - a cold 8B load plus inference on a
# first run can genuinely take a few minutes.
REQUEST_TIMEOUT = 420

# Ollama re-encodes what it is given anyway, and a 4K texture just makes the
# request bigger and the vision encoder slower for no extra detail.
MAX_EDGE = 1024


@dataclass(frozen=True)
class VisionModelSpec:
	key: str          # the exact name `ollama pull` expects
	label: str        # what the dropdown shows
	vram_note: str


# Smallest first, same as textgen's list: the image model is the one that
# needs the VRAM, so the cheap choice should be the easy one to reach for.
# Sizes are the actual download sizes from Ollama's registry.
VISION_MODELS = (
	VisionModelSpec(
		"moondream", "Moondream 2 (1.8B)",
		"~1.6 GB - tiny and quick, but brief; not much beyond a caption",
	),
	VisionModelSpec(
		"qwen3-vl:4b", "Qwen3-VL 4B",
		"~3.1 GB - most of the 8B quality at half the size",
	),
	VisionModelSpec(
		"llava:7b", "LLaVA 7B",
		"~4.4 GB - the old standard, noticeably vaguer than the Qwens",
	),
	VisionModelSpec(
		"minicpm-v:8b", "MiniCPM-V 2.6 (8B)",
		"~5.1 GB - strongest on fine detail and any text in the picture",
	),
	VisionModelSpec(
		"qwen2.5vl:7b", "Qwen2.5-VL 7B",
		"~5.6 GB - what most prompt-reversal workflows are built on",
	),
	VisionModelSpec(
		"qwen3-vl:8b", "Qwen3-VL 8B",
		"~5.7 GB - best all-round description at this size. Recommended",
	),
)

BY_KEY = {m.key: m for m in VISION_MODELS}
DEFAULT_MODEL = "qwen3-vl:8b"


@dataclass(frozen=True)
class DescribeStyle:
	key: str
	label: str
	instruction: str
	# Prompt-shaped answers want one line; prose wants its paragraphs kept.
	collapse: bool = True


STYLES = (
	DescribeStyle(
		"prompt", "Prompt for regenerating",
		"""Describe this image as a prompt for an AI image generator.

Cover, in this order: the subject, its important details, the material and
texture, the colour palette, the lighting, and the art style.

Rules:
- One line of comma-separated phrases, not sentences.
- Be concrete and visual. Name actual colours.
- Do not say "image", "picture", "photo" or "render" - describe what is in
  it, not that it is one.
- No interpretation of mood, story or meaning.
- Reply with the prompt only: no preamble, no explanation, no quotes, no
  markdown.
- Aim for 40 to 70 words.""",
	),
	DescribeStyle(
		"texture", "Texture / material",
		"""This picture is a surface or material texture. Describe it for
someone recreating it as a game texture.

Cover: what the material is, the colour palette with actual colour names,
the surface detail and how rough or smooth it is, any wear, damage, dirt or
staining, and how light behaves on it (matte, satin, glossy, metallic).

Rules:
- One line of comma-separated phrases, ready to paste into an image
  generator.
- Be concrete. No mood, no story, no interpretation.
- Reply with the description only: no preamble, no explanation, no quotes.
- Aim for 30 to 60 words.""",
	),
	DescribeStyle(
		"detailed", "Detailed description",
		"""Describe this image in vivid detail, for someone who cannot see it
and wants to recreate it.

Cover the colour palette naming the actual colours, the materials and
surface texture, the lighting and where it comes from, the composition and
where things sit in the frame, and the art style.

Rules:
- Two or three short paragraphs of plain prose.
- Be concrete and visual. No interpretation of mood or meaning.
- Reply with the description only: no preamble, no headings, no markdown.""",
		collapse=False,
	),
	DescribeStyle(
		"short", "Short caption",
		"""Caption this image in one short sentence.

Reply with the caption only: no preamble, no quotes, no markdown.""",
	),
)

STYLE_BY_KEY = {s.key: s for s in STYLES}
DEFAULT_STYLE = "prompt"


def encode_image(path, max_edge=MAX_EDGE):
	"""Return the picture as base64 PNG, which is what Ollama accepts.

	Ollama cannot decode DDS or TGA itself, so the conversion happens here -
	the same reason backend.py decodes a texture before handing it to a
	diffusion pipeline. Large pictures are shrunk first: the vision encoder
	works from a small fixed grid anyway, so sending 4K only costs time.
	"""
	import base64
	import io

	from PIL import Image

	from . import formats

	image = formats.open_image(path)
	if image.mode not in ("RGB", "L"):
		image = image.convert("RGB")
	if max(image.size) > max_edge:
		image.thumbnail((max_edge, max_edge), Image.LANCZOS)

	buffer = io.BytesIO()
	image.save(buffer, format="PNG")
	return base64.b64encode(buffer.getvalue()).decode("ascii")


def describe(path, model_key, style_key=DEFAULT_STYLE, timeout=REQUEST_TIMEOUT):
	"""Describe one picture. Raises RuntimeError with something actionable."""
	import requests

	style = STYLE_BY_KEY.get(style_key, STYLE_BY_KEY[DEFAULT_STYLE])
	payload = {
		"model": model_key,
		"messages": [{
			"role": "user",
			"content": style.instruction,
			"images": [encode_image(path)],
		}],
		"stream": False,
		# Low temperature: this is meant to report what is actually there,
		# not to write creatively about it.
		"options": {"temperature": 0.2, "top_p": 0.9},
		# Drop the model as soon as it has answered - a 6 GB VLM sitting
		# resident is exactly what the image model needs the room back from.
		"keep_alive": 0,
	}

	try:
		r = requests.post(host() + "/api/chat", json=payload, timeout=timeout)
	except Exception as exc:
		raise RuntimeError(
			f"Could not reach Ollama at {host()} ({exc}). Is it still running?"
		) from exc

	if r.status_code == 404:
		raise RuntimeError(
			f"Ollama does not have {model_key!r} yet - press Download first."
		)
	try:
		r.raise_for_status()
	except Exception as exc:
		raise RuntimeError(f"Ollama returned {r.status_code}: {r.text[:300]}") from exc

	content = (r.json().get("message") or {}).get("content", "")
	cleaned = textgen._clean(content, collapse=style.collapse)
	if not cleaned:
		raise RuntimeError(
			"The model returned an empty description. If it is not a vision "
			"model it will ignore the picture - check the model choice."
		)
	return cleaned
