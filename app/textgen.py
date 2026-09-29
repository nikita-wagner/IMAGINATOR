"""Prompt rewriting with a small local LLM, served by Ollama.

Ollama runs the model rather than this app loading GGUF weights itself, for
two practical reasons:

* there is no prebuilt llama-cpp-python wheel for this Python version, and
  building it from source needs a C++ toolchain;
* the transformers route would mean downloading full fp16 weights (~15 GB for
  a 7B) to end up with worse VRAM behaviour than a ~5 GB Q4 GGUF.

Ollama also unloads an idle model on its own, which matters here: the image
model in the worker process already wants most of the 12 GB.

Nothing in this module imports torch - it is all HTTP to localhost.
"""

import json
import os
import re
from dataclasses import dataclass


# 127.0.0.1, not "localhost": on Windows localhost resolves to ::1 first, and
# Ollama listens on IPv4 only by default, so every single request would eat a
# ~2 s connect-failure penalty before falling back. Measured here: 2.03 s via
# localhost against 0.001 s via 127.0.0.1.
DEFAULT_HOST = "http://127.0.0.1:11434"
# A 7B model answering on CPU only can take a while; a GPU answers in seconds.
REQUEST_TIMEOUT = 180
INSTALL_URL = "https://ollama.com/download"


@dataclass(frozen=True)
class TextModelSpec:
	key: str          # the exact name `ollama pull` expects
	label: str        # what the dropdown shows
	vram_note: str


# The models are listed smallest first: on a 12 GB card the image model is the
# one that needs the room, so the safe choice should be the easy one to pick.
MODELS = (
	TextModelSpec(
		"llama3.2:3b", "Llama 3.2 3B Instruct",
		"~2 GB - leaves almost all VRAM for the image model",
	),
	TextModelSpec(
		"phi4-mini", "Phi-4 mini",
		"~2.5 GB - leaves almost all VRAM for the image model",
	),
	TextModelSpec(
		"mistral:7b", "Mistral 7B Instruct v0.3",
		"~4-5 GB - fast and reliable",
	),
	TextModelSpec(
		"qwen2.5:7b", "Qwen2.5 7B Instruct",
		"~5 GB - strong at structured rewriting, good in other languages",
	),
	TextModelSpec(
		"llama3.1:8b", "Llama 3.1 8B Instruct",
		"~5-6 GB - the most capable of these, least room to spare",
	),
)

BY_KEY = {m.key: m for m in MODELS}
DEFAULT_MODEL = "llama3.2:3b"


SYSTEM_PROMPT = """You rewrite short prompts for an image-generation model.

Fix spelling and grammar, then restructure the text into one clear prompt:
the subject first, then its important details, then style, lighting and
quality words.

Rules:
- Keep the user's intent and every named subject exactly as given. Never
  swap the subject for a different one.
- Add only detail that is clearly implied. Do not invent a new scene.
- Reply with the rewritten prompt and nothing else: no preamble, no
  explanation, no quotes, no markdown, no bullet points.
- Keep it under 60 words, comma-separated."""


SYSTEM_NEGATIVE = """You rewrite "negative prompts" for an image-generation
model. A negative prompt is a list of qualities the picture should NOT have.

Fix spelling and grammar and return a clean, comma-separated list of
undesirable traits.

Rules:
- Keep everything the user already listed.
- Every item must be something to avoid, never something to include.
- Reply with the list and nothing else: no preamble, no explanation, no
  quotes, no markdown, no bullet points.
- Keep it under 40 words."""


def host():
	"""Base URL of the Ollama server, honouring Ollama's own OLLAMA_HOST var."""
	raw = (os.environ.get("OLLAMA_HOST") or "").strip().rstrip("/")
	if not raw:
		return DEFAULT_HOST
	if not raw.startswith(("http://", "https://")):
		raw = "http://" + raw
	return raw


def setup_hint():
	"""What to tell the user when no Ollama server answers."""
	return (
		f"No Ollama server is answering at {host()}.\n\n"
		"The prompt improver runs a small language model locally through "
		"Ollama.\n\n"
		f"1. Install Ollama from {INSTALL_URL}\n"
		"2. It starts by itself once installed.\n"
		"3. Come back here and press Check.\n\n"
		"The language model itself downloads on first use - about 2 GB for "
		"Llama 3.2 3B."
	)


def available():
	"""(ok, message) - whether an Ollama server is reachable right now."""
	try:
		import requests

		r = requests.get(host() + "/api/tags", timeout=4)
		r.raise_for_status()
		return True, "Ollama is running."
	except Exception:
		return False, f"Ollama is not reachable at {host()}."


def installed_models():
	"""Model names Ollama already holds locally. Empty set if it is not up."""
	try:
		import requests

		r = requests.get(host() + "/api/tags", timeout=4)
		r.raise_for_status()
		names = set()
		for entry in r.json().get("models", []):
			name = entry.get("name") or entry.get("model") or ""
			if not name:
				continue
			names.add(name)
			# Ollama reports "phi4-mini:latest"; record the bare name too so a
			# spec key without an explicit tag still matches.
			names.add(name.split(":")[0])
		return names
	except Exception:
		return set()


def is_installed(model_key, names=None):
	names = installed_models() if names is None else names
	return model_key in names


def loaded_models():
	"""Models Ollama currently has resident (in VRAM or RAM), from /api/ps.

	Each entry includes a `size_vram` in bytes when the model is (fully or
	partly) on the GPU - that is what actually competes with the image model.
	Empty list if nothing is loaded or the server is not reachable.
	"""
	try:
		import requests

		r = requests.get(host() + "/api/ps", timeout=4)
		r.raise_for_status()
		return r.json().get("models", [])
	except Exception:
		return []


def unload(model_key):
	"""Ask Ollama to drop one model from memory right now.

	This is the same call `improve()` makes implicitly via keep_alive=0 - a
	generate request with no prompt and keep_alive=0 unloads instead of
	loading. Safe to call even if the model was never loaded, or if Ollama
	is not running at all (it just does nothing).
	"""
	try:
		import requests

		requests.post(
			host() + "/api/generate",
			json={"model": model_key, "keep_alive": 0},
			timeout=10,
		)
		return True
	except Exception:
		return False


def unload_all():
	"""Drop every model Ollama currently has loaded. Returns how many."""
	models = loaded_models()
	for entry in models:
		name = entry.get("name") or entry.get("model")
		if name:
			unload(name)
	return len(models)


def _pull_line(msg):
	"""Turn one streamed /api/pull JSON line into a short status string."""
	status = msg.get("status", "")
	total, done = msg.get("total"), msg.get("completed")
	if total and done:
		pct = 100.0 * done / total
		return f"{status} - {pct:.0f}% of {total / 1024 ** 3:.1f} GB"
	return status


def pull(model_key, on_progress=None):
	"""Download a model into Ollama, reporting progress lines as they stream."""
	import requests

	with requests.post(
		host() + "/api/pull",
		json={"model": model_key, "stream": True},
		stream=True,
		timeout=(10, None),      # no read timeout: a big pull takes minutes
	) as r:
		r.raise_for_status()
		for line in r.iter_lines():
			if not line:
				continue
			try:
				msg = json.loads(line)
			except ValueError:
				continue
			if msg.get("error"):
				raise RuntimeError(msg["error"])
			if on_progress:
				text = _pull_line(msg)
				if text:
					on_progress(text)


def _clean(text, collapse=True):
	"""Strip the preamble and quoting an instruct model adds anyway.

	Even told firmly not to, these models like to answer with "Here is the
	rewritten prompt:", wrap the result in quotes, or fence it in markdown.

	`collapse` folds the answer onto one line, which is what a prompt wants.
	Pass False for prose meant to keep its paragraphs - vision.py does that
	for its longer description styles.
	"""
	# Some models emit a visible chain-of-thought block first.
	text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
	text = text.strip()

	if text.startswith("```"):
		text = re.sub(r"^```[A-Za-z]*\n?", "", text)
		text = re.sub(r"\n?```\s*$", "", text).strip()

	lines = text.split("\n")
	if len(lines) > 1 and re.match(
		r"^(here|sure|certainly|okay|ok|rewritten|improved|revised|prompt)\b.*:\s*$",
		lines[0].strip(), re.IGNORECASE,
	):
		text = "\n".join(lines[1:]).strip()

	# A single leading bullet is common when the model ignores the no-list rule.
	text = re.sub(r"^\s*[-*•]\s+", "", text)

	if len(text) >= 2 and text[0] in "\"'“" and text[-1] in "\"'”":
		text = text[1:-1].strip()

	if collapse:
		text = " ".join(part.strip() for part in text.split("\n") if part.strip())
	return text.strip()


def improve(text, model_key, negative=False, timeout=REQUEST_TIMEOUT):
	"""Rewrite one prompt. Raises RuntimeError with something actionable."""
	import requests

	payload = {
		"model": model_key,
		"messages": [
			{"role": "system", "content": SYSTEM_NEGATIVE if negative else SYSTEM_PROMPT},
			{"role": "user", "content": text.strip()},
		],
		"stream": False,
		# Low temperature: this is a tidy-up, not a creative writing task.
		"options": {"temperature": 0.4, "top_p": 0.9},
		# 0 = unload right after this reply. The image model wants most of
		# the VRAM on the card this was built for, so the text model should
		# not linger just in case another rewrite comes soon - a cold load
		# on the next Improve click is a fair trade for never competing with
		# a running generation.
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
	cleaned = _clean(content)
	if not cleaned:
		raise RuntimeError("The model returned an empty rewrite.")
	return cleaned
