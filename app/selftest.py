"""Headless check that a build can actually generate, not merely start.

A frozen build can open its window and still be broken: diffusers and
transformers resolve most of their API lazily, so a missing module only shows up
the moment a pipeline is built. Running

    AIImageStudio.exe --selftest

loads a real model, generates a small image and writes what happened to
selftest.log next to the executable. Exit code 0 means the build is sound.
"""

import sys
import time
import traceback
from pathlib import Path


def _out_dir():
	if getattr(sys, "frozen", False):
		return Path(sys.executable).parent
	return Path(__file__).resolve().parent.parent


def run():
	log_path = _out_dir() / "selftest.log"
	lines = []

	def say(text):
		lines.append(text)
		log_path.write_text("\n".join(lines), encoding="utf-8")
		print(text)

	started = time.time()
	say(f"AI Image Studio self-test  ({time.strftime('%Y-%m-%d %H:%M:%S')})")
	say(f"frozen={getattr(sys, 'frozen', False)}  python={sys.version.split()[0]}")

	try:
		import torch

		say(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
		if torch.cuda.is_available():
			p = torch.cuda.get_device_properties(0)
			say(f"gpu: {p.name}  {p.total_memory / 1024 ** 3:.1f} GB")

		import diffusers
		import transformers

		say(f"diffusers {diffusers.__version__}  transformers {transformers.__version__}")

		# Touch the exact pipeline classes the app uses, so a lazy-import gap
		# fails here with a clear message instead of mid-render.
		from diffusers import (
			StableDiffusion3Pipeline, StableDiffusion3Img2ImgPipeline,
			FluxKontextPipeline, FluxTransformer2DModel, GGUFQuantizationConfig,
			FluxPipeline, ZImagePipeline, ZImageTransformer2DModel,
			QwenImage21Pipeline, QwenImage21Transformer2DModel, Krea2Pipeline,
			PipelineQuantizationConfig,
			AutoPipelineForText2Image, AutoPipelineForImage2Image,
		)

		say("pipeline classes import OK")

		from . import backend
		from .backend import JobConfig, Progress

		class Say(Progress):
			def log(self, text):
				say("  " + text)

			def status(self, text):
				pass

			def cancelled(self):
				return False

		say("generating a 512x512 test image with SD3.5 Medium...")
		out = _out_dir() / "Output" / "_selftest"
		backend.run_job(JobConfig(
			mode=backend.MODE_TXT2IMG,
			model_key="sd35-medium",
			prompt="a red cube on a white table",
			out_dir=str(out),
			width=512, height=512, steps=4, guidance=7.0,
			seed=1234, random_seed=False, num_images=1,
		), Say())

		files = sorted(out.glob("*.png"))
		if not files:
			say("FAILED: the job reported success but wrote no file")
			return 1
		say(f"wrote {files[-1].name}")
		say(f"PASSED in {time.time() - started:.0f}s")
		return 0

	except Exception:
		say("FAILED:\n" + traceback.format_exc())
		return 1
