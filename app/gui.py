"""Tkinter front-end for the diffusion pipelines in backend.py.

The heavy imports (torch, diffusers) deliberately happen inside the worker
thread on the first run, so the window appears instantly instead of after the
~20 s it takes to import torch.
"""

import json
import multiprocessing as mp
import os
import queue
import random
import re
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from queue import Empty as QueueEmpty

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
	# Optional: enables dragging files from Explorer onto the Files panel.
	# Plain Tkinter has no drag-and-drop of its own. Everything still works
	# without this - _setup_dnd() below just does nothing if it's missing.
	from tkinterdnd2 import TkinterDnD, DND_FILES, COPY
	HAS_DND = True
except ImportError:
	HAS_DND = False

from . import models as registry
from . import backend
from . import formats
from . import textgen
from . import vision
from . import batch
from . import hftoken
from .thumbgrid import ThumbnailGrid
from .backend import (
	JobConfig,
	MODE_ORDER, MODE_LABELS, MODE_TASK,
	MODE_TXT2IMG, MODE_IMG2IMG, MODE_BATCH, MODE_SWEEP, MODE_CHAIN,
)
from .models import TXT2IMG, IMG2IMG
from .worker_process import worker_main


APP_NAME = "AI Image Studio"

# How many results the gallery remembers, and keeps between sessions. Older
# ones drop off the end of the list - the files themselves are never touched.
MAX_RESULTS = 300

MEMORY_MODES = (
	("auto", "Auto (recommended)"),
	("none", "Keep everything on the GPU (fastest)"),
	("cpu_offload", "Model CPU offload (less VRAM)"),
	("sequential_offload", "Sequential CPU offload (least VRAM, slowest)"),
)


def app_dir():
	"""Folder the app lives in - next to the .exe when frozen."""
	if getattr(sys, "frozen", False):
		return Path(sys.executable).parent
	return Path(__file__).resolve().parent.parent


def settings_path():
	base = os.environ.get("LOCALAPPDATA") or str(Path.home())
	folder = Path(base) / "AIImageStudio"
	folder.mkdir(parents=True, exist_ok=True)
	return folder / "settings.json"


class ProcessWorker:
	"""Runs generation jobs in a child process and forwards its progress
	messages onto `ui_queue` (the same queue.Queue the GUI drains on a timer).

	Cancelling hard-kills the process - the only way to guarantee a stuck
	model load or denoising step actually stops - and a fresh process is
	spawned lazily for the next job.
	"""

	def __init__(self, ui_queue):
		self.ui_queue = ui_queue
		self.process = None
		self.job_queue = None
		self.progress_queue = None
		self.busy = False

	def is_running(self):
		return self.process is not None and self.process.is_alive()

	def _spawn(self):
		ctx = mp.get_context("spawn")
		self.job_queue = ctx.Queue()
		self.progress_queue = ctx.Queue()
		self.process = ctx.Process(
			target=worker_main, args=(self.job_queue, self.progress_queue),
			daemon=True,
		)
		self.process.start()
		threading.Thread(
			target=self._pump, args=(self.process, self.progress_queue),
			daemon=True,
		).start()

	def _pump(self, proc, pq):
		"""Forward messages from one process's queue until that process has
		died and its queue has run dry. `proc`/`pq` are captured by argument
		so a pump thread for a process that was just killed recognises its
		own queue is done and exits, rather than confusing itself with
		whatever process happens to be current by the time it wakes up."""
		while True:
			try:
				item = pq.get(timeout=0.2)
			except QueueEmpty:
				if not proc.is_alive():
					return
				continue
			except (EOFError, OSError, ValueError):
				return
			if item[0] == "worker_ready":
				continue
			self.ui_queue.put(item)

	def submit(self, cfg):
		if not self.is_running():
			self._spawn()
		self.busy = True
		self.job_queue.put(cfg)

	def stop(self):
		"""Hard-kill the worker process, if one is running. Always effective:
		it does not matter whether the process is idle, loading a model, or
		mid-denoise - TerminateProcess ends it immediately."""
		if self.process is not None:
			if self.process.is_alive():
				self.process.kill()
				self.process.join(timeout=3)
			self.process = None
		self.job_queue = None
		self.progress_queue = None
		self.busy = False


class App(ttk.Frame):
	def __init__(self, master, scale=1.0):
		super().__init__(master, padding=10)
		self.master = master
		self.scale = scale
		self.grid(row=0, column=0, sticky="nsew")
		master.rowconfigure(0, weight=1)
		master.columnconfigure(0, weight=1)

		self.queue = queue.Queue()
		self.worker = ProcessWorker(self.queue)
		self.textgen_busy = False      # one prompt rewrite/download/unload at a time
		self.vision_busy = False       # one describe/download at a time
		self.vision_available = False  # Ollama up at all
		self.textgen_ready = False     # Ollama up and the chosen model present
		self.textgen_available = False # Ollama up at all (enough to Unload)
		self.hf_busy = False           # one token save/remove/check at a time
		self.preview_image = None      # PhotoImage, kept alive
		self.preview_source = None     # path of the image currently previewed
		self.preview_origin = None     # the pre-existing file it was edited from, if any
		self._model_keys = []
		self._base_model_keys = []
		self._files_enabled = True     # False in text-only tasks
		self._select_anchor = None     # for Shift-click range selection in grid view
		# The results gallery: every picture saved so far, oldest first, each
		# with the input it was edited from (None for text-to-image).
		self.results = []
		self._results_selected = set()   # by path, so it survives reordering
		self._results_anchor = None
		self._batch_busy = False         # an Export all / Replace all is running
		self._sash_fractions = None      # divider positions restored from settings
		self._left_width = None          # controls column width from settings
		self._left_target = None         # px the controls want; None = the default
		self._last_app_w = 0

		self._build_vars()
		self._build_ui()
		self._load_settings()
		self._on_mode_change()
		self._refresh_textgen()
		self._refresh_vision()
		self._refresh_hf()
		self.after(80, self._drain_queue)
		self.after(100, self._place_sashes)
		master.protocol("WM_DELETE_WINDOW", self._on_close)

	# -- state ------------------------------------------------------------
	def _build_vars(self):
		root = app_dir()
		self.v_mode = tk.StringVar(value=MODE_TXT2IMG)
		self.v_model = tk.StringVar()
		self.v_base_model = tk.StringVar()
		self.v_quant = tk.StringVar()
		self.v_memory = tk.StringVar(value=MEMORY_MODES[0][1])
		self.v_prompt_hint = tk.StringVar()
		self.v_out = tk.StringVar(value=str(root / "Output"))
		self.v_width = tk.StringVar(value="1024")
		self.v_height = tk.StringVar(value="1024")
		self.v_steps = tk.StringVar(value="30")
		self.v_guidance = tk.StringVar(value="7.5")
		self.v_strength = tk.StringVar(value="0.6")
		self.v_seed = tk.StringVar(value="251217772")
		self.v_random_seed = tk.BooleanVar(value=True)
		self.v_count = tk.StringVar(value="1")
		self.v_sweep_min = tk.StringVar(value="1")
		self.v_sweep_max = tk.StringVar(value="30")
		self.v_autodefaults = tk.BooleanVar(value=True)
		self.v_status = tk.StringVar(value="Ready.")
		self.v_device = tk.StringVar(value="Checking hardware...")
		self.v_modelnote = tk.StringVar(value="")
		self.v_filecount = tk.StringVar(value="No input files selected.")
		default_text = textgen.BY_KEY[textgen.DEFAULT_MODEL]
		self.v_textmodel = tk.StringVar(value=default_text.label)
		self.v_textstatus = tk.StringVar(value="Checking for Ollama...")
		default_export = formats.BY_KEY[formats.DEFAULT_EXPORT]
		self.v_export = tk.StringVar(value=default_export.label)
		self.v_export_note = tk.StringVar(value=default_export.note)
		default_vision = vision.BY_KEY[vision.DEFAULT_MODEL]
		self.v_vision_model = tk.StringVar(value=default_vision.label)
		self.v_vision_style = tk.StringVar(
			value=vision.STYLE_BY_KEY[vision.DEFAULT_STYLE].label)
		self.v_vision_status = tk.StringVar(value="Checking for Ollama...")
		self.v_results_status = tk.StringVar(value="")
		# Never saved to settings.json - the entry is cleared once the token
		# is stored in HuggingFace's own token file.
		self.v_hf_token = tk.StringVar()
		self.v_hf_show = tk.BooleanVar(value=False)
		self.v_hf_status = tk.StringVar(value="Checking the HuggingFace token...")

	# -- layout -----------------------------------------------------------
	def px(self, value):
		"""Scale a hand-tuned pixel size for the current display DPI."""
		return int(value * self.scale)

	def _build_ui(self):
		self.columnconfigure(0, weight=1)
		self.rowconfigure(1, weight=1)

		self._build_topbar().grid(row=0, column=0, sticky="ew", pady=(0, 8))

		# Controls on the left, preview side on the right, with a divider that
		# can be dragged. A classic tk.PanedWindow rather than ttk: only it has
		# a per-pane minsize, so neither side can be dragged away to nothing,
		# and stretch, which keeps window resizes going to the preview side
		# while the controls hold whatever width they were given.
		self.main_pane = tk.PanedWindow(
			self, orient="horizontal", sashwidth=self.px(6), sashpad=self.px(1),
			sashrelief="flat", sashcursor=self._hresize_cursor(), showhandle=False,
			borderwidth=0,
			background=ttk.Style().lookup("TFrame", "background") or "SystemButtonFace")
		self.main_pane.grid(row=1, column=0, sticky="nsew")
		self.bind("<Configure>", self._on_app_resize)
		self.main_pane.bind("<ButtonRelease-1>", self._remember_left_width, add="+")

		# The left column is tabbed: Generate holds every control a run needs,
		# Settings the app-wide things you set once (the HuggingFace token).
		ttk.Style().configure("TNotebook.Tab", padding=(self.px(12), self.px(3)))
		self.left_tabs = ttk.Notebook(self.main_pane)

		left_wrap, left, self.left_canvas = self._scroll_page(self.left_tabs)
		self.left_wrap = left_wrap
		self._build_prompts(left).grid(row=0, column=0, sticky="ew", pady=(0, 8))
		self._build_files(left).grid(row=1, column=0, sticky="ew", pady=(0, 8))
		self._build_settings(left).grid(row=2, column=0, sticky="ew", pady=(0, 8))
		self._build_actions(left).grid(row=3, column=0, sticky="ew")
		self._bind_wheel(left, self.left_canvas)
		self.left_tabs.add(left_wrap, text="Generate")

		settings_wrap, settings_page, settings_canvas = self._scroll_page(self.left_tabs)
		self._build_settings_tab(settings_page)
		self._bind_wheel(settings_page, settings_canvas)
		self.left_tabs.add(settings_wrap, text="Settings")
		self.main_pane.add(self.left_tabs, minsize=self.px(400), width=self.px(510),
		                   stretch="never")

		# Preview, the results gallery right under it, then Describe image and
		# the Log sharing one tabbed panel. Four stacked panels don't fit a
		# laptop screen - the preview had already been squeezed to a strip -
		# so the two that are read rather than watched share a tab strip, and
		# the dividers can be dragged to give any of the three more room.
		#
		# That side scrolls as well. Preview, results, Describe image and the
		# Log stack up to more height than a laptop screen has, and a
		# PanedWindow just squeezes whatever does not fit. So the PanedWindow
		# sits in a scrolling canvas and is never made shorter than its
		# panels need; on a window tall enough it fills the space and there
		# is nothing to scroll.
		right_wrap = ttk.Frame(self.main_pane)
		right_wrap.rowconfigure(0, weight=1)
		right_wrap.columnconfigure(0, weight=1)
		self.right_canvas = tk.Canvas(right_wrap, highlightthickness=0, borderwidth=0)
		right_vsb = ttk.Scrollbar(right_wrap, orient="vertical",
		                          command=self.right_canvas.yview)
		self.right_canvas.configure(yscrollcommand=right_vsb.set)
		self.right_canvas.grid(row=0, column=0, sticky="nsew")
		right_vsb.grid(row=0, column=1, sticky="ns")

		self.right_pane = ttk.PanedWindow(self.right_canvas, orient="vertical")
		self._right_window = self.right_canvas.create_window(
			(0, 0), window=self.right_pane, anchor="nw")
		self.right_pane.add(self._build_preview(self.right_pane), weight=3)
		self.right_pane.add(self._build_results(self.right_pane), weight=2)
		self.right_pane.add(self._build_describe(self.right_pane), weight=2)
		self.right_pane.add(self._build_log(self.right_pane), weight=2)
		self.right_canvas.bind("<Configure>", self._fit_right_pane)
		self._bind_wheel(self.right_pane, self.right_canvas)

		self.main_pane.add(right_wrap, minsize=self.px(380), stretch="always")

	def _scroll_page(self, parent):
		"""A tab page that scrolls vertically: (outer frame, inner frame, canvas).

		The controls do not fit on a short screen, so they live on a canvas
		that scrolls rather than silently clipping the bottom of the panel.
		Build into the inner frame, add the outer one to the notebook.
		"""
		wrap = ttk.Frame(parent)
		wrap.rowconfigure(0, weight=1)
		wrap.columnconfigure(0, weight=1)

		canvas = tk.Canvas(wrap, highlightthickness=0, borderwidth=0)
		vsb = ttk.Scrollbar(wrap, orient="vertical", command=canvas.yview)
		canvas.configure(yscrollcommand=vsb.set)
		canvas.grid(row=0, column=0, sticky="nsew")
		vsb.grid(row=0, column=1, sticky="ns")

		inner = ttk.Frame(canvas)
		window = canvas.create_window((0, 0), window=inner, anchor="nw")
		inner.columnconfigure(0, weight=1)
		inner.bind("<Configure>", lambda e: canvas.configure(
			scrollregion=canvas.bbox("all")))
		canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))
		return wrap, inner, canvas

	def _bind_wheel(self, widget, canvas):
		"""Wheel-scroll `canvas` from anywhere over `widget`, leaving alone
		the widgets that scroll themselves (text boxes, thumbnail grids)."""

		def on_wheel(event):
			first, last = canvas.yview()
			if first <= 0.0 and last >= 1.0:
				return
			canvas.yview_scroll(-1 * (event.delta // 120), "units")

		def walk(w):
			# A thumbnail grid is skipped whole, tiles and all - otherwise the
			# wheel over the gaps between its tiles would scroll the panel.
			# Plain canvases are not skipped: the preview is one, and the
			# wheel over it should scroll its side of the window.
			if isinstance(w, (tk.Text, tk.Listbox, ThumbnailGrid)):
				return
			w.bind("<MouseWheel>", on_wheel)
			for child in w.winfo_children():
				walk(child)

		walk(widget)
		canvas.bind("<MouseWheel>", on_wheel)

	def _build_topbar(self):
		box = ttk.LabelFrame(self, text="What should it do?", padding=8)
		box.columnconfigure(1, weight=1)
		box.columnconfigure(3, weight=1)

		ttk.Label(box, text="Task").grid(row=0, column=0, sticky="w", padx=(0, 6))
		self.cb_mode = ttk.Combobox(
			box, state="readonly", width=24,
			values=[MODE_LABELS[m] for m in MODE_ORDER],
		)
		self.cb_mode.current(0)
		self.cb_mode.grid(row=0, column=1, columnspan=3, sticky="ew", pady=2)
		self.cb_mode.bind("<<ComboboxSelected>>", self._on_mode_change)

		ttk.Label(box, text="Model").grid(row=1, column=0, sticky="w", padx=(0, 6))
		self.cb_model = ttk.Combobox(box, state="readonly", width=24,
		                             textvariable=self.v_model)
		self.cb_model.grid(row=1, column=1, columnspan=3, sticky="ew", pady=2)
		self.cb_model.bind("<<ComboboxSelected>>", self._on_model_change)

		self.lbl_base = ttk.Label(box, text="Base model")
		self.cb_base_model = ttk.Combobox(box, state="readonly", width=24,
		                                  textvariable=self.v_base_model)
		self.lbl_base.grid(row=2, column=0, sticky="w", padx=(0, 6))
		self.cb_base_model.grid(row=2, column=1, columnspan=3, sticky="ew", pady=2)

		self.lbl_quant = ttk.Label(box, text="Quant")
		self.cb_quant = ttk.Combobox(box, state="readonly", width=18,
		                             textvariable=self.v_quant)
		self.lbl_quant.grid(row=3, column=0, sticky="w", padx=(0, 6))
		self.cb_quant.grid(row=3, column=1, sticky="ew", pady=2)

		ttk.Label(box, text="Memory").grid(row=3, column=2, sticky="e", padx=(8, 6))
		self.cb_memory = ttk.Combobox(
			box, state="readonly", width=18, textvariable=self.v_memory,
			values=[label for _, label in MEMORY_MODES],
		)
		self.cb_memory.grid(row=3, column=3, sticky="ew", pady=2)

		note = ttk.Label(box, textvariable=self.v_modelnote, foreground="#555")
		note.grid(row=4, column=0, columnspan=4, sticky="w", pady=(4, 0))
		ttk.Label(box, textvariable=self.v_device, foreground="#555").grid(
			row=5, column=0, columnspan=4, sticky="w")
		return box

	def _build_prompts(self, parent):
		box = ttk.LabelFrame(parent, text="Prompt", padding=8)
		box.columnconfigure(0, weight=1)   # labels and text boxes stretch
		# Column 1 holds the Improve buttons at their natural width.

		# -- the language model that rewrites prompts -------------------------
		row = ttk.Frame(box)
		row.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
		row.columnconfigure(1, weight=1)
		ttk.Label(row, text="Text model").grid(row=0, column=0, sticky="w", padx=(0, 6))
		self.cb_textmodel = ttk.Combobox(
			row, state="readonly", width=20, textvariable=self.v_textmodel,
			values=[m.label for m in textgen.MODELS],
		)
		self.cb_textmodel.grid(row=0, column=1, sticky="ew")
		self.cb_textmodel.bind("<<ComboboxSelected>>", self._on_textmodel_change)
		self.btn_textsetup = ttk.Button(row, text="Check", width=10,
		                                command=self._textgen_button)
		self.btn_textsetup.grid(row=0, column=2, padx=(6, 0))
		self.btn_textunload = ttk.Button(row, text="Unload", width=8,
		                                 command=self._textgen_unload,
		                                 state="disabled")
		self.btn_textunload.grid(row=0, column=3, padx=(4, 0))
		ttk.Label(row, textvariable=self.v_textstatus, foreground="#555",
		          wraplength=self.px(430), justify="left").grid(
			row=1, column=0, columnspan=4, sticky="w", pady=(2, 0))

		# -- base image prompt, chain mode only -------------------------------
		self.lbl_base_prompt = ttk.Label(box, text="Base image prompt")
		self.btn_improve_base = ttk.Button(
			box, text="Improve", width=9, command=lambda: self._improve("base"))
		self.txt_base_prompt = tk.Text(box, height=2, wrap="word", undo=True)
		self.lbl_base_prompt.grid(row=1, column=0, sticky="w")
		self.btn_improve_base.grid(row=1, column=1, sticky="e", padx=(6, 0))
		self.txt_base_prompt.grid(row=2, column=0, columnspan=2, sticky="ew",
		                          pady=(0, 6))

		# -- the main prompt ---------------------------------------------------
		self.lbl_prompt_hint = ttk.Label(box, textvariable=self.v_prompt_hint)
		self.lbl_prompt_hint.grid(row=3, column=0, sticky="w")
		self.btn_improve_prompt = ttk.Button(
			box, text="Improve", width=9, command=lambda: self._improve("prompt"))
		self.btn_improve_prompt.grid(row=3, column=1, sticky="e", padx=(6, 0))
		self.txt_prompt = tk.Text(box, height=3, wrap="word", undo=True)
		self.txt_prompt.grid(row=4, column=0, columnspan=2, sticky="ew")

		# -- the negative prompt -----------------------------------------------
		self.lbl_neg = ttk.Label(box, text="Negative prompt (what to avoid)")
		self.lbl_neg.grid(row=5, column=0, sticky="w", pady=(6, 0))
		self.btn_improve_neg = ttk.Button(
			box, text="Improve", width=9, command=lambda: self._improve("negative"))
		self.btn_improve_neg.grid(row=5, column=1, sticky="e", padx=(6, 0), pady=(6, 0))
		self.txt_neg = tk.Text(box, height=2, wrap="word", undo=True)
		self.txt_neg.grid(row=6, column=0, columnspan=2, sticky="ew")
		return box

	def _build_files(self, parent):
		box = ttk.LabelFrame(parent, text="Files", padding=8)
		box.columnconfigure(0, weight=1)
		self.files_box = box

		self.lbl_inputs = ttk.Label(box, text="Input images")
		self.lbl_inputs.grid(row=0, column=0, columnspan=2, sticky="w")

		btns = ttk.Frame(box)
		btns.grid(row=1, column=0, sticky="ew", pady=(2, 4))
		self.btn_addfiles = ttk.Button(btns, text="Add files...", command=self._add_files)
		self.btn_addfolder = ttk.Button(btns, text="Add folder...", command=self._add_folder)
		self.btn_remove = ttk.Button(btns, text="Remove", command=self._remove_selected)
		self.btn_clear = ttk.Button(btns, text="Clear", command=self._clear_files)
		for i, b in enumerate((self.btn_addfiles, self.btn_addfolder,
		                       self.btn_remove, self.btn_clear)):
			btns.columnconfigure(i, weight=1)
			b.grid(row=0, column=i, padx=(0, 4), sticky="ew")

		# The file list is shown as a thumbnail grid; an invisible Listbox
		# underneath is still the one source of truth for content and
		# selection (self._files(), .curselection()) - the grid's selection
		# helpers write through to it rather than keeping a second copy.
		self.lst_files = tk.Listbox(box, exportselection=False)
		self.lst_files.bind("<<ListboxSelect>>", self._on_file_select)

		self.file_grid = ThumbnailGrid(
			box, scale=self.scale, height=140,
			on_click=self._select_exclusive,
			on_ctrl_click=self._select_toggle,
			on_shift_click=self._select_range,
			on_double=self._open_grid_tile,
		)

		self.file_grid.grid(row=2, column=0, columnspan=2, sticky="nsew")

		hint = "Drop images or folders here" if HAS_DND else ""
		ttk.Label(box, textvariable=self.v_filecount, foreground="#555").grid(
			row=3, column=0, sticky="w", pady=(2, 0))
		if hint:
			ttk.Label(box, text=hint, foreground="#888").grid(
				row=3, column=1, sticky="e", pady=(2, 0))

		ttk.Label(box, text="Save results to").grid(row=4, column=0, sticky="w",
		                                            pady=(6, 0))
		outrow = ttk.Frame(box)
		outrow.grid(row=5, column=0, columnspan=2, sticky="ew")
		outrow.columnconfigure(0, weight=1)
		ttk.Entry(outrow, textvariable=self.v_out, width=12).grid(
			row=0, column=0, sticky="ew")
		ttk.Button(outrow, text="Browse...", command=self._pick_out).grid(
			row=0, column=1, padx=(4, 0))
		ttk.Button(outrow, text="Open", command=self._open_out).grid(
			row=0, column=2, padx=(4, 0))

		self._setup_dnd()
		return box

	def _build_settings(self, parent):
		box = ttk.LabelFrame(parent, text="Settings", padding=8)
		for c in (1, 3):
			box.columnconfigure(c, weight=1)

		def field(row, col, label, var, width=10):
			lbl = ttk.Label(box, text=label)
			lbl.grid(row=row, column=col, sticky="e", padx=(0, 6), pady=2)
			ent = ttk.Entry(box, textvariable=var, width=width)
			ent.grid(row=row, column=col + 1, sticky="ew", pady=2)
			return lbl, ent

		self.lbl_w, self.ent_w = field(0, 0, "Width", self.v_width)
		self.lbl_h, self.ent_h = field(0, 2, "Height", self.v_height)
		self.lbl_steps, self.ent_steps = field(1, 0, "Steps", self.v_steps)
		self.lbl_guid, self.ent_guid = field(1, 2, "Guidance", self.v_guidance)
		self.lbl_str, self.ent_str = field(2, 0, "Strength", self.v_strength)
		self.lbl_count, self.ent_count = field(2, 2, "How many", self.v_count)
		self.lbl_smin, self.ent_smin = field(3, 0, "Sweep from", self.v_sweep_min)
		self.lbl_smax, self.ent_smax = field(3, 2, "Sweep to", self.v_sweep_max)

		ttk.Label(box, text="Seed").grid(row=4, column=0, sticky="e", padx=(0, 6), pady=2)
		seedrow = ttk.Frame(box)
		seedrow.grid(row=4, column=1, columnspan=3, sticky="ew", pady=2)
		seedrow.columnconfigure(0, weight=1)
		self.ent_seed = ttk.Entry(seedrow, textvariable=self.v_seed)
		self.ent_seed.grid(row=0, column=0, sticky="ew")
		ttk.Button(seedrow, text="Roll", width=6, command=self._roll_seed).grid(
			row=0, column=1, padx=(4, 8))
		ttk.Checkbutton(seedrow, text="Random each image",
		                variable=self.v_random_seed,
		                command=self._sync_seed_state).grid(row=0, column=2)

		ttk.Checkbutton(box, text="Apply the model's recommended settings when I "
		                         "switch models",
		                variable=self.v_autodefaults).grid(
			row=5, column=0, columnspan=4, sticky="w", pady=(4, 0))
		self.hint_dim = ttk.Label(box, text="", foreground="#555")
		self.hint_dim.grid(row=6, column=0, columnspan=4, sticky="w")
		return box

	def _build_actions(self, parent):
		box = ttk.Frame(parent)
		box.columnconfigure(0, weight=1)

		row = ttk.Frame(box)
		row.grid(row=0, column=0, sticky="ew")
		row.columnconfigure(2, weight=1)
		self.btn_go = ttk.Button(row, text="Generate", command=self._start)
		self.btn_go.grid(row=0, column=0, padx=(0, 6))
		self.btn_cancel = ttk.Button(row, text="Cancel", command=self._cancel,
		                             state="disabled")
		self.btn_cancel.grid(row=0, column=1)
		self.btn_unload = ttk.Button(row, text="Unload model", command=self._unload)
		self.btn_unload.grid(row=0, column=3, sticky="e")

		self.pb_items = ttk.Progressbar(box, mode="determinate")
		self.pb_items.grid(row=1, column=0, sticky="ew", pady=(8, 2))
		self.pb_steps = ttk.Progressbar(box, mode="determinate")
		self.pb_steps.grid(row=2, column=0, sticky="ew")
		ttk.Label(box, textvariable=self.v_status).grid(row=3, column=0, sticky="w",
		                                                pady=(4, 0))
		return box

	def _build_settings_tab(self, parent):
		"""The Settings tab: things set once rather than per run. `parent` is
		the inner frame of a _scroll_page, so the tab scrolls when it overflows."""
		page = ttk.Frame(parent, padding=(0, 8, 0, 0))
		page.grid(row=0, column=0, sticky="nsew")
		page.columnconfigure(0, weight=1)
		page.rowconfigure(1, weight=1)
		def wrapped(master, **kw):
			# Wrap at the label's real width rather than a fixed pixel count, so
			# the text reflows with the column instead of running off its edge.
			lbl = ttk.Label(master, justify="left", **kw)
			lbl.bind("<Configure>", lambda e: lbl.configure(wraplength=max(e.width, 50)))
			return lbl

		# -- HuggingFace access token -----------------------------------------
		box = ttk.LabelFrame(page, text="HuggingFace access token", padding=8)
		box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
		box.columnconfigure(0, weight=1)

		wrapped(box, textvariable=self.v_hf_status).grid(
			row=0, column=0, columnspan=3, sticky="ew")

		ttk.Label(box, text="New token").grid(row=1, column=0, columnspan=3,
		                                      sticky="w", pady=(8, 2))
		self.ent_hf_token = ttk.Entry(box, textvariable=self.v_hf_token, show="•")
		self.ent_hf_token.grid(row=2, column=0, sticky="ew")
		self.ent_hf_token.bind("<Return>", lambda e: self._hf_save())
		ttk.Checkbutton(box, text="Show", variable=self.v_hf_show,
		                command=self._hf_toggle_show).grid(row=2, column=1, padx=(6, 0))

		btns = ttk.Frame(box)
		btns.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(6, 0))
		self.btn_hf_save = ttk.Button(btns, text="Save", command=self._hf_save)
		self.btn_hf_remove = ttk.Button(btns, text="Remove saved token",
		                                command=self._hf_remove)
		self.btn_hf_get = ttk.Button(btns, text="Get a token...",
		                             command=self._hf_open_tokens_page)
		for i, b in enumerate((self.btn_hf_save, self.btn_hf_remove, self.btn_hf_get)):
			btns.columnconfigure(i, weight=1)
			b.grid(row=0, column=i, padx=(0, 4), sticky="ew")

		wrapped(box, foreground="#555", text=(
			"Needed for gated models (FLUX.1 dev, SRPO, Krea 2) - a \"Read\" token "
			"is enough. It is checked with HuggingFace before saving, then stored "
			"in HuggingFace's own token file (%USERPROFILE%\\.cache\\huggingface\\token), "
			"shared with other HuggingFace tools - never in this app's settings."
		)).grid(row=4, column=0, columnspan=3, sticky="ew", pady=(8, 0))

		# -- which model repos the token can download --------------------------
		acc = ttk.LabelFrame(page, text="Model access", padding=8)
		acc.grid(row=1, column=0, sticky="nsew")
		acc.columnconfigure(0, weight=1)
		acc.rowconfigure(2, weight=1)
		wrapped(acc, foreground="#555", text=(
			"Checks every repo the image models download from. A gated repo needs "
			"its licence accepted on huggingface.co with the same account."
		)).grid(row=0, column=0, columnspan=2, sticky="ew")
		self.btn_hf_check = ttk.Button(acc, text="Check access",
		                               command=self._hf_check_access)
		self.btn_hf_check.grid(row=1, column=0, sticky="w", pady=(6, 4))
		self.txt_hf_access = tk.Text(acc, height=10, wrap="word", state="disabled",
		                             font="TkDefaultFont")
		self.txt_hf_access.grid(row=2, column=0, sticky="nsew")
		self.txt_hf_access.tag_configure("bad", foreground="#b00020")
		scroll = ttk.Scrollbar(acc, orient="vertical", command=self.txt_hf_access.yview)
		self.txt_hf_access.configure(yscrollcommand=scroll.set)
		scroll.grid(row=2, column=1, sticky="ns")
		return page

	def _build_preview(self, parent):
		box = ttk.LabelFrame(parent, text="Preview", padding=6)
		box.rowconfigure(0, weight=1)
		box.columnconfigure(0, weight=1)
		self.canvas = tk.Canvas(box, background="#1e1e1e", highlightthickness=0,
		                        width=420, height=280)
		self.canvas.grid(row=0, column=0, sticky="nsew")
		self.canvas.bind("<Configure>", lambda e: self._render_preview())
		self.canvas.bind("<Button-1>", lambda e: self._open_preview())
		self.v_preview_name = tk.StringVar(value="Nothing generated yet.")
		ttk.Label(box, textvariable=self.v_preview_name, foreground="#555").grid(
			row=1, column=0, sticky="w", pady=(4, 0))

		# -- export the previewed picture in another format --------------------
		exp = ttk.Frame(box)
		exp.grid(row=2, column=0, sticky="ew", pady=(6, 0))
		exp.columnconfigure(1, weight=1)
		ttk.Label(exp, text="Export as").grid(row=0, column=0, sticky="w", padx=(0, 6))
		self.cb_export = ttk.Combobox(
			exp, state="readonly", width=24, textvariable=self.v_export,
			values=[f.label for f in formats.EXPORT_FORMATS],
		)
		self.cb_export.grid(row=0, column=1, sticky="ew")
		self.cb_export.bind("<<ComboboxSelected>>", self._on_export_format_change)
		self.btn_export = ttk.Button(exp, text="Export...", width=10,
		                             command=self._export_preview)
		self.btn_export.grid(row=0, column=2, padx=(6, 0))
		self.btn_replace = ttk.Button(exp, text="Replace original", width=15,
		                              command=self._replace_original)
		self.btn_replace.grid(row=0, column=3, padx=(4, 0))
		ttk.Label(exp, textvariable=self.v_export_note, foreground="#555",
		          wraplength=self.px(420), justify="left").grid(
			row=1, column=0, columnspan=4, sticky="w", pady=(2, 0))
		return box

	def _build_results(self, parent):
		"""Every picture generated so far, newest first, as a clickable grid -
		plus the all-at-once versions of the preview's Export and Replace
		original, using the format chosen under the preview."""
		box = ttk.LabelFrame(parent, text="Results", padding=6)
		box.columnconfigure(0, weight=1)
		box.rowconfigure(1, weight=1)

		bar = ttk.Frame(box)
		bar.grid(row=0, column=0, sticky="ew")
		bar.columnconfigure(0, weight=1)
		self.btn_export_all = ttk.Button(bar, text="Export all...",
		                                 command=self._export_all)
		self.btn_replace_all = ttk.Button(bar, text="Replace all",
		                                  command=self._replace_all)
		self.btn_results_remove = ttk.Button(bar, text="Remove",
		                                     command=self._results_remove)
		self.btn_results_clear = ttk.Button(bar, text="Clear",
		                                    command=self._results_clear)
		for i, b in enumerate((self.btn_export_all, self.btn_replace_all,
		                       self.btn_results_remove, self.btn_results_clear), 1):
			b.grid(row=0, column=i, padx=(4, 0))

		self.results_grid = ThumbnailGrid(
			box, scale=self.scale, height=130, thumb=72,
			empty_text="Generated images will appear here.",
			on_click=self._result_click,
			on_ctrl_click=self._result_ctrl_click,
			on_shift_click=self._result_shift_click,
			on_double=self._result_open,
		)
		self.results_grid.grid(row=1, column=0, sticky="nsew", pady=(4, 0))
		# The count / progress line lives in the panel title rather than beside
		# the buttons, where it got cut off at any sensible window width.
		self.results_box = box
		self.v_results_status.trace_add("write", self._sync_results_title)
		return box

	def _build_describe(self, parent):
		"""Image -> text: describe whatever is in the preview, with a local
		vision model, so the description can go straight back into a prompt."""
		box = ttk.LabelFrame(parent, text="Describe image", padding=6)
		box.columnconfigure(1, weight=1)

		ttk.Label(box, text="Vision model").grid(row=0, column=0, sticky="w",
		                                         padx=(0, 6))
		self.cb_vision = ttk.Combobox(
			box, state="readonly", width=18, textvariable=self.v_vision_model,
			values=[m.label for m in vision.VISION_MODELS],
		)
		self.cb_vision.grid(row=0, column=1, sticky="ew")
		self.cb_vision.bind("<<ComboboxSelected>>", self._on_vision_model_change)
		self.btn_vision_setup = ttk.Button(box, text="Check", width=10,
		                                   command=self._vision_button)
		self.btn_vision_setup.grid(row=0, column=2, padx=(6, 0))
		self.btn_describe = ttk.Button(box, text="Describe", width=10,
		                               command=self._describe_preview)
		self.btn_describe.grid(row=0, column=3, padx=(4, 0))

		ttk.Label(box, text="Style").grid(row=1, column=0, sticky="w",
		                                  padx=(0, 6), pady=(4, 0))
		self.cb_vision_style = ttk.Combobox(
			box, state="readonly", width=18, textvariable=self.v_vision_style,
			values=[s.label for s in vision.STYLES],
		)
		self.cb_vision_style.grid(row=1, column=1, sticky="ew", pady=(4, 0))

		ttk.Label(box, textvariable=self.v_vision_status, foreground="#555",
		          wraplength=self.px(420), justify="left").grid(
			row=2, column=0, columnspan=4, sticky="w", pady=(2, 0))

		self.txt_describe = tk.Text(box, height=4, wrap="word", undo=True)
		desc_scroll = ttk.Scrollbar(box, orient="vertical",
		                            command=self.txt_describe.yview)
		self.txt_describe.configure(yscrollcommand=desc_scroll.set)
		self.txt_describe.grid(row=3, column=0, columnspan=4, sticky="ew",
		                       pady=(4, 0))
		desc_scroll.grid(row=3, column=4, sticky="ns", pady=(4, 0))

		actions = ttk.Frame(box)
		actions.grid(row=4, column=0, columnspan=5, sticky="ew", pady=(4, 0))
		self.btn_desc_prompt = ttk.Button(actions, text="Use as prompt",
		                                  command=self._describe_to_prompt)
		self.btn_desc_copy = ttk.Button(actions, text="Copy",
		                                command=self._describe_copy)
		self.btn_desc_clear = ttk.Button(actions, text="Clear",
		                                 command=self._describe_clear)
		for i, b in enumerate((self.btn_desc_prompt, self.btn_desc_copy,
		                       self.btn_desc_clear)):
			actions.columnconfigure(i, weight=1)
			b.grid(row=0, column=i, padx=(0, 4), sticky="ew")
		return box

	def _build_log(self, parent):
		box = ttk.LabelFrame(parent, text="Log", padding=6)
		box.rowconfigure(0, weight=1)
		box.columnconfigure(0, weight=1)
		self.txt_log = tk.Text(box, height=10, wrap="word", state="disabled",
		                       background="#111", foreground="#ddd",
		                       insertbackground="#ddd")
		scroll = ttk.Scrollbar(box, orient="vertical", command=self.txt_log.yview)
		self.txt_log.configure(yscrollcommand=scroll.set)
		self.txt_log.grid(row=0, column=0, sticky="nsew")
		scroll.grid(row=0, column=1, sticky="ns")
		return box

	# -- mode / model plumbing -------------------------------------------
	def mode(self):
		return MODE_ORDER[self.cb_mode.current()]

	def _model_label(self, spec):
		mark = "" if registry.is_cached(spec) else "  (will download)"
		return spec.label + mark

	def _on_mode_change(self, _event=None):
		mode = self.mode()
		task = MODE_TASK[mode]

		specs = registry.models_for(task)
		self._model_keys = [s.key for s in specs]
		labels = [self._model_label(s) for s in specs]
		self.cb_model["values"] = labels
		if self.v_model.get() not in labels:
			self.cb_model.current(0)

		# The chain mode needs a separate text-to-image model for stage one.
		base_specs = registry.models_for(TXT2IMG)
		self._base_model_keys = [s.key for s in base_specs]
		self.cb_base_model["values"] = [self._model_label(s) for s in base_specs]
		if self.v_base_model.get() not in self.cb_base_model["values"]:
			self.cb_base_model.current(0)

		is_chain = mode == MODE_CHAIN
		is_sweep = mode == MODE_SWEEP
		needs_files = mode in (MODE_IMG2IMG, MODE_BATCH)
		text_driven = mode in (MODE_TXT2IMG, MODE_SWEEP, MODE_CHAIN)

		self._show(
			self.lbl_base, self.cb_base_model, show=is_chain,
			label_grid=dict(row=2, column=0, sticky="w", padx=(0, 6)),
			widget_grid=dict(row=2, column=1, columnspan=3, sticky="ew", pady=2),
		)
		self._show(
			self.lbl_base_prompt, self.txt_base_prompt, show=is_chain,
			label_grid=dict(row=1, column=0, sticky="w"),
			widget_grid=dict(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 6)),
		)
		if is_chain:
			self.btn_improve_base.grid(row=1, column=1, sticky="e", padx=(6, 0))
		else:
			self.btn_improve_base.grid_remove()

		# Kept short: the Improve button sits on the same row and a longer
		# label gets clipped rather than wrapped.
		self.v_prompt_hint.set(
			"Edit instruction (what to change)"
			if mode in (MODE_IMG2IMG, MODE_BATCH, MODE_CHAIN)
			else "Prompt (what to draw)"
		)

		self._files_enabled = needs_files
		for w in (self.btn_addfiles, self.btn_addfolder, self.btn_remove,
		          self.btn_clear, self.lst_files):
			w.configure(state="normal" if needs_files else "disabled")
		self.lbl_inputs.configure(
			text="Input image (only the first in the list is used)"
			if mode == MODE_IMG2IMG else "Input images"
		)
		if not needs_files:
			self.lbl_inputs.configure(text="Input images (not used by this task)")

		self._enable(self.lbl_smin, self.ent_smin, self.lbl_smax, self.ent_smax,
		             enabled=is_sweep)
		self._enable(self.lbl_count, self.ent_count, enabled=not is_sweep)
		self._enable(self.lbl_steps, self.ent_steps, enabled=not is_sweep)
		self.hint_dim.configure(
			text="Width/Height: 0 = keep each input image at its own size."
			if needs_files else ""
		)
		self._on_model_change()
		self._sync_seed_state()
		self._refresh_filecount()

	def _on_model_change(self, event=None):
		spec = self.selected_spec()
		if spec is None:
			return
		has_quant = bool(spec.quants)
		self._show(
			self.lbl_quant, self.cb_quant, show=has_quant,
			label_grid=dict(row=3, column=0, sticky="w", padx=(0, 6)),
			widget_grid=dict(row=3, column=1, sticky="ew", pady=2),
		)
		if has_quant:
			self.cb_quant["values"] = [label for _, label in spec.quants]
			if self.v_quant.get() not in self.cb_quant["values"]:
				keys = [k for k, _ in spec.quants]
				self.cb_quant.current(keys.index(registry.quant_for(spec, None)))

		# FLUX Kontext and InstructPix2Pix ignore strength; SD img2img depends on it.
		uses_strength = (spec.kind not in backend.NO_STRENGTH_KINDS
		                 and self.mode() in (MODE_IMG2IMG, MODE_BATCH, MODE_CHAIN))
		self._enable(self.lbl_str, self.ent_str, enabled=uses_strength)

		note = spec.vram_note
		if not registry.is_cached(spec):
			note += "  -  not downloaded yet"
		self.v_modelnote.set(note)

		if event is not None and self.v_autodefaults.get():
			self._apply_defaults(spec)

	def _apply_defaults(self, spec):
		d = spec.defaults
		if "steps" in d:
			self.v_steps.set(str(d["steps"]))
		if "guidance" in d:
			self.v_guidance.set(str(d["guidance"]))
		if d.get("width"):
			self.v_width.set(str(d["width"]))
		if d.get("height"):
			self.v_height.set(str(d["height"]))
		self._log(f"Applied recommended settings for {spec.label}.")

	def selected_spec(self):
		idx = self.cb_model.current()
		if idx < 0 or idx >= len(self._model_keys):
			return None
		return registry.BY_KEY[self._model_keys[idx]]

	def selected_base_spec(self):
		idx = self.cb_base_model.current()
		if idx < 0 or idx >= len(self._base_model_keys):
			return registry.BY_KEY[self._base_model_keys[0]]
		return registry.BY_KEY[self._base_model_keys[idx]]

	@staticmethod
	def _show(label, widget, show, label_grid, widget_grid):
		"""Grid or un-grid a label/widget pair so hidden rows collapse away."""
		if show:
			label.grid(**label_grid)
			widget.grid(**widget_grid)
		else:
			label.grid_remove()
			widget.grid_remove()

	@staticmethod
	def _enable(*widgets, enabled):
		state = "normal" if enabled else "disabled"
		for w in widgets:
			try:
				w.configure(state=state)
			except tk.TclError:
				pass

	def _sync_seed_state(self):
		self._enable(self.ent_seed, enabled=not self.v_random_seed.get())

	def _roll_seed(self):
		self.v_seed.set(str(random.randint(0, 2**31 - 1)))
		self.v_random_seed.set(False)
		self._sync_seed_state()

	# -- file list --------------------------------------------------------
	def _files(self):
		return list(self.lst_files.get(0, "end"))

	@contextmanager
	def _writable_list(self):
		"""Tk silently drops insert/delete on a disabled Listbox, and the list is
		disabled in the text-only tasks - so re-enable it around every edit."""
		previous = str(self.lst_files.cget("state"))
		self.lst_files.configure(state="normal")
		try:
			yield
		finally:
			self.lst_files.configure(state=previous)

	def _add_paths(self, paths):
		existing = {p.lower() for p in self._files()}
		added = False
		with self._writable_list():
			for p in paths:
				if p and p.lower() not in existing:
					self.lst_files.insert("end", p)
					existing.add(p.lower())
					added = True
		self._refresh_filecount()
		if added:
			self._refresh_grid_if_visible()

	def _add_files(self):
		patterns = " ".join("*" + e for e in formats.READABLE_EXTS)
		paths = filedialog.askopenfilenames(
			title="Choose input images",
			filetypes=[("Images and textures", patterns),
			           ("DDS textures", "*.dds"),
			           ("All files", "*.*")],
			initialdir=str(app_dir()),
		)
		self._add_paths(list(paths))

	def _add_folder(self):
		folder = filedialog.askdirectory(title="Choose a folder of images",
		                                 initialdir=str(app_dir()))
		if folder:
			self._add_paths([folder])

	def _remove_selected(self):
		with self._writable_list():
			for i in reversed(self.lst_files.curselection()):
				self.lst_files.delete(i)
		self._refresh_filecount()
		self._refresh_grid_if_visible()

	def _clear_files(self):
		with self._writable_list():
			self.lst_files.delete(0, "end")
		self._refresh_filecount()
		self._refresh_grid_if_visible()

	def _refresh_filecount(self):
		if MODE_TASK.get(self.mode()) != IMG2IMG or self.mode() == MODE_CHAIN:
			self.v_filecount.set("This task starts from text, so no inputs are needed.")
			return
		entries = self._files()
		if not entries:
			self.v_filecount.set("No input files selected.")
			return
		try:
			n = len(backend.collect_images(entries))
		except Exception:
			n = 0
		if self.mode() == MODE_IMG2IMG:
			self.v_filecount.set(f"{n} image(s) listed - the first one will be used.")
		else:
			self.v_filecount.set(f"{n} image(s) will be processed.")

	def _on_file_select(self, _event=None):
		self._update_grid_selection_highlight()
		sel = self.lst_files.curselection()
		if not sel:
			return
		path = Path(self.lst_files.get(sel[0]))
		if path.is_file():
			self._set_preview(str(path), label=f"Input: {path.name}")

	def _refresh_grid_if_visible(self):
		self._rebuild_grid()

	# -- selection, shared between both views ------------------------------
	# The listbox is the one source of truth for "what's selected"
	# (.curselection()), whether or not it is the view actually on screen -
	# grid tiles write through to it instead of keeping their own copy, so
	# Remove/Clear/preview-on-select all keep working unchanged either way.
	def _select_exclusive(self, idx):
		if not self._files_enabled:
			return
		self.lst_files.selection_clear(0, "end")
		self.lst_files.selection_set(idx)
		self.lst_files.activate(idx)
		self._select_anchor = idx
		self._on_file_select()

	def _select_toggle(self, idx):
		if not self._files_enabled:
			return
		if idx in self.lst_files.curselection():
			self.lst_files.selection_clear(idx)
		else:
			self.lst_files.selection_set(idx)
		self._select_anchor = idx
		self._on_file_select()

	def _select_range(self, idx):
		if not self._files_enabled:
			return
		anchor = self._select_anchor if self._select_anchor is not None else idx
		lo, hi = sorted((anchor, idx))
		self.lst_files.selection_clear(0, "end")
		self.lst_files.selection_set(lo, hi)
		self._on_file_select()

	def _update_grid_selection_highlight(self):
		self.file_grid.set_selected(self.lst_files.curselection())

	# -- grid view ---------------------------------------------------------
	def _open_grid_tile(self, idx):
		path = Path(self.lst_files.get(idx))
		if path.is_file():
			os.startfile(str(path))

	def _rebuild_grid(self):
		self.file_grid.set_items(self._files())
		self._update_grid_selection_highlight()

	# -- drag and drop -------------------------------------------------------
	def _setup_dnd(self):
		if not HAS_DND:
			return
		try:
			for widget in (self.files_box, self.lst_files, self.file_grid.canvas):
				widget.drop_target_register(DND_FILES)
				widget.dnd_bind("<<DropEnter>>", self._on_drop_enter)
				widget.dnd_bind("<<DropLeave>>", self._on_drop_leave)
				widget.dnd_bind("<<Drop>>", self._on_files_dropped)
		except tk.TclError:
			# The tkdnd Tcl extension is only loaded into a TkinterDnD.Tk()
			# root (main() creates one when tkinterdnd2 is installed). A
			# plain tk.Tk() root - e.g. in a script driving App() directly
			# for testing - doesn't have it, so drag-and-drop just quietly
			# isn't available rather than the whole window failing to build.
			pass

	def _on_drop_enter(self, event):
		if self._files_enabled:
			self.lst_files.configure(highlightbackground="#3a7bd5",
			                         highlightcolor="#3a7bd5", highlightthickness=2)
			self.file_grid.canvas.configure(highlightbackground="#3a7bd5",
			                           highlightcolor="#3a7bd5", highlightthickness=2)
		return event.action

	def _on_drop_leave(self, event=None):
		self.lst_files.configure(highlightbackground="SystemButtonFace",
		                         highlightthickness=1)
		self.file_grid.canvas.configure(highlightbackground="#1e1e1e", highlightthickness=1)

	def _on_files_dropped(self, event):
		# <<Drop>> handlers are required to return a drop action (tkdnd's
		# contract - see dnd_bind's docstring), or Explorer's drag cursor
		# reports the drop as refused even though this code did handle it.
		self._on_drop_leave()
		if not self._files_enabled:
			self._log("Dropped files ignored - this task does not use input images.")
			return COPY
		try:
			raw = self.master.tk.splitlist(event.data)
		except tk.TclError:
			raw = [event.data]

		accepted, rejected = [], []
		for p in raw:
			path = Path(p)
			if path.is_dir() or path.suffix.lower() in formats.READABLE_EXTS:
				accepted.append(str(path))
			else:
				rejected.append(path.name)

		if accepted:
			self._add_paths(accepted)
			self._log(f"Added {len(accepted)} item(s) by drag and drop.")
		if rejected:
			shown = ", ".join(rejected[:5]) + (" ..." if len(rejected) > 5 else "")
			self._log(f"Ignored {len(rejected)} dropped file(s) with an "
			         f"unsupported extension: {shown}")
		return COPY

	def _pick_out(self):
		folder = filedialog.askdirectory(title="Choose an output folder",
		                                 initialdir=self.v_out.get() or str(app_dir()))
		if folder:
			self.v_out.set(folder)

	def _open_out(self):
		folder = Path(self.v_out.get())
		folder.mkdir(parents=True, exist_ok=True)
		os.startfile(str(folder))

	def _open_preview(self):
		if self.preview_source and Path(self.preview_source).is_file():
			os.startfile(self.preview_source)

	# -- exporting the previewed picture ----------------------------------
	def selected_export_format(self):
		label = self.v_export.get()
		for fmt in formats.EXPORT_FORMATS:
			if fmt.label == label:
				return fmt
		return formats.BY_KEY[formats.DEFAULT_EXPORT]

	def _on_export_format_change(self, _event=None):
		self.v_export_note.set(self.selected_export_format().note)

	def _export_preview(self):
		"""Re-encode whatever is in the preview into the chosen format.

		The preview may be a generated result or an input file the user
		clicked, so this doubles as a plain converter - point it at a .dds and
		export a .png, or the other way round.
		"""
		if not self.preview_source or not Path(self.preview_source).is_file():
			messagebox.showinfo(
				APP_NAME,
				"Nothing to export yet. Generate an image, or click one in the "
				"file list to preview it first.")
			return

		fmt = self.selected_export_format()
		source = Path(self.preview_source)
		target = filedialog.asksaveasfilename(
			title=f"Export as {fmt.label}",
			defaultextension=fmt.extension,
			initialfile=source.stem + fmt.extension,
			initialdir=self.v_out.get() or str(source.parent),
			filetypes=[(fmt.label, "*" + fmt.extension), ("All files", "*.*")],
		)
		if not target:
			return

		try:
			written = formats.convert_file(source, target, fmt)
		except Exception as exc:
			self._log(f"Export failed: {type(exc).__name__}: {exc}")
			messagebox.showerror(APP_NAME, f"Could not export: {exc}")
			return

		size_kb = written.stat().st_size / 1024
		self._log(f"Exported {source.name} as {fmt.label} -> {written} "
		          f"({size_kb:.0f} KB)")
		self.v_export_note.set(f"Exported to {written.name} ({size_kb:.0f} KB).")

	def _replace_original(self):
		"""Write the current preview back over the file it came from - same
		folder, same base name, no save dialog. For an edited result that
		means the input it was generated from; for a plain input preview
		(nothing generated yet) it means itself, so this doubles as an
		in-place re-encode (e.g. recompress a DXT1 texture as DXT5).

		The export format is still whatever's chosen in the dropdown - this
		does not try to detect the original's own encoding, since a DDS's
		specific compression (DXT1 vs DXT5 vs BC5, ...) isn't recoverable
		from its extension alone anyway.
		"""
		if not self.preview_source or not Path(self.preview_source).is_file():
			messagebox.showinfo(
				APP_NAME,
				"Nothing to export yet. Generate an image, or click one in the "
				"file list to preview it first.")
			return
		if not self.preview_origin or not Path(self.preview_origin).is_file():
			messagebox.showinfo(
				APP_NAME,
				"This preview isn't linked to a file to replace. Generate "
				"from an image-to-image task, or click an input file in the "
				"list to preview it, then try again.")
			return

		if self.worker.busy or self._batch_busy:
			messagebox.showinfo(
				APP_NAME, "Wait for the running job to finish first - it may be "
				"reading or rewriting that original.")
			return

		fmt = self.selected_export_format()
		result_source = Path(self.preview_source)
		origin = Path(self.preview_origin)
		target = origin.with_suffix(fmt.extension)
		same_extension = target.suffix.lower() == origin.suffix.lower()

		if same_extension:
			prompt = (f"Replace the original file?\n\n{target}\n\n"
			         f"This overwrites it with the current preview and "
			         f"cannot be undone.")
		else:
			prompt = (
				f"The chosen export format ({fmt.label}) doesn't match the "
				f"original's extension ({origin.suffix}), so this can't "
				f"overwrite it in place - it will save a new file next to it "
				f"instead:\n\n{target}\n\nContinue?")
		if not messagebox.askyesno(APP_NAME, prompt):
			return

		try:
			written = formats.convert_file(result_source, target, fmt)
		except Exception as exc:
			self._log(f"Replace original failed: {type(exc).__name__}: {exc}")
			messagebox.showerror(APP_NAME, f"Could not replace the original: {exc}")
			return

		size_kb = written.stat().st_size / 1024
		verb = "Replaced" if same_extension else "Saved alongside"
		self._log(f"{verb} {origin.name} -> {written} ({size_kb:.0f} KB)")
		self.v_export_note.set(f"{verb} {written.name} ({size_kb:.0f} KB).")
		# The file on disk now matches what's shown, so it's its own origin
		# for any further replace/export without re-selecting anything.
		self.preview_origin = str(written)
		self._refresh_grid_if_visible()    # the original may be in the file grid

	# -- results gallery ---------------------------------------------------
	def _sync_results_title(self, *_):
		status = self.v_results_status.get()
		self.results_box.configure(text=f"Results - {status}" if status else "Results")

	def _results_display(self):
		"""Gallery order: newest first. self.results itself stays oldest first."""
		return self.results[::-1]

	def _add_result(self, path, origin):
		self.results.append({"path": str(path),
		                     "origin": str(origin) if origin else None})
		self.results_grid.insert(0, str(path))
		self.results_grid.scroll_to_top()
		overflow = len(self.results) - MAX_RESULTS
		if overflow > 0:
			total = len(self.results)
			self.results_grid.remove(range(total - overflow, total))
			for entry in self.results[:overflow]:
				self._results_selected.discard(entry["path"])
			self.results = self.results[overflow:]
		self._refresh_results_selection()
		self._update_results_status()

	def _refresh_results_selection(self):
		shown = self._results_display()
		self.results_grid.set_selected(
			[i for i, e in enumerate(shown) if e["path"] in self._results_selected])

	def _update_results_status(self):
		n = len(self.results)
		if not n:
			self.v_results_status.set("")
			return
		linked = sum(1 for e in self.results if e.get("origin"))
		text = f"{n} image{'s' if n != 1 else ''}"
		if linked:
			text += f", {linked} made from an input file"
		self.v_results_status.set(text)

	def _result_at(self, index):
		shown = self._results_display()
		return shown[index] if 0 <= index < len(shown) else None

	def _result_click(self, index):
		entry = self._result_at(index)
		if entry is None:
			return
		self._results_selected = {entry["path"]}
		self._results_anchor = entry["path"]
		self._refresh_results_selection()
		self._preview_result(entry)

	def _result_ctrl_click(self, index):
		entry = self._result_at(index)
		if entry is None:
			return
		if entry["path"] in self._results_selected:
			self._results_selected.discard(entry["path"])
		else:
			self._results_selected.add(entry["path"])
			self._preview_result(entry)
		self._results_anchor = entry["path"]
		self._refresh_results_selection()

	def _result_shift_click(self, index):
		shown = self._results_display()
		if not 0 <= index < len(shown):
			return
		anchor = next((i for i, e in enumerate(shown)
		               if e["path"] == self._results_anchor), index)
		lo, hi = sorted((anchor, index))
		self._results_selected = {shown[i]["path"] for i in range(lo, hi + 1)}
		self._refresh_results_selection()
		self._preview_result(shown[index])

	def _result_open(self, index):
		entry = self._result_at(index)
		if entry and Path(entry["path"]).is_file():
			os.startfile(entry["path"])

	def _preview_result(self, entry):
		if Path(entry["path"]).is_file():
			self._set_preview(entry["path"], origin=entry.get("origin"))
		else:
			self._log(f"{Path(entry['path']).name} is no longer on disk.")

	def _results_remove(self):
		if not self._results_selected:
			messagebox.showinfo(
				APP_NAME, "Select one or more results first - click, Ctrl-click "
				"or Shift-click in the gallery.")
			return
		shown = self._results_display()
		gone = [i for i, e in enumerate(shown) if e["path"] in self._results_selected]
		self.results_grid.remove(gone)
		self.results = [e for e in self.results
		                if e["path"] not in self._results_selected]
		self._results_selected.clear()
		self._results_anchor = None
		self._update_results_status()
		self._log(f"Removed {len(gone)} result(s) from the list - the files "
		          f"themselves are untouched.")

	def _results_clear(self):
		if not self.results:
			return
		if not messagebox.askyesno(
				APP_NAME,
				f"Clear all {len(self.results)} results from the list?\n\n"
				"The files stay on disk. What goes is each result's link to "
				"the input it was made from, which Replace all needs."):
			return
		self.results = []
		self._results_selected.clear()
		self._results_anchor = None
		self.results_grid.clear()
		self._update_results_status()

	# -- Export all / Replace all ----------------------------------------------
	@staticmethod
	def _skipped_lines(plan):
		lines = []
		if plan.no_origin:
			lines.append(f"{plan.no_origin} made from text, with no original to replace")
		if plan.older:
			lines.append(f"{plan.older} older result(s) of an original that has a "
			             f"newer one - the newest is used")
		if plan.missing_result:
			lines.append(f"{plan.missing_result} whose result file is gone from disk")
		if plan.missing_origin:
			lines.append(f"{plan.missing_origin} whose original file is gone from disk")
		if plan.clashes:
			lines.append(f"{plan.clashes} that would have written the same file as "
			             f"another")
		return lines

	def _batch_ready(self):
		if self._batch_busy:
			return False
		if not self.results:
			messagebox.showinfo(APP_NAME, "There are no results to work on yet.")
			return False
		return True

	def _export_all(self):
		if not self._batch_ready():
			return
		fmt = self.selected_export_format()
		folder = filedialog.askdirectory(
			title=f"Export all {len(self.results)} results as {fmt.label}",
			initialdir=self.v_out.get() or str(app_dir()))
		if not folder:
			return
		plan = batch.plan_export(self.results, folder, fmt, overwrite=False)
		if not plan.jobs:
			messagebox.showinfo(APP_NAME, "None of the result files exist any more.")
			return
		if plan.existing:
			answer = messagebox.askyesnocancel(
				APP_NAME,
				f"{plan.existing} of the names already exist in\n{folder}\n\n"
				"Yes - overwrite those files\n"
				"No - keep both (the new ones get a number)\n"
				"Cancel - don't export anything")
			if answer is None:
				return
			if answer:
				plan = batch.plan_export(self.results, folder, fmt, overwrite=True)
		self._run_batch("Exporting", "Exported", plan, fmt, folder)

	def _replace_all(self):
		if not self._batch_ready():
			return
		if self.worker.busy:
			messagebox.showinfo(
				APP_NAME, "A generation is still running and may be reading those "
				"originals - wait for it to finish first.")
			return
		fmt = self.selected_export_format()
		plan = batch.plan_replace(self.results, fmt)
		if not plan.jobs:
			why = self._skipped_lines(plan)
			messagebox.showinfo(
				APP_NAME, "Nothing to replace." + (
					"\n\n" + "\n".join(f"  - {w}" for w in why) if why else ""))
			return

		text = [f"Replace {len(plan.jobs)} original file(s) with their newest "
		        f"result, as {fmt.label}?", ""]
		if plan.in_place:
			text.append(f"  - {plan.in_place} overwritten in place")
		if plan.alongside:
			text.append(f"  - {plan.alongside} saved next to the original instead - "
			            f"its extension isn't {fmt.extension}, so it can't be "
			            f"overwritten in place")
		skipped = self._skipped_lines(plan)
		if skipped:
			text += ["", "Skipped:"] + [f"  - {line}" for line in skipped]
		text += ["", "This overwrites files and cannot be undone."]
		if not messagebox.askyesno(APP_NAME, "\n".join(text)):
			return
		self._run_batch("Replacing", "Replaced", plan, fmt, None)

	def _set_batch_buttons(self, enabled):
		state = "normal" if enabled else "disabled"
		for b in (self.btn_export_all, self.btn_replace_all,
		          self.btn_results_remove, self.btn_results_clear):
			b.configure(state=state)

	def _run_batch(self, doing, did, plan, fmt, where):
		"""Run a plan off the UI thread - a few hundred DDS encodes take a
		while - reporting progress into the gallery's status line."""
		self._batch_busy = True
		self._set_batch_buttons(False)
		total = len(plan.jobs)
		self.v_results_status.set(f"{doing} 0/{total}...")
		self._log(f"{doing} {total} file(s) as {fmt.label}...")

		def progress(done, count, job):
			self.queue.put(("batch_progress", (doing, done, count)))

		def work():
			try:
				failed = batch.run(plan.jobs, fmt, progress)
			except Exception as exc:
				failed = [(None, f"{type(exc).__name__}: {exc}")]
			self.queue.put(("batch_done", (did, total, failed, where)))

		threading.Thread(target=work, daemon=True).start()

	def _batch_finished(self, did, total, failed, where):
		self._batch_busy = False
		self._set_batch_buttons(True)
		done = total - len(failed)
		place = f" to {where}" if where else ""
		self._log(f"{did} {done} of {total} file(s){place}.")
		for job, error in failed:
			self._log(f"  failed: {job.source.name if job else '?'}: {error}")
		if failed:
			self.v_results_status.set(
				f"{did} {done}/{total} - {len(failed)} failed, see the log")
			self._reveal_log()
		else:
			self.v_results_status.set(f"{did} {done} file(s){place}.")
		# A replaced original may be on screen right now - redraw anything
		# that shows one. Thumbnails are cached by modification time, so the
		# file list picks the new picture up rather than the old one.
		self._refresh_grid_if_visible()
		self._render_preview()

	# -- dividers ---------------------------------------------------------------
	def _hresize_cursor(self):
		"""The left-right resize cursor the ttk dividers use on this platform
		(size_we on Windows), so both kinds of divider feel the same."""
		try:
			return self.tk.eval("set ::ttk::Cursors(hresize)") or "sb_h_double_arrow"
		except tk.TclError:
			return "sb_h_double_arrow"

	def _sash_px(self):
		"""The themed sash between panels, rounded up a little."""
		return self.px(8)

	def _right_min_height(self):
		"""What the right-hand panels need, stacked, to show everything."""
		names = self.right_pane.panes()
		need = sum(self.nametowidget(name).winfo_reqheight() for name in names)
		return need + max(0, len(names) - 1) * self._sash_px()

	def _fit_right_pane(self, event=None):
		"""Keep the preview side's PanedWindow as wide as what is visible and
		at least as tall as its panels need - taller only when the window has
		room to spare, which is exactly when there is nothing to scroll."""
		width = self.right_canvas.winfo_width()
		height = max(self.right_canvas.winfo_height(), self._right_min_height())
		self.right_canvas.itemconfigure(self._right_window, width=width, height=height)
		self.right_canvas.configure(scrollregion=(0, 0, width, height))

	def _place_sashes(self, tries=0):
		"""Put the dividers where they belong once the window has a real size:
		the saved positions if there are any, otherwise each panel's own
		height plus a share of the spare room. Either way no panel ends up
		shorter than it needs while there is room to avoid it - that squeeze
		is what used to hide the Describe tab's buttons."""
		if self.right_canvas.winfo_height() < 100:
			if tries < 30:
				self.after(100, lambda: self._place_sashes(tries + 1))
			return
		self._fit_right_pane()
		self.right_canvas.update_idletasks()

		pane = self.right_pane
		names = pane.panes()
		n = len(names)
		reqs = [self.nametowidget(name).winfo_reqheight() for name in names]
		sash = self._sash_px()
		gaps = (n - 1) * sash
		total = pane.winfo_height()
		spare = max(0, total - gaps - sum(reqs))

		fr = self._sash_fractions
		if fr and len(fr) == n - 1:
			bounds = [0] + [int(total * f) for f in fr] + [total]
			heights = [max(0, bounds[i + 1] - bounds[i] - (sash if i < n - 1 else 0))
			           for i in range(n)]
		else:
			# Preview and the two prose panels take most of the spare; the
			# results strip needs least.
			weights = [3, 1, 2, 2][:n] or [1] * n
			heights = [reqs[i] + spare * weights[i] // sum(weights) for i in range(n)]

		if total >= sum(reqs) + gaps:
			# Every panel keeps at least the height it asked for; a shortfall
			# is taken from whichever panel has the most room to give. This
			# is what stops the Log (the last panel) being squeezed.
			for i in range(n):
				if heights[i] >= reqs[i]:
					continue
				deficit = reqs[i] - heights[i]
				heights[i] = reqs[i]
				for j in sorted(range(n), key=lambda k: heights[k] - reqs[k],
				                reverse=True):
					if j == i or deficit <= 0:
						break
					give = min(deficit, heights[j] - reqs[j])
					heights[j] -= give
					deficit -= give

		pos, acc = [], 0
		for i in range(n - 1):
			acc += heights[i]
			pos.append(acc)
			acc += sash
		for _ in range(2):
			for i, p_ in enumerate(pos):
				pane.sashpos(i, p_)

		self._keep_left_width()

	# -- the controls / preview divider ---------------------------------------
	def _default_left_px(self):
		return self.px(self._left_width) if self._left_width else self.px(510)

	def _on_app_resize(self, event):
		"""Only a real window resize, not a sash drag: dragging the sash
		redistributes width inside main_pane and never changes this frame's
		own width."""
		if event.widget is self and event.width != self._last_app_w:
			self._last_app_w = event.width
			self._keep_left_width()

	def _remember_left_width(self, _event=None):
		"""After the user lets go of the divider, note where they put it - but
		only if the window was wide enough that it was a real choice and not
		the divider pinned against the preview side's minimum."""
		try:
			x, _y = self.main_pane.sash_coord(0)
			room = self.main_pane.winfo_width()
			if room > self.px(460) and x < room - self.px(380) - 4:
				self._left_target = x
		except tk.TclError:
			pass

	def _keep_left_width(self, _event=None):
		"""Hold the controls at their chosen width while the window resizes,
		giving the change to the preview side. tk.PanedWindow lets the window
		take space from the fixed pane once the stretchy one hits its minimum,
		so this re-asserts the width on every resize - clamped so the preview
		side always keeps at least its own minimum."""
		try:
			self.main_pane.update_idletasks()
			room = self.main_pane.winfo_width()
			if room < self.px(460):
				return
			want = self._left_target or self._default_left_px()
			x = max(self.px(400), min(want, room - self.px(380)))
			cur, _y = self.main_pane.sash_coord(0)
			if abs(cur - x) > 1:
				self.main_pane.sash_place(0, x, 1)
		except tk.TclError:
			pass

	def _left_width_state(self):
		"""The chosen controls width, in unscaled pixels so it means the same
		after a display-scale change."""
		want = self._left_target or self._default_left_px()
		return round(want / self.scale) if want else None

	def _sash_state(self):
		try:
			height = self.right_pane.winfo_height()
			if height < 200:
				return None
			n = len(self.right_pane.panes())
			return [round(self.right_pane.sashpos(i) / height, 3)
			        for i in range(n - 1)]
		except tk.TclError:
			return None

	def _reveal_log(self):
		"""Scroll the preview side down so the Log panel is in view - the Log
		is the bottom panel, so this is just scroll-to-end. Used when
		something has just been written there worth seeing."""
		try:
			self.right_canvas.update_idletasks()
			self.right_canvas.yview_moveto(1.0)
		except tk.TclError:
			pass

	# -- preview ----------------------------------------------------------
	def _set_preview(self, path, label=None, origin=None):
		"""`origin` is the pre-existing file this picture was edited from -
		defaults to `path` itself, so previewing a plain input file (nothing
		generated yet) still gives Replace Original something to target."""
		self.preview_source = path
		self.preview_origin = origin if origin else path
		self.v_preview_name.set(label or Path(path).name)
		self._render_preview()

	def _render_preview(self):
		if not self.preview_source or not Path(self.preview_source).is_file():
			return
		try:
			from PIL import Image, ImageTk
		except ImportError:
			return
		cw = max(self.canvas.winfo_width(), 50)
		ch = max(self.canvas.winfo_height(), 50)
		try:
			img = Image.open(self.preview_source)
			img.thumbnail((cw - 8, ch - 8), Image.LANCZOS)
			self.preview_image = ImageTk.PhotoImage(img)
		except Exception:
			return
		self.canvas.delete("all")
		self.canvas.create_image(cw // 2, ch // 2, image=self.preview_image)

	# -- log --------------------------------------------------------------
	def _log(self, text):
		self.txt_log.configure(state="normal")
		self.txt_log.insert("end", text + "\n")
		self.txt_log.see("end")
		self.txt_log.configure(state="disabled")

	# -- prompt improver --------------------------------------------------
	def selected_text_spec(self):
		label = self.v_textmodel.get()
		for spec in textgen.MODELS:
			if spec.label == label:
				return spec
		return textgen.BY_KEY[textgen.DEFAULT_MODEL]

	def _improve_targets(self):
		return {
			"prompt": self.txt_prompt,
			"negative": self.txt_neg,
			"base": self.txt_base_prompt,
		}

	def _set_improve_state(self, enabled):
		state = "normal" if enabled else "disabled"
		for b in (self.btn_improve_prompt, self.btn_improve_neg,
		          self.btn_improve_base):
			b.configure(state=state)

	def _on_textmodel_change(self, _event=None):
		self._refresh_textgen(background=True)

	def _refresh_textgen(self, background=True):
		"""Probe Ollama and the selected model, then update the status line.

		Runs off the UI thread: a refused connection is usually instant, but a
		misconfigured OLLAMA_HOST pointing somewhere real would otherwise hang
		the window for the length of the timeout.
		"""
		spec = self.selected_text_spec()

		def work():
			ok, message = textgen.available()
			installed = textgen.is_installed(spec.key) if ok else False
			self.queue.put(("textgen_state", (ok, installed, spec.key, message)))

		if background:
			threading.Thread(target=work, daemon=True).start()
		else:
			work()

	def _apply_textgen_state(self, ok, installed, model_key, message):
		spec = textgen.BY_KEY.get(model_key, self.selected_text_spec())
		self.textgen_ready = ok and installed
		self.textgen_available = ok
		self.btn_textunload.configure(state="normal" if ok else "disabled")
		if not ok:
			self.v_textstatus.set(f"{message} Press Check for setup steps.")
			self.btn_textsetup.configure(text="Check", state="normal")
		elif not installed:
			self.v_textstatus.set(
				f"Ollama is running, but {spec.label} is not downloaded yet "
				f"({spec.vram_note}).")
			self.btn_textsetup.configure(text="Download", state="normal")
		else:
			self.v_textstatus.set(f"Ready - {spec.label}, {spec.vram_note}.")
			self.btn_textsetup.configure(text="Check", state="normal")

	def _textgen_button(self):
		"""One button: sets up Ollama, downloads the model, or just re-checks."""
		if self.btn_textsetup.cget("text") == "Download":
			self._textgen_download()
			return
		ok, _ = textgen.available()
		if not ok:
			messagebox.showinfo(APP_NAME, textgen.setup_hint())
		self._refresh_textgen()

	def _textgen_download(self):
		if self._ollama_busy():
			return
		spec = self.selected_text_spec()
		self.textgen_busy = True
		self.btn_textsetup.configure(state="disabled")
		self.btn_textunload.configure(state="disabled")
		self._set_improve_state(False)
		self._log(f"Downloading {spec.label} into Ollama ({spec.vram_note})...")

		def work():
			try:
				textgen.pull(
					spec.key,
					on_progress=lambda s: self.queue.put(("textgen_status", s)),
				)
				self.queue.put(("textgen_pulled", spec.key))
			except Exception as exc:
				self.queue.put(("textgen_error", f"{type(exc).__name__}: {exc}"))

		threading.Thread(target=work, daemon=True).start()

	def _textgen_unload(self):
		"""Manually drop whatever Ollama is holding in memory right now.

		Mostly a safety net: every Improve click already unloads its model
		immediately afterwards (keep_alive=0), so under normal use there is
		nothing left to free. This covers the odd cases - Ollama was already
		running a model before this app opened, or something outside the app
		loaded one - where "just make sure the VRAM is free" is worth an
		explicit button rather than trusting that it already is."""
		if self._ollama_busy():
			return
		self.btn_textunload.configure(state="disabled")
		self.v_textstatus.set("Unloading...")

		def work():
			n = textgen.unload_all()
			self.queue.put(("textgen_unloaded", n))

		threading.Thread(target=work, daemon=True).start()

	def _improve(self, which):
		"""Rewrite one of the prompt boxes with the selected language model."""
		if self._ollama_busy():
			return
		widget = self._improve_targets()[which]
		text = widget.get("1.0", "end").strip()
		if not text:
			messagebox.showinfo(
				APP_NAME, "Write something in that box first, then press Improve.")
			return

		spec = self.selected_text_spec()
		self.textgen_busy = True
		self._set_improve_state(False)
		self.btn_textsetup.configure(state="disabled")
		self.btn_textunload.configure(state="disabled")
		self.v_textstatus.set(f"Rewriting with {spec.label}...")
		self._log(f"Improving the {which} prompt with {spec.label}")
		self._log(f"  before: {text}")

		def work():
			# The reachability check belongs here, not on the UI thread: an
			# OLLAMA_HOST pointing at a routable but dead address would
			# otherwise freeze the window for the whole timeout.
			ok, _ = textgen.available()
			if not ok:
				self.queue.put(("textgen_setup", None))
				return
			try:
				out = textgen.improve(text, spec.key, negative=(which == "negative"))
				self.queue.put(("textgen_done", (which, out)))
			except Exception as exc:
				self.queue.put(("textgen_error", str(exc)))

		threading.Thread(target=work, daemon=True).start()

	def _apply_improved(self, which, new_text):
		widget = self._improve_targets()[which]
		# Tk drops an undo separator between the delete and the insert while
		# autoseparators is on, which would make Ctrl+Z undo only half the
		# rewrite and leave the box empty. Turn it off just for the swap, so
		# the whole replacement is a single undo step, then restore it so
		# ordinary typing keeps its own separators.
		widget.configure(autoseparators=False)
		widget.edit_separator()
		widget.delete("1.0", "end")
		widget.insert("1.0", new_text)
		widget.edit_separator()
		widget.configure(autoseparators=True)
		self._log(f"  after:  {new_text}")
		self.v_textstatus.set("Rewritten - press Ctrl+Z in the box to undo.")
		self._textgen_idle()

	def _textgen_idle(self):
		self.textgen_busy = False
		self._set_improve_state(True)
		self.btn_textsetup.configure(state="normal")
		self.btn_textunload.configure(
			state="normal" if self.textgen_available else "disabled")

	# -- image -> text ------------------------------------------------------
	def selected_vision_spec(self):
		label = self.v_vision_model.get()
		for spec in vision.VISION_MODELS:
			if spec.label == label:
				return spec
		return vision.BY_KEY[vision.DEFAULT_MODEL]

	def selected_vision_style(self):
		label = self.v_vision_style.get()
		for style in vision.STYLES:
			if style.label == label:
				return style
		return vision.STYLE_BY_KEY[vision.DEFAULT_STYLE]

	def _ollama_busy(self):
		"""Prompt rewriting and image description share one Ollama server, and
		on a 12 GB card two models resident at once is what runs it out of
		room - so only one of them runs at a time."""
		return self.textgen_busy or self.vision_busy

	def _on_vision_model_change(self, _event=None):
		self._refresh_vision()

	def _refresh_vision(self):
		spec = self.selected_vision_spec()

		def work():
			ok, message = vision.available()
			installed = vision.is_installed(spec.key) if ok else False
			self.queue.put(("vision_state", (ok, installed, spec.key, message)))

		threading.Thread(target=work, daemon=True).start()

	def _apply_vision_state(self, ok, installed, model_key, message):
		spec = vision.BY_KEY.get(model_key, self.selected_vision_spec())
		self.vision_available = ok
		if not ok:
			self.v_vision_status.set(f"{message} Press Check for setup steps.")
			self.btn_vision_setup.configure(text="Check", state="normal")
		elif not installed:
			self.v_vision_status.set(
				f"Ollama is running, but {spec.label} is not downloaded yet "
				f"({spec.vram_note}).")
			self.btn_vision_setup.configure(text="Download", state="normal")
		else:
			self.v_vision_status.set(f"Ready - {spec.label}, {spec.vram_note}.")
			self.btn_vision_setup.configure(text="Check", state="normal")

	def _vision_button(self):
		if self.btn_vision_setup.cget("text") == "Download":
			self._vision_download()
			return
		ok, _ = vision.available()
		if not ok:
			messagebox.showinfo(APP_NAME, vision.setup_hint())
		self._refresh_vision()

	def _vision_download(self):
		if self._ollama_busy():
			return
		spec = self.selected_vision_spec()
		self.vision_busy = True
		self.btn_vision_setup.configure(state="disabled")
		self.btn_describe.configure(state="disabled")
		self._log(f"Downloading {spec.label} into Ollama ({spec.vram_note})...")

		def work():
			try:
				vision.pull(
					spec.key,
					on_progress=lambda s: self.queue.put(("vision_status", s)),
				)
				self.queue.put(("vision_pulled", spec.key))
			except Exception as exc:
				self.queue.put(("vision_error", f"{type(exc).__name__}: {exc}"))

		threading.Thread(target=work, daemon=True).start()

	def _describe_preview(self):
		"""Describe whatever is currently in the preview."""
		if self._ollama_busy():
			return
		if not self.preview_source or not Path(self.preview_source).is_file():
			messagebox.showinfo(
				APP_NAME,
				"Nothing to describe yet. Generate an image, or click one in "
				"the file list to preview it first.")
			return

		spec = self.selected_vision_spec()
		style = self.selected_vision_style()
		source = Path(self.preview_source)
		self.vision_busy = True
		self.btn_describe.configure(state="disabled")
		self.btn_vision_setup.configure(state="disabled")
		self.v_vision_status.set(f"Looking at {source.name} with {spec.label}...")
		self._log(f"Describing {source.name} with {spec.label} ({style.label})")

		def work():
			ok, _ = vision.available()
			if not ok:
				self.queue.put(("vision_setup", None))
				return
			try:
				text = vision.describe(str(source), spec.key, style.key)
				self.queue.put(("vision_done", text))
			except Exception as exc:
				self.queue.put(("vision_error", str(exc)))

		threading.Thread(target=work, daemon=True).start()

	def _apply_description(self, text):
		self.txt_describe.configure(autoseparators=False)
		self.txt_describe.edit_separator()
		self.txt_describe.delete("1.0", "end")
		self.txt_describe.insert("1.0", text)
		self.txt_describe.edit_separator()
		self.txt_describe.configure(autoseparators=True)
		self._log(f"  {text}")
		self.v_vision_status.set("Described. Edit it here, or send it to the prompt.")
		self._vision_idle()

	def _vision_idle(self):
		self.vision_busy = False
		self.btn_describe.configure(state="normal")
		self.btn_vision_setup.configure(state="normal")

	def _describe_text(self):
		return self.txt_describe.get("1.0", "end").strip()

	def _describe_to_prompt(self):
		text = self._describe_text()
		if not text:
			messagebox.showinfo(APP_NAME, "Describe an image first.")
			return
		# Same one-step-undo handling as the prompt improver: Tk would
		# otherwise split the replace into two, and Ctrl+Z would leave the
		# box empty rather than restoring what was there.
		self.txt_prompt.configure(autoseparators=False)
		self.txt_prompt.edit_separator()
		self.txt_prompt.delete("1.0", "end")
		self.txt_prompt.insert("1.0", text)
		self.txt_prompt.edit_separator()
		self.txt_prompt.configure(autoseparators=True)
		self._log("Description sent to the prompt (Ctrl+Z there undoes it).")
		self.v_vision_status.set("Sent to the prompt box.")

	def _describe_copy(self):
		text = self._describe_text()
		if not text:
			return
		self.master.clipboard_clear()
		self.master.clipboard_append(text)
		self.v_vision_status.set("Copied to the clipboard.")

	def _describe_clear(self):
		self.txt_describe.delete("1.0", "end")
		self.v_vision_status.set("")

	# -- running ----------------------------------------------------------
	def _int(self, var, name, minimum=0):
		try:
			value = int(float(var.get()))
		except ValueError:
			raise ValueError(f"{name} must be a whole number.")
		if value < minimum:
			raise ValueError(f"{name} must be at least {minimum}.")
		return value

	def _float(self, var, name):
		try:
			return float(var.get())
		except ValueError:
			raise ValueError(f"{name} must be a number.")

	def _build_config(self):
		mode = self.mode()
		spec = self.selected_spec()
		if spec is None:
			raise ValueError("Pick a model first.")

		prompt = self.txt_prompt.get("1.0", "end").strip()
		if not prompt:
			raise ValueError("The prompt is empty.")

		cfg = JobConfig(
			mode=mode,
			model_key=spec.key,
			base_model_key=self.selected_base_spec().key,
			quant=next((k for k, label in spec.quants
			            if label == self.v_quant.get()), registry.quant_for(spec, None)),
			prompt=prompt,
			negative_prompt=self.txt_neg.get("1.0", "end").strip(),
			base_prompt=self.txt_base_prompt.get("1.0", "end").strip(),
			inputs=self._files(),
			out_dir=self.v_out.get().strip() or str(app_dir() / "Output"),
			width=self._int(self.v_width, "Width"),
			height=self._int(self.v_height, "Height"),
			steps=self._int(self.v_steps, "Steps", minimum=1),
			guidance=self._float(self.v_guidance, "Guidance"),
			strength=self._float(self.v_strength, "Strength"),
			seed=self._int(self.v_seed, "Seed") if not self.v_random_seed.get() else 0,
			random_seed=self.v_random_seed.get(),
			num_images=self._int(self.v_count, "How many", minimum=1),
			sweep_min=self._int(self.v_sweep_min, "Sweep from"),
			sweep_max=self._int(self.v_sweep_max, "Sweep to"),
			memory_mode=next((k for k, label in MEMORY_MODES
			                  if label == self.v_memory.get()), "auto"),
		)

		if MODE_TASK[mode] == IMG2IMG and mode != MODE_CHAIN and not cfg.inputs:
			raise ValueError("Add at least one input image for this task.")
		if mode == MODE_CHAIN and not cfg.base_prompt:
			raise ValueError("The base image prompt is empty.")
		if not (0.0 <= cfg.strength <= 1.0):
			raise ValueError("Strength must be between 0.0 and 1.0.")
		return cfg

	def _start(self):
		if self.worker.busy:
			return
		if self._batch_busy:
			messagebox.showinfo(
				APP_NAME, "Wait for the export / replace to finish first - it may "
				"be rewriting files this job would read.")
			return
		try:
			cfg = self._build_config()
		except ValueError as exc:
			messagebox.showerror(APP_NAME, str(exc))
			return
		except Exception as exc:
			messagebox.showerror(APP_NAME, f"{type(exc).__name__}: {exc}")
			return

		self.btn_go.configure(state="disabled")
		self.btn_cancel.configure(state="normal")
		self.btn_unload.configure(state="disabled")
		self.pb_items["value"] = 0
		self.pb_steps["value"] = 0
		self.v_status.set("Starting...")
		self._log("-" * 60)
		self._log(f"{MODE_LABELS[cfg.mode]}  |  {registry.BY_KEY[cfg.model_key].label}")

		# Every Improve click already unloads its model right after replying
		# (keep_alive=0), so normally there is nothing left in Ollama's VRAM
		# by the time Generate is pressed. This is the belt-and-suspenders
		# case: Ollama was already holding a model before this app opened, or
		# a chat elsewhere left one loaded. Fire-and-forget in the background
		# - it is done well before the image worker process actually needs
		# the VRAM, and must never delay Generate if Ollama is slow/unreachable.
		if self.textgen_available:
			threading.Thread(target=textgen.unload_all, daemon=True).start()

		self.worker.submit(cfg)

	def _cancel(self):
		# A hard process-kill, not a cooperative flag: this is what makes
		# Cancel effective even mid model-load, when nothing else could
		# interrupt it. The worker process is gone once stop() returns, so
		# the UI can be reset immediately - there is no "done" message coming.
		self.v_status.set("Cancelling - stopping the worker process...")
		self.btn_cancel.configure(state="disabled")
		self.worker.stop()
		self._finish("Cancelled.")

	def _unload(self):
		if self.worker.busy:
			return
		if not self.worker.is_running():
			self.v_status.set("No model is loaded.")
			return
		self.worker.stop()
		self._log("Model unloaded, worker process stopped and VRAM freed.")
		self.v_status.set("Model unloaded.")

	def _finish(self, message):
		self.worker.busy = False
		self.v_status.set(message)
		self._log(message)
		self.btn_go.configure(state="normal")
		self.btn_cancel.configure(state="disabled")
		self.btn_unload.configure(state="normal")
		self.pb_steps["value"] = 0

	def _drain_queue(self):
		try:
			while True:
				kind, payload = self.queue.get_nowait()
				if kind == "log":
					self._log(payload)
				elif kind == "status":
					self.v_status.set(payload)
				elif kind == "device":
					self.v_device.set(payload)
				elif kind == "steps":
					done, total = payload
					self.pb_steps["maximum"] = max(total, 1)
					self.pb_steps["value"] = done
				elif kind == "items":
					done, total = payload
					self.pb_items["maximum"] = max(total, 1)
					self.pb_items["value"] = done
				elif kind == "saved":
					saved_path, saved_source = payload
					self._set_preview(saved_path, origin=saved_source)
					self._add_result(saved_path, saved_source)
				elif kind == "done":
					self._finish(payload)
				elif kind == "error":
					self._finish("Failed.")
					self._reveal_log()    # the traceback is there
					messagebox.showerror(APP_NAME, payload)
				elif kind == "batch_progress":
					doing, done, total = payload
					self.v_results_status.set(f"{doing} {done}/{total}...")
				elif kind == "batch_done":
					self._batch_finished(*payload)
				elif kind == "textgen_state":
					self._apply_textgen_state(*payload)
				elif kind == "textgen_status":
					self.v_textstatus.set(payload)
				elif kind == "textgen_done":
					self._apply_improved(*payload)
				elif kind == "textgen_pulled":
					self._log("Download finished.")
					self._textgen_idle()
					self._refresh_textgen()
				elif kind == "textgen_unloaded":
					n = payload
					self._log(
						f"Unloaded {n} Ollama model(s), VRAM freed."
						if n else "Nothing was loaded in Ollama.")
					self.v_textstatus.set(
						f"Unloaded {n} model(s)." if n else "Nothing was loaded.")
					self.btn_textunload.configure(
						state="normal" if self.textgen_available else "disabled")
				elif kind == "textgen_error":
					self._log(f"Prompt improver: {payload}")
					self.v_textstatus.set("The rewrite failed - see the log.")
					self._textgen_idle()
					messagebox.showerror(APP_NAME, payload)
				elif kind == "textgen_setup":
					self._textgen_idle()
					self._refresh_textgen()
					messagebox.showinfo(APP_NAME, textgen.setup_hint())
				elif kind == "vision_state":
					self._apply_vision_state(*payload)
				elif kind == "vision_status":
					self.v_vision_status.set(payload)
				elif kind == "vision_done":
					self._apply_description(payload)
				elif kind == "vision_pulled":
					self._log("Download finished.")
					self._vision_idle()
					self._refresh_vision()
				elif kind == "vision_error":
					self._log(f"Describe image: {payload}")
					self.v_vision_status.set("The description failed - see the log.")
					self._vision_idle()
					messagebox.showerror(APP_NAME, payload)
				elif kind == "vision_setup":
					self._vision_idle()
					self._refresh_vision()
					messagebox.showinfo(APP_NAME, vision.setup_hint())
				elif kind == "hf_state":
					self._apply_hf_state(*payload)
				elif kind == "hf_saved":
					self._hf_saved(payload)
				elif kind == "hf_removed":
					self._hf_set_busy(False)
					self._log("HuggingFace token removed.")
					self._refresh_hf()
				elif kind == "hf_error":
					self._hf_set_busy(False)
					self._log(f"HuggingFace token: {payload}")
					self._refresh_hf()
					messagebox.showerror(APP_NAME, payload)
				elif kind == "hf_access":
					self._hf_access_line(*payload)
				elif kind == "hf_access_done":
					self._hf_set_busy(False)
					self._refresh_hf()   # re-enables Remove when a token exists
		except queue.Empty:
			pass
		self.after(80, self._drain_queue)

	# -- HuggingFace token (Settings tab) ---------------------------------
	def _refresh_hf(self):
		"""Look up which account the active token belongs to, off the UI thread."""

		def work():
			token, source = hftoken.current()
			if not token:
				self.queue.put(("hf_state", (
					"No token saved. Gated models (FLUX.1 dev, SRPO, Krea 2) "
					"can't download until you add one.", False)))
				return
			try:
				name = hftoken.whoami(token)
				text = f"Logged in as {name}  ({hftoken.mask(token)}, from {source})."
			except ValueError as exc:
				text = f"A token is set ({hftoken.mask(token)}, from {source}), but: {exc}"
			self.queue.put(("hf_state", (text, True)))

		threading.Thread(target=work, daemon=True).start()

	def _apply_hf_state(self, text, has_token):
		self.v_hf_status.set(text)
		self.btn_hf_remove.configure(state="normal" if has_token else "disabled")

	def _hf_set_busy(self, busy):
		self.hf_busy = busy
		state = "disabled" if busy else "normal"
		for b in (self.btn_hf_save, self.btn_hf_check):
			b.configure(state=state)
		if busy:
			self.btn_hf_remove.configure(state="disabled")

	def _hf_toggle_show(self):
		self.ent_hf_token.configure(show="" if self.v_hf_show.get() else "•")

	def _hf_open_tokens_page(self):
		import webbrowser

		webbrowser.open(hftoken.TOKENS_URL)

	def _hf_save(self):
		if self.hf_busy:
			return
		token = self.v_hf_token.get().strip()
		if not token:
			messagebox.showinfo(APP_NAME, "Paste your HuggingFace access token into "
			                              "the box first.")
			return
		self._hf_set_busy(True)
		self.v_hf_status.set("Checking the token with HuggingFace...")

		def work():
			try:
				name = hftoken.save(token)
				self.queue.put(("hf_saved", name))
			except Exception as exc:
				self.queue.put(("hf_error", str(exc)))

		threading.Thread(target=work, daemon=True).start()

	def _hf_saved(self, name):
		self._hf_set_busy(False)
		self.v_hf_token.set("")   # don't leave it sitting in the box
		self._log(f"HuggingFace token saved - logged in as {name}.")
		if hftoken.env_override():
			messagebox.showwarning(APP_NAME, (
				f"The token for {name} was saved, but HF_TOKEN is set in the "
				"environment and HuggingFace uses that one instead. Remove the "
				"environment variable for the saved token to take effect."))
		self._refresh_hf()

	def _hf_remove(self):
		if self.hf_busy:
			return
		if not messagebox.askyesno(APP_NAME, (
				"Remove the saved HuggingFace token from this computer?\n\n"
				"Gated models will stop downloading until you add one again. "
				"Other HuggingFace tools on this computer share the same token.")):
			return
		self._hf_set_busy(True)

		def work():
			try:
				hftoken.remove()
				self.queue.put(("hf_removed", None))
			except Exception as exc:
				self.queue.put(("hf_error", str(exc)))

		threading.Thread(target=work, daemon=True).start()

	def _hf_check_access(self):
		if self.hf_busy:
			return
		self._hf_set_busy(True)
		self.txt_hf_access.configure(state="normal")
		self.txt_hf_access.delete("1.0", "end")
		self.txt_hf_access.configure(state="disabled")
		repos = hftoken.repos_to_check(registry.MODELS)

		def work():
			try:
				token, _ = hftoken.current()
				for repo, ok, note in hftoken.check_access(repos, token):
					self.queue.put(("hf_access", (repo, ok, note)))
			finally:
				self.queue.put(("hf_access_done", None))

		threading.Thread(target=work, daemon=True).start()

	def _hf_access_line(self, repo, ok, note):
		self.txt_hf_access.configure(state="normal")
		mark = "✓" if ok else "✗"
		self.txt_hf_access.insert("end", f"{mark} {repo}" + ("" if ok else f" - {note}")
		                          + "\n", () if ok else ("bad",))
		self.txt_hf_access.see("end")
		self.txt_hf_access.configure(state="disabled")

	# -- hardware banner --------------------------------------------------
	def probe_hardware(self):
		"""Import torch off the UI thread so the window is usable immediately."""

		def work():
			try:
				import torch

				if torch.cuda.is_available():
					p = torch.cuda.get_device_properties(0)
					text = (f"GPU: {p.name} - {p.total_memory / 1024 ** 3:.1f} GB VRAM "
					        f"| torch {torch.__version__}")
				else:
					text = (f"No CUDA GPU found - running on CPU, which is very slow. "
					        f"| torch {torch.__version__}")
			except Exception as exc:
				text = f"Could not load torch: {exc}"
			self.queue.put(("device", text))
			self.queue.put(("status", "Ready."))

		threading.Thread(target=work, daemon=True).start()

	# -- settings ---------------------------------------------------------
	def _collect_settings(self):
		return {
			"mode": self.mode(),
			"model": self.v_model.get(),
			"base_model": self.v_base_model.get(),
			"quant": self.v_quant.get(),
			"memory": self.v_memory.get(),
			"text_model": self.v_textmodel.get(),
			"export_format": self.v_export.get(),
			"results": self.results[-MAX_RESULTS:],
			"sashes": self._sash_state(),
			"left_width": self._left_width_state(),
			"vision_model": self.v_vision_model.get(),
			"vision_style": self.v_vision_style.get(),
			"prompt": self.txt_prompt.get("1.0", "end").strip(),
			"base_prompt": self.txt_base_prompt.get("1.0", "end").strip(),
			"negative": self.txt_neg.get("1.0", "end").strip(),
			"inputs": self._files(),
			"out": self.v_out.get(),
			"width": self.v_width.get(),
			"height": self.v_height.get(),
			"steps": self.v_steps.get(),
			"guidance": self.v_guidance.get(),
			"strength": self.v_strength.get(),
			"seed": self.v_seed.get(),
			"random_seed": self.v_random_seed.get(),
			"count": self.v_count.get(),
			"sweep_min": self.v_sweep_min.get(),
			"sweep_max": self.v_sweep_max.get(),
			"autodefaults": self.v_autodefaults.get(),
			"geometry": self.master.geometry(),
		}

	def _restore_geometry(self, value):
		"""Apply a saved window geometry, but only if it would land somewhere
		reachable. A monitor that's since been unplugged, or a display-scale
		change between sessions, can otherwise save a size/position that
		reopens off-screen or absurdly large - worse than just keeping the
		sane default main() already computed."""
		if not value:
			return
		m = re.match(r"^(\d+)x(\d+)\+(-?\d+)\+(-?\d+)$", value)
		if not m:
			return
		w, h, x, y = (int(v) for v in m.groups())
		sw = self.master.winfo_screenwidth()
		sh = self.master.winfo_screenheight()
		margin = 40   # at least this much of the title bar must stay grabbable
		if not (0 < w <= sw and 0 < h <= sh
		        and -margin <= x <= sw - margin and -margin <= y <= sh - margin):
			return
		self.master.geometry(value)

	def _restore_results(self, data):
		"""The gallery, tab and divider positions from last session. Kept
		apart from the rest of _load_settings so a malformed entry here can't
		stop the window geometry being restored after it."""
		try:
			saved = [e for e in data.get("results", [])
			         if isinstance(e, dict) and e.get("path")
			         and Path(e["path"]).is_file()]
			self.results = [{"path": e["path"], "origin": e.get("origin") or None}
			                for e in saved][-MAX_RESULTS:]
			self.results_grid.set_items([e["path"] for e in self._results_display()])
			self._update_results_status()
			fr = data.get("sashes")
			if (isinstance(fr, list) and 2 <= len(fr) <= 6
			        and all(isinstance(v, (int, float)) for v in fr)
			        and all(0.02 <= v <= 0.98 for v in fr)
			        and all(fr[i] < fr[i + 1] for i in range(len(fr) - 1))):
				self._sash_fractions = tuple(float(v) for v in fr)
			lw = data.get("left_width")
			if isinstance(lw, int) and 300 <= lw <= 3000:
				self._left_width = lw
				self._left_target = self.px(lw)
		except Exception:
			pass

	def _load_settings(self):
		try:
			data = json.loads(settings_path().read_text(encoding="utf-8"))
		except Exception:
			return
		try:
			if data.get("mode") in MODE_ORDER:
				self.cb_mode.current(MODE_ORDER.index(data["mode"]))
			for var, key in (
				(self.v_model, "model"), (self.v_base_model, "base_model"),
				(self.v_quant, "quant"), (self.v_memory, "memory"),
				(self.v_out, "out"), (self.v_width, "width"),
				(self.v_height, "height"), (self.v_steps, "steps"),
				(self.v_guidance, "guidance"), (self.v_strength, "strength"),
				(self.v_seed, "seed"), (self.v_count, "count"),
				(self.v_sweep_min, "sweep_min"), (self.v_sweep_max, "sweep_max"),
			):
				if data.get(key):
					var.set(data[key])
			if data.get("text_model") in [m.label for m in textgen.MODELS]:
				self.v_textmodel.set(data["text_model"])
			if data.get("vision_model") in [m.label for m in vision.VISION_MODELS]:
				self.v_vision_model.set(data["vision_model"])
			if data.get("vision_style") in [s.label for s in vision.STYLES]:
				self.v_vision_style.set(data["vision_style"])
			if data.get("export_format") in [f.label for f in formats.EXPORT_FORMATS]:
				self.v_export.set(data["export_format"])
				self._on_export_format_change()
			if self.v_memory.get() not in [label for _, label in MEMORY_MODES]:
				self.v_memory.set(MEMORY_MODES[0][1])
			self.v_random_seed.set(bool(data.get("random_seed", True)))
			self.v_autodefaults.set(bool(data.get("autodefaults", True)))
			for key, widget in (("prompt", self.txt_prompt),
			                    ("base_prompt", self.txt_base_prompt),
			                    ("negative", self.txt_neg)):
				if data.get(key):
					widget.insert("1.0", data[key])
				# Restored text is the baseline, not an edit - otherwise the
				# first Ctrl+Z after startup would wipe the loaded prompt.
				widget.edit_reset()
			self._add_paths([p for p in data.get("inputs", []) if Path(p).exists()])
			self._restore_results(data)
			self._restore_geometry(data.get("geometry"))
		except Exception:
			pass

	def _on_close(self):
		try:
			settings_path().write_text(
				json.dumps(self._collect_settings(), indent=2), encoding="utf-8")
		except Exception:
			pass
		self.worker.stop()
		self.master.destroy()


def enable_dpi_awareness():
	"""Without this Windows bitmap-stretches the window on a scaled display, so
	every label comes out blurry. Returns the display scale factor."""
	scale = 1.0
	if sys.platform != "win32":
		return scale
	try:
		import ctypes

		# PROCESS_PER_MONITOR_DPI_AWARE
		ctypes.windll.shcore.SetProcessDpiAwareness(2)
	except Exception:
		try:
			ctypes.windll.user32.SetProcessDPIAware()
		except Exception:
			return scale
	try:
		import ctypes

		dc = ctypes.windll.user32.GetDC(0)
		dpi = ctypes.windll.gdi32.GetDeviceCaps(dc, 88)   # LOGPIXELSX
		ctypes.windll.user32.ReleaseDC(0, dc)
		if dpi:
			scale = dpi / 96.0
	except Exception:
		pass
	return scale


def main():
	scale = enable_dpi_awareness()
	root = TkinterDnD.Tk() if HAS_DND else tk.Tk()
	root.title(APP_NAME)

	# Tk now works in real pixels, so fonts and widget padding have to be scaled
	# up by hand to stay the same physical size as the rest of the desktop.
	if scale > 1.05:
		root.tk.call("tk", "scaling", scale * 96 / 72)
		import tkinter.font as tkfont

		for name in tkfont.names(root):
			font = tkfont.nametofont(name, root)
			size = font.cget("size")
			if size:
				sign = 1 if size > 0 else -1
				font.configure(size=sign * max(8, int(round(abs(size) * scale))))

	# Ask for a comfortable size, but never larger than the screen minus the
	# taskbar - an oversized window gets clipped, not scrolled.
	width = min(int(1180 * scale), root.winfo_screenwidth() - int(60 * scale))
	height = min(int(900 * scale), root.winfo_screenheight() - int(90 * scale))
	root.geometry(f"{width}x{height}+{int(30 * scale)}+{int(20 * scale)}")
	root.minsize(int(820 * scale), int(560 * scale))
	try:
		style = ttk.Style()
		if "vista" in style.theme_names():
			style.theme_use("vista")
	except tk.TclError:
		pass
	app = App(root, scale=scale)
	app.probe_hardware()
	root.mainloop()
