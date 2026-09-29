"""A scrollable grid of thumbnail tiles, shared by the Files panel and the
results gallery.

Thumbnails are decoded with Pillow on a background thread - a Tk PhotoImage
belongs to the interpreter and has to be built on the UI thread - and handed
back through a queue this widget polls for itself, only while work is
outstanding. So it needs no wiring into the app's own message loop.

Every tile has its own id rather than being addressed by position. That is
what lets tiles be inserted and removed while decodes are still in flight: a
thumbnail whose tile has since gone is simply dropped, instead of landing on
whatever tile now happens to sit at that index.

Selection belongs to whoever owns the grid. Clicks are reported through the
callbacks as a position in the current order, and set_selected() only draws
the highlight - which lets the Files panel keep using its listbox as the one
record of what is selected, while the gallery keeps its own.
"""

import queue
import threading
from pathlib import Path

import tkinter as tk
from tkinter import ttk

from . import backend
from . import formats


# Decoded thumbnails, shared by every grid. Keyed on mtime and size as well as
# the path, so a file that is overwritten in place - Replace original does
# exactly that - gets decoded again rather than still showing the old picture.
_CACHE = {}
_CACHE_LOCK = threading.Lock()
_CACHE_MAX = 600


def short_name(name, limit=22):
	stem = Path(name).stem
	if len(stem) <= limit:
		return name
	return stem[: limit - 1] + "…" + Path(name).suffix


def _cache_key(path, size):
	try:
		st = Path(path).stat()
	except OSError:
		return None
	return (str(Path(path).resolve()).lower(), st.st_mtime_ns, st.st_size, size)


def decode(path, size):
	"""("image", PIL image), ("folder", count) or ("broken", None).

	A folder is one entry in the Files list - it only gets expanded into its
	pictures at job time, so files dropped into it later are still used - so
	it gets one tile, showing a count instead of a picture.
	"""
	p = Path(path)
	if p.is_dir():
		try:
			return ("folder", len(backend.collect_images([str(p)])))
		except Exception:
			return ("folder", 0)

	key = _cache_key(p, size)
	if key is not None:
		with _CACHE_LOCK:
			hit = _CACHE.get(key)
		if hit is not None:
			return ("image", hit)
	try:
		from PIL import Image

		image = formats.open_image(p)
		if image.mode not in ("RGB", "RGBA"):
			image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
		image.thumbnail((size, size), Image.LANCZOS)
	except Exception:
		return ("broken", None)
	if key is not None:
		with _CACHE_LOCK:
			_CACHE[key] = image
			while len(_CACHE) > _CACHE_MAX:
				_CACHE.pop(next(iter(_CACHE)))
	return ("image", image)


class ThumbnailGrid(ttk.Frame):
	BG = "#1e1e1e"
	TILE_BG = "#2a2a2a"
	SELECTED = "#3a7bd5"

	def __init__(self, master, scale=1.0, height=140, thumb=86, empty_text="",
	             on_click=None, on_ctrl_click=None, on_shift_click=None,
	             on_double=None):
		super().__init__(master)
		self.scale = scale
		self.thumb = thumb
		self.on_click = on_click
		self.on_ctrl_click = on_ctrl_click
		self.on_shift_click = on_shift_click
		self.on_double = on_double

		self.canvas = tk.Canvas(self, height=self.px(height), highlightthickness=1,
		                        highlightbackground=self.BG, background=self.BG)
		self.scrollbar = ttk.Scrollbar(self, orient="vertical",
		                               command=self.canvas.yview)
		self.canvas.configure(yscrollcommand=self.scrollbar.set)
		self.columnconfigure(0, weight=1)
		self.rowconfigure(0, weight=1)
		self.canvas.grid(row=0, column=0, sticky="nsew")
		self.scrollbar.grid(row=0, column=1, sticky="ns")

		self.inner = tk.Frame(self.canvas, background=self.BG)
		self._window = self.canvas.create_window((0, 0), window=self.inner,
		                                         anchor="nw")
		self._empty = self.canvas.create_text(
			self.px(10), self.px(10), anchor="nw", text=empty_text,
			fill="#777", font=("", 9))
		self.inner.bind("<Configure>", lambda e: self.canvas.configure(
			scrollregion=self.canvas.bbox("all")))
		self.canvas.bind("<Configure>", self._on_resize)
		for w in (self.canvas, self.inner, self.scrollbar):
			w.bind("<MouseWheel>", self._on_wheel)

		self.paths = []        # display order
		self.tiles = []        # (tid, frame, thumb label, name label), same order
		self.photos = {}       # tid -> PhotoImage; Tk drops an image nobody holds
		self._by_tid = {}
		self._next_tid = 0
		self._jobs = queue.Queue()
		self._done = queue.Queue()
		self._pending = 0
		self._polling = False
		self._worker = None
		self._update_empty()

	def px(self, value):
		return int(value * self.scale)

	# -- contents ------------------------------------------------------------
	def set_items(self, paths, labels=None):
		for _tid, frame, _thumb, _name in self.tiles:
			frame.destroy()
		self.tiles = []
		self.paths = []
		self._by_tid.clear()
		self.photos.clear()
		for i, path in enumerate(paths):
			self._add_tile(i, path, labels[i] if labels else None)
		self._changed()

	def insert(self, index, path, label=None):
		self._add_tile(index, path, label)
		self._changed()

	def remove(self, indices):
		for i in sorted(set(indices), reverse=True):
			if 0 <= i < len(self.tiles):
				tid, frame, _thumb, _name = self.tiles.pop(i)
				self.paths.pop(i)
				frame.destroy()
				self._by_tid.pop(tid, None)
				self.photos.pop(tid, None)
		self._changed()

	def clear(self):
		self.set_items([])

	def set_selected(self, indices):
		chosen = set(indices)
		for i, (_tid, frame, _thumb, _name) in enumerate(self.tiles):
			color = self.SELECTED if i in chosen else self.BG
			frame.configure(highlightbackground=color, highlightcolor=color)

	def scroll_to_top(self):
		self.canvas.yview_moveto(0)

	# -- tiles -----------------------------------------------------------------
	def _add_tile(self, index, path, label):
		tid = self._next_tid
		self._next_tid += 1
		box_px = self.px(self.thumb + 4)
		frame = tk.Frame(self.inner, background=self.BG, highlightthickness=2,
		                 highlightbackground=self.BG, highlightcolor=self.BG)
		box = tk.Frame(frame, width=box_px, height=box_px, background=self.TILE_BG)
		box.pack_propagate(False)
		box.pack(padx=4, pady=(4, 2))
		thumb = tk.Label(box, background=self.TILE_BG, foreground="#666",
		                 text="…", font=("", 9))
		thumb.pack(expand=True)
		name = tk.Label(frame, text=label or short_name(Path(path).name),
		                foreground="#ccc", background=self.BG,
		                wraplength=self.px(self.thumb + 8), justify="center",
		                font=("", 8))
		name.pack(padx=4, pady=(0, 4))

		for w in (frame, box, thumb, name):
			w.bind("<Button-1>", lambda e, t=tid: self._fire(self.on_click, t))
			w.bind("<Control-Button-1>",
			       lambda e, t=tid: self._fire(self.on_ctrl_click, t))
			w.bind("<Shift-Button-1>",
			       lambda e, t=tid: self._fire(self.on_shift_click, t))
			w.bind("<Double-Button-1>", lambda e, t=tid: self._fire(self.on_double, t))
			w.bind("<MouseWheel>", self._on_wheel)

		tile = (tid, frame, thumb, name)
		self.tiles.insert(index, tile)
		self.paths.insert(index, str(path))
		self._by_tid[tid] = tile
		self._queue_decode(tid, path)

	def _fire(self, callback, tid):
		if callback is None:
			return
		for i, tile in enumerate(self.tiles):
			if tile[0] == tid:
				callback(i)
				return

	# -- thumbnails --------------------------------------------------------------
	def _queue_decode(self, tid, path):
		self._pending += 1
		self._jobs.put((tid, str(path), self.px(self.thumb)))
		if self._worker is None:
			self._worker = threading.Thread(target=self._work, daemon=True)
			self._worker.start()
		if not self._polling:
			self._polling = True
			self.after(50, self._poll)

	def _work(self):
		while True:
			tid, path, size = self._jobs.get()
			# A tile removed while it waited is not worth decoding. Reading
			# the dict from here is a single membership test, safe under the
			# GIL; at worst one stale decode slips through and gets dropped.
			if tid not in self._by_tid:
				result = ("gone", None)
			else:
				result = decode(path, size)
			self._done.put((tid, result))

	def _poll(self):
		try:
			while True:
				tid, result = self._done.get_nowait()
				self._pending -= 1
				self._apply(tid, result)
		except queue.Empty:
			pass
		if self._pending > 0:
			try:
				self.after(50, self._poll)
				return
			except tk.TclError:
				pass    # the widget is being torn down
		self._polling = False

	def _apply(self, tid, result):
		tile = self._by_tid.get(tid)
		if tile is None:
			return
		_tid, _frame, thumb, _name = tile
		kind, payload = result
		try:
			if kind == "folder":
				plural = "" if payload == 1 else "s"
				thumb.configure(text=f"📁\n{payload} image{plural}", image="")
			elif kind == "image":
				from PIL import ImageTk

				photo = ImageTk.PhotoImage(payload)
				self.photos[tid] = photo
				thumb.configure(image=photo, text="")
			else:
				thumb.configure(text="?", image="")
		except tk.TclError:
			pass

	# -- layout ------------------------------------------------------------------
	def _changed(self):
		self._layout()
		self._update_empty()

	def _layout(self):
		width = max(self.canvas.winfo_width(), self.px(160))
		cols = max(1, width // self.px(self.thumb + 24))
		for i, (_tid, frame, _thumb, _name) in enumerate(self.tiles):
			row, col = divmod(i, cols)
			frame.grid(row=row, column=col, padx=3, pady=3, sticky="n")

	def _on_resize(self, event):
		self.canvas.itemconfigure(self._window, width=event.width)
		self._layout()

	def _update_empty(self):
		self.canvas.itemconfigure(self._empty,
		                          state="hidden" if self.tiles else "normal")

	def _on_wheel(self, event):
		self.canvas.yview_scroll(-1 * (event.delta // 120), "units")
