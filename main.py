"""Entry point for AI Image Studio."""

import multiprocessing
import sys
import traceback
from pathlib import Path


def _crash_log_path():
	return Path(__file__).resolve().parent / "crash.log"


def main():
	# Generation jobs run in a child process (see app/worker_process.py) so
	# Cancel can hard-kill one instead of hoping it notices a flag - this
	# must run before anything else touches multiprocessing.
	multiprocessing.freeze_support()

	from app.iosink import ensure_streams

	ensure_streams()

	if "--selftest" in sys.argv:
		from app.selftest import run

		sys.exit(run())

	from app.gui import main as run_gui

	run_gui()


if __name__ == "__main__":
	try:
		main()
	except Exception:
		# run.bat launches this with pythonw.exe, which has no console - an
		# exception here would otherwise vanish with no visible sign anything
		# went wrong. Write it to a log file and, if a display is available,
		# show it in a dialog too.
		tb = traceback.format_exc()
		try:
			_crash_log_path().write_text(tb, encoding="utf-8")
		except Exception:
			pass
		try:
			import tkinter as tk
			from tkinter import messagebox

			root = tk.Tk()
			root.withdraw()
			messagebox.showerror(
				"AI Image Studio - startup failed",
				f"{tb}\n\nThis was also saved to crash.log next to main.py.",
			)
		except Exception:
			pass
		raise
