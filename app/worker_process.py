"""The child process that actually runs generation jobs.

Model loading and denoising are blocking calls into torch/CUDA with no safe
point to interrupt early from Python - a cooperative "please stop" flag only
gets checked between denoising steps, so it does nothing for the 20-60+
seconds a model load can take. Running each job in its own process turns
Cancel into an OS-level TerminateProcess instead: instant and unconditionally
effective, no matter what the job happens to be doing at the time.

The cost is that a cancelled job's loaded model is gone with it, so the next
job pays a fresh load - a fair trade against the app getting stuck and
needing to be restarted.

Only `worker_main` runs in the child. Keep its imports deferred (inside the
function body) so importing this module - which multiprocessing does in the
child just to find `worker_main` - stays cheap.
"""


def worker_main(job_queue, progress_queue):
	from .iosink import ensure_streams

	ensure_streams()

	from . import backend
	from .backend import Progress

	class QueueProgress(Progress):
		"""Forwards progress straight to the parent process over the queue."""

		def __init__(self, q):
			self.q = q

		def log(self, text):
			self.q.put(("log", text))

		def status(self, text):
			self.q.put(("status", text))

		def steps(self, done, total):
			self.q.put(("steps", (done, total)))

		def items(self, done, total):
			self.q.put(("items", (done, total)))

		def saved(self, path, source=None):
			self.q.put(("saved", (path, source)))

		def cancelled(self):
			# Cancellation happens by killing this whole process, so there is
			# never a cooperative flag worth checking here.
			return False

	progress_queue.put(("worker_ready", None))

	while True:
		cfg = job_queue.get()   # blocks until the GUI sends a job or None
		if cfg is None:
			return               # graceful shutdown, requested by the GUI
		cb = QueueProgress(progress_queue)
		try:
			backend.run_job(cfg, cb)
			progress_queue.put(("done", "Finished."))
		except Exception as exc:
			import traceback

			progress_queue.put(("log", traceback.format_exc()))
			progress_queue.put(("error", f"{type(exc).__name__}: {exc}"))
