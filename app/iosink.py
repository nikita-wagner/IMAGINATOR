"""A safe fallback so libraries can print() even with no console attached.

pythonw.exe (and any process it spawns) leaves sys.stdout / sys.stderr as
None. Libraries like tqdm and huggingface_hub write progress bars straight to
those streams and crash with AttributeError the instant they try - so every
entry point that might run without a console installs this first.
"""


class Sink:
	def write(self, _text):
		return 0

	def flush(self):
		pass

	def isatty(self):
		return False


def ensure_streams():
	import sys

	if sys.stdout is None:
		sys.stdout = Sink()
	if sys.stderr is None:
		sys.stderr = Sink()
