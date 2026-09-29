"""Image formats for a texture workflow: reading DDS/TGA input, and exporting
a finished picture to compressed DDS or the usual formats.

Pillow decodes DDS (DXT1/3/5, BC1-BC7, ATI1/2, uncompressed) and TGA straight
into an ordinary image, so an input .dds needs no conversion step on disk - it
is decoded in memory and handed to the pipeline exactly like a PNG would be.
Only the export side has to pick an encoding.
"""

from dataclasses import dataclass
from pathlib import Path


# Everything Pillow can decode that is worth offering as an input.
READABLE_EXTS = (
	".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".dds", ".tga",
)


@dataclass(frozen=True)
class ExportFormat:
	key: str                      # stable id used in saved settings
	label: str                    # what the dropdown shows
	extension: str
	pixel_format: str = ""        # DDS block compression Pillow wants; "" = none
	note: str = ""
	drop_alpha: bool = False      # JPEG cannot carry an alpha channel
	force_rgb: bool = False       # BC5 refuses anything but RGB


# DXT1/DXT3/DXT5 are the D3D9 names; BC1/BC2/BC3 are the same encodings under
# their DirectX 10+ names, which is why both are shown on the same row.
EXPORT_FORMATS = (
	ExportFormat("png", "PNG", ".png", note="Lossless, keeps alpha."),
	ExportFormat("dds_dxt5", "DDS - DXT5 / BC3", ".dds", "DXT5",
	             "Smooth alpha. The usual choice for game textures."),
	ExportFormat("dds_dxt1", "DDS - DXT1 / BC1", ".dds", "DXT1",
	             "Smallest at 4 bits/pixel; alpha is on/off only."),
	ExportFormat("dds_dxt3", "DDS - DXT3 / BC2", ".dds", "DXT3",
	             "Sharp-edged alpha rather than smooth."),
	ExportFormat("dds_bc5", "DDS - BC5 (normal maps)", ".dds", "BC5",
	             "Red and green only - blue is discarded. For normal maps.",
	             force_rgb=True),
	ExportFormat("dds_raw", "DDS - uncompressed", ".dds",
	             note="No quality loss, but the largest file."),
	ExportFormat("tga", "TGA", ".tga", note="Uncompressed, accepted everywhere."),
	ExportFormat("webp", "WebP", ".webp", note="Small, keeps alpha."),
	ExportFormat("jpg", "JPEG", ".jpg", note="Lossy, drops alpha.",
	             drop_alpha=True),
	ExportFormat("bmp", "BMP", ".bmp", note="Uncompressed."),
	ExportFormat("tiff", "TIFF", ".tif", note="Uncompressed, keeps alpha."),
)

BY_KEY = {f.key: f for f in EXPORT_FORMATS}
DEFAULT_EXPORT = "png"


def describe(image):
	"""A short label for what was just decoded, for the log line."""
	fmt = (getattr(image, "format", "") or "").upper()
	# Pillow's DDS reader records which block format it found.
	pixel = getattr(image, "pixel_format", "")
	if fmt and pixel:
		return f"{fmt} {pixel} ({image.mode})"
	if fmt:
		return f"{fmt} ({image.mode})"
	return image.mode


def open_image(path):
	"""Open any supported input, loading it fully so the file handle closes.

	Raises ValueError with something readable when Pillow cannot decode that
	particular variant - a DDS holding an exotic format fails deep inside the
	decoder otherwise, with nothing pointing at the file that caused it.
	"""
	from PIL import Image

	try:
		image = Image.open(path)
		image.load()
		return image
	except Exception as exc:
		raise ValueError(
			f"Could not read {Path(path).name}: {exc}"
		) from exc


def save_as(image, path, fmt):
	"""Write a PIL image to `path` using the given ExportFormat."""
	if fmt.force_rgb:
		# BC5 stores two channels and rejects every other mode outright.
		if image.mode != "RGB":
			image = image.convert("RGB")
	elif fmt.drop_alpha:
		if image.mode != "RGB":
			image = image.convert("RGB")
	elif image.mode not in ("RGB", "RGBA", "L", "LA"):
		# Palette images and the like: DDS in particular only accepts these four.
		image = image.convert("RGBA" if "A" in image.getbands() else "RGB")

	kwargs = {}
	if fmt.pixel_format:
		kwargs["pixel_format"] = fmt.pixel_format
	if fmt.key == "jpg":
		kwargs["quality"] = 95

	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	image.save(path, **kwargs)
	return path


def convert_file(src, dst, fmt):
	"""Re-encode a file on disk into another format."""
	image = open_image(src)
	return save_as(image, dst, fmt)
