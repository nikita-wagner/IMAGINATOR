"""Planning and carrying out Export all / Replace all over the results gallery.

Kept free of Tk so the planning - which files, under what names, what gets
skipped and why - can be tested on its own. That matters more than usual
here: Replace all overwrites files, the plan is exactly what the confirmation
dialog shows before anything is touched, and so it has to be right.

Results are dicts with "path" (the generated file) and "origin" (the input it
was edited from, or None for a text-to-image result), in generation order,
oldest first - the same list the gallery keeps.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from . import formats
from .backend import fit_filename


def _key(path):
	"""Compare paths the way Windows does: case-insensitive and normalised."""
	return os.path.normcase(os.path.abspath(str(path)))


@dataclass
class Job:
	source: Path
	target: Path
	in_place: bool = False     # overwrites the very file the result came from
	order: int = 0             # position in the results list; higher = newer


@dataclass
class Plan:
	jobs: list = field(default_factory=list)
	# Why results were left out. Every result lands in exactly one of these or
	# becomes a job, which is what the tests check.
	no_origin: int = 0         # made from text, so there is nothing to replace
	older: int = 0             # an older result of an original with a newer one
	missing_result: int = 0    # the generated file is gone from disk
	missing_origin: int = 0    # the original file is gone from disk
	clashes: int = 0           # would have written the same file as another job
	# Export only.
	existing: int = 0          # names already taken in the folder beforehand
	renamed: int = 0           # numbered to keep two files apart

	@property
	def in_place(self):
		return sum(1 for j in self.jobs if j.in_place)

	@property
	def alongside(self):
		return sum(1 for j in self.jobs if not j.in_place)


def plan_replace(results, fmt):
	"""Replace all: each original is overwritten by its newest result.

	An original edited several times only gets its newest result - writing
	every variant in turn would leave the last one there anyway, after
	pointlessly overwriting the file several times over. A result whose file
	has gone missing is passed over, so the newest one still on disk is used.
	"""
	plan = Plan()
	newest = {}
	for order, entry in enumerate(results):
		origin = entry.get("origin")
		if not origin:
			plan.no_origin += 1
			continue
		if not Path(entry["path"]).is_file():
			plan.missing_result += 1
			continue
		if not Path(origin).is_file():
			plan.missing_origin += 1
			continue
		key = _key(origin)
		if key in newest:
			plan.older += 1
		newest[key] = (order, entry)

	by_target = {}
	for order, entry in newest.values():
		origin = Path(entry["origin"])
		target = origin.with_suffix(fmt.extension)
		in_place = target.suffix.lower() == origin.suffix.lower()
		by_target.setdefault(_key(target), []).append(
			Job(Path(entry["path"]), target, in_place, order))

	for candidates in by_target.values():
		if len(candidates) == 1:
			plan.jobs.append(candidates[0])
			continue
		# rock.dds and rock.png both edited, PNG chosen: rock.dds's result
		# would be saved "alongside" as rock.png - straight on top of
		# rock.png's own in-place replace. The in-place one is what the user
		# asked for; failing that, keep the newest.
		in_place = [c for c in candidates if c.in_place]
		keep = in_place[0] if in_place else max(candidates, key=lambda c: c.order)
		plan.jobs.append(keep)
		plan.clashes += len(candidates) - 1

	plan.jobs.sort(key=lambda j: j.order)
	return plan


def _claim(folder, name, overwrite, claimed):
	"""The target for `name`: the plain name if it may be used, otherwise the
	first free "name_1", "name_2"... Numbering never overwrites anything,
	even in overwrite mode - that mode only covers the plain names."""
	target = folder / name
	if _key(target) not in claimed and (overwrite or not target.exists()):
		return target
	stem, ext = os.path.splitext(name)
	n = 1
	while True:
		candidate = folder / f"{stem}_{n}{ext}"
		if _key(candidate) not in claimed and not candidate.exists():
			return candidate
		n += 1


def plan_export(results, folder, fmt, overwrite=False, original_names=True):
	"""Export all: every result into `folder`, in `fmt`.

	A result edited from an input file is named after that input - rock.dds
	edited and exported as DDS comes out as rock.dds - which is what lets a
	batch of edited textures drop straight back into a game folder. Newest
	first, so when several results share a name the newest one gets the plain
	name and the older ones are numbered after it.

	`existing` counts the names already taken in the folder, whichever mode is
	chosen - it is what the caller asks the user about before deciding.
	"""
	plan = Plan()
	folder = Path(folder)
	claimed = set()
	taken = set()
	for order in range(len(results) - 1, -1, -1):
		entry = results[order]
		source = Path(entry["path"])
		if not source.is_file():
			plan.missing_result += 1
			continue
		origin = entry.get("origin")
		named_after = Path(origin) if (original_names and origin) else source
		name = fit_filename(folder, named_after.stem + fmt.extension)
		wanted = folder / name
		if wanted.exists() and _key(wanted) not in taken:
			taken.add(_key(wanted))
			plan.existing += 1
		target = _claim(folder, name, overwrite, claimed)
		if _key(target) != _key(wanted):
			plan.renamed += 1
		claimed.add(_key(target))
		plan.jobs.append(Job(source, target, False, order))
	return plan


def run(jobs, fmt, progress=None):
	"""Carry out a plan. Returns [(job, error message)] for whatever failed.

	A source that is also another job's target - a chained edit, where one
	result was itself the input for the next - is read into memory before
	anything is written. So every job converts its file as it was when the
	plan was made, whatever order the writes happen in.
	"""
	targets = {_key(j.target) for j in jobs}
	preloaded = {}
	for job in jobs:
		k = _key(job.source)
		if k in targets and k not in preloaded:
			try:
				preloaded[k] = formats.open_image(job.source)
			except Exception:
				pass    # the job itself reports it when it runs

	failed = []
	for i, job in enumerate(jobs, 1):
		try:
			image = preloaded.get(_key(job.source))
			if image is None:
				image = formats.open_image(job.source)
			formats.save_as(image, job.target, fmt)
		except Exception as exc:
			failed.append((job, f"{type(exc).__name__}: {exc}"))
		if progress:
			progress(i, len(jobs), job)
	return failed
