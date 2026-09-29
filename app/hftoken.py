"""The HuggingFace access token: showing, saving, removing and testing it.

The token is kept exactly where huggingface_hub itself looks for it
(%USERPROFILE%\\.cache\\huggingface\\token), not in this app's settings.json.
That way every download - including the ones in the generation worker
process - picks it up with no plumbing, and `huggingface-cli` and any other
script on this machine share the same login.

Nothing here touches Tk; the GUI calls these off the UI thread, since every
check is a network round trip.
"""

import os

# Settings -> Access Tokens on huggingface.co. A "Read" token is enough for
# downloading, gated repos included.
TOKENS_URL = "https://huggingface.co/settings/tokens"


def current():
	"""(token or None, where it comes from) for display."""
	env = os.environ.get("HF_TOKEN")
	if env:
		# huggingface_hub prefers the environment over the saved file, so
		# saving from the app would appear to do nothing - say so up front.
		return env, "the HF_TOKEN environment variable"
	from huggingface_hub import get_token

	token = get_token()
	return token, ("the HuggingFace token file" if token else None)


def env_override():
	return bool(os.environ.get("HF_TOKEN"))


def mask(token):
	"""hf_abcd...wxyz - enough to recognise which token it is, not to use it."""
	if not token:
		return ""
	if len(token) <= 12:
		return token[:3] + "..."
	return f"{token[:7]}...{token[-4:]}"


def whoami(token=None):
	"""The account name a token belongs to. Raises with a readable message."""
	from huggingface_hub import whoami as hf_whoami

	try:
		info = hf_whoami(token=token or None)
	except Exception as exc:
		raise ValueError(_readable(exc)) from exc
	return info.get("name") or info.get("fullname") or "unknown account"


def save(token):
	"""Validate the token, then store it as the active HuggingFace login.

	Returns the account name. Raises ValueError, and stores nothing, when the
	token is rejected.
	"""
	token = token.strip()
	if not token:
		raise ValueError("Paste a token first.")
	name = whoami(token)   # validate before overwriting a working token
	from huggingface_hub import login

	login(token=token, add_to_git_credential=False, skip_if_logged_in=False)
	return name


def remove():
	"""Forget the saved token. The environment variable, if set, stays."""
	from huggingface_hub import logout

	try:
		logout()
	except Exception as exc:
		# logout() refuses outright while HF_TOKEN is set, and raises if
		# there was nothing saved - neither is worth an error dialog.
		if env_override():
			raise ValueError(
				"The saved token was removed, but HF_TOKEN is still set in the "
				"environment, so HuggingFace keeps using that one.") from exc


def repos_to_check(specs):
	"""Every HuggingFace repo the image models download from, deduplicated."""
	seen = []
	for spec in specs:
		for repo in (spec.repo, spec.gguf_repo, spec.lora_repo):
			if repo and repo not in seen:
				seen.append(repo)
	return seen


def check_access(repos, token=None):
	"""Yield (repo, ok, note) for each repo, as each check finishes.

	Gated repos (FLUX.1-dev, Krea 2, ...) need the licence accepted on the
	website with the same account the token belongs to; this is what tells
	you whether that has happened, before a download fails halfway.
	"""
	from huggingface_hub import HfApi
	from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

	api = HfApi()
	for repo in repos:
		try:
			api.auth_check(repo, token=token or None)
			yield repo, True, "ok"
		except GatedRepoError:
			yield repo, False, "gated - accept the licence on its HuggingFace page"
		except RepositoryNotFoundError:
			yield repo, False, "not found (private, or the token is invalid)"
		except Exception as exc:
			yield repo, False, _readable(exc)


def _readable(exc):
	"""Turn a requests/hub error into one line a person can act on."""
	text = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
	probe = f"{type(exc).__name__} {text}"
	if "401" in probe or "Invalid user token" in probe or "Unauthorized" in probe:
		return "HuggingFace rejected this token - check it was copied whole."
	if any(s in probe for s in ("ConnectionError", "ConnectError", "Failed to resolve",
	                            "Max retries", "timed out", "Timeout")):
		return "Could not reach huggingface.co - check the internet connection."
	return text[:200]
