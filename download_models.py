"""Download the GGUF models for the benchmark straight into the LM Studio models folder.

    python download_models.py            # download everything missing
    python download_models.py --dry-run  # only show the plan
    python download_models.py --hashes   # sha256 of every .gguf in the models folder (for the paper appendix)

Layout follows LM Studio: <models dir>/<publisher>/<repo>/<file>.gguf, so the models show up in LM Studio
without any import step. Files that already exist with the right size are skipped; partial downloads resume.
"""
import argparse
import hashlib
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Xet backend with all cores: the fast path of huggingface_hub (needs `pip install huggingface_hub[hf_xet]`).
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from huggingface_hub import HfApi, hf_hub_download

# LM Studio's default models folder; override with the LMSTUDIO_MODELS env var if you moved it.
MODELS_DIR = Path(os.environ.get("LMSTUDIO_MODELS", Path.home() / ".lmstudio" / "models"))

# (repo, file). All Q4_K_M: ~5 GB for 7-9B, so the model plus a 4096-token KV cache fit into 8 GB VRAM.
MODELS = [
    ("lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
    ("mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated-GGUF", "meta-llama-3.1-8b-instruct-abliterated.Q4_K_M.gguf"),
    ("lmstudio-community/gemma-3-4b-it-GGUF", "gemma-3-4b-it-Q4_K_M.gguf"),
    ("lmstudio-community/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf"),
    ("lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF", "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"),
    ("bartowski/Ministral-8B-Instruct-2410-GGUF", "Ministral-8B-Instruct-2410-Q4_K_M.gguf"),
    ("lmstudio-community/Phi-4-mini-instruct-GGUF", "Phi-4-mini-instruct-Q4_K_M.gguf"),
    ("lmstudio-community/granite-3.3-8b-instruct-GGUF", "granite-3.3-8b-instruct-Q4_K_M.gguf"),
]
# Already on disk, nothing to download: Qwen3-8B (lmstudio-community), Ministral-3-3B, Qwen3.5-9B-Uncensored (hf/unsec).


def plan():
    api = HfApi()
    items = []
    for repo, fname in MODELS:
        info = api.model_info(repo, files_metadata=True)
        size = next(s.size for s in info.siblings if s.rfilename == fname)
        dest = MODELS_DIR / repo.split("/")[0] / repo.split("/")[1] / fname
        items.append((repo, fname, size, dest))
    return items


def fetch(item):
    repo, fname, size, dest = item
    if dest.exists() and dest.stat().st_size == size:
        return f"skip  {fname} (already complete)"
    hf_hub_download(repo_id=repo, filename=fname, local_dir=dest.parent)
    got = dest.stat().st_size
    if got != size:
        raise RuntimeError(f"{fname}: size {got} != expected {size}")
    return f"done  {fname} ({size / 1e9:.2f} GB)"


def hashes():
    """sha256 of every GGUF file, so the paper can name the exact weights (a re-uploaded GGUF can differ)."""
    for f in sorted(MODELS_DIR.rglob("*.gguf")):
        h = hashlib.sha256()
        with f.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 24), b""):
                h.update(block)
        print(f"{h.hexdigest()}  {f.relative_to(MODELS_DIR).as_posix()}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--hashes", action="store_true", help="print sha256 of the local GGUF files and exit")
    ap.add_argument("--workers", type=int, default=3, help="parallel files")
    a = ap.parse_args()
    if a.hashes:
        hashes()
        return

    items = plan()
    todo = [i for i in items if not (i[3].exists() and i[3].stat().st_size == i[2])]
    need = sum(i[2] for i in todo)
    free = shutil.disk_usage(MODELS_DIR.anchor).free
    print(f"{len(todo)}/{len(items)} files to download, {need / 1e9:.1f} GB needed, {free / 1e9:.1f} GB free")
    for repo, fname, size, _ in items:
        print(f"  {size / 1e9:5.2f} GB  {repo}/{fname}")
    if a.dry_run:
        return
    if need > free - 5e9:
        raise SystemExit("not enough free disk space")

    with ThreadPoolExecutor(a.workers) as ex:
        for msg in ex.map(fetch, items):
            print(msg, flush=True)
    print("all models present in", MODELS_DIR)


if __name__ == "__main__":
    main()
