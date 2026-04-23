#!/usr/bin/env python3
# --------------------------------------------------------
# Hugging Face Spaces entry point for the SAB3R demo.
#
# This file is duplicated to the Space root at upload time together with
# demo.py (and the mast3r/, dust3r/ library code). Spaces looks for app.py
# by default; this wrapper just pins Spaces-appropriate defaults and calls
# demo.py's main(). For local dev, run demo.py directly.
#
# FeatUp note:
#   FeatUp (https://github.com/mhamilton723/FeatUp) is not on PyPI and its
#   setup.py imports torch at build time. HF Spaces builds dependencies in
#   an isolated venv that does *not* include torch, so the pip install
#   fails at build. We install FeatUp at runtime instead, with
#   --no-build-isolation, after torch is already importable.
# --------------------------------------------------------
import os
import sys
import subprocess

# ZeroGPU requirement: `spaces` must be imported BEFORE any CUDA-initializing
# package (torch, etc.). Importing torch first and then spaces raises
# "CUDA has been initialized before importing the `spaces` package."
# This is the first import that touches CUDA indirectly, so do it up front.
# Guarded with try/except so local dev (where `spaces` isn't installed) works.
try:
    import spaces  # noqa: F401  (must be imported before torch on ZeroGPU)
except ImportError:
    pass


def _ensure_featup():
    try:
        import featup  # noqa: F401
    except ImportError:
        # FeatUp compiles CUDA extensions during install. On ZeroGPU, no GPU is
        # attached at app-startup (the GPU only becomes visible inside @spaces.GPU
        # decorated functions), so torch can't auto-detect a CUDA arch and
        # _get_cuda_arch_flags() raises IndexError. Give it an explicit list
        # covering every GPU Spaces might allocate: T4 (7.5), A10G (8.6),
        # A100 (8.0), L4 (8.9), H100/H200 (9.0).
        env = os.environ.copy()
        env.setdefault("TORCH_CUDA_ARCH_LIST", "7.5;8.0;8.6;8.9;9.0")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--no-build-isolation",
            "git+https://github.com/mhamilton723/FeatUp",
        ], env=env)
    _ensure_clip_bpe_vocab()


def _ensure_clip_bpe_vocab():
    """FeatUp's maskclip tokenizer expects `bpe_simple_vocab_16e6.txt.gz` to
    sit next to `simple_tokenizer.py`, but the file isn't shipped in the
    FeatUp package. Download it from OpenAI's CLIP repo if missing.

    We must NOT `import featup.featurizers.maskclip` to find the path — that
    package's `__init__.py` runs the tokenizer constructor which will raise
    the very FileNotFoundError we're trying to prevent. Instead import just
    the top-level `featup` package and derive the maskclip directory.
    """
    import urllib.request
    import featup  # top-level only, does not trigger maskclip chain
    maskclip_dir = os.path.join(
        os.path.dirname(featup.__file__), "featurizers", "maskclip"
    )
    bpe_path = os.path.join(maskclip_dir, "bpe_simple_vocab_16e6.txt.gz")
    if os.path.isfile(bpe_path):
        return
    url = "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"
    print(f"[sab3r] Downloading CLIP BPE vocab to {bpe_path}")
    os.makedirs(maskclip_dir, exist_ok=True)
    urllib.request.urlretrieve(url, bpe_path)


_ensure_featup()

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from demo import main  # noqa: E402  (import deferred until after path/featup setup)

if __name__ == "__main__":
    main([
        "--model_name", "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
        "--server_name", "0.0.0.0",
        "--server_port", os.environ.get("GRADIO_SERVER_PORT", "7860"),
    ])
