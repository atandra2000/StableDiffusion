"""
Download the released checkpoint from Hugging Face Hub.

Usage:
    python scripts/download_checkpoint.py
    python scripts/download_checkpoint.py --output checkpoints/sd_epoch_042.pt
"""

import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, login
except ImportError:
    print("Error: huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

REPO_ID = "atandra2000/sd-from-scratch-v1"
FILENAME = "sd_epoch_042.pt"

def parse_args():
    p = argparse.ArgumentParser(description="Download SD-From-Scratch checkpoint")
    p.add_argument("--output", type=str, default="checkpoints/sd_epoch_042.pt",
                   help="Output path for the checkpoint")
    p.add_argument("--token", type=str, default=None,
                   help="HF Hub token (optional for public repos)")
    return p.parse_args()

def main():
    args = parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        size_gb = output.stat().st_size / (1024**3)
        print(f"Already exists: {output} ({size_gb:.1f} GB)")
        return

    print(f"Downloading {FILENAME} from {REPO_ID}...")
    print(f"Size: ~12.5 GB — this will take a while.")

    if args.token:
        login(token=args.token)

    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=output.parent,
        local_dir_use_symlinks=False,
    )

    size_gb = Path(path).stat().st_size / (1024**3)
    print(f"Downloaded: {path} ({size_gb:.1f} GB)")

if __name__ == "__main__":
    main()
