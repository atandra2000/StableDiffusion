"""
06_filter_dataset.py — Filter HF dataset and create matching latent subset.

Removes celebrity names, garbage prompts, and NSFW content from the
hf_dataset/train Arrow dataset, then creates a hardlinked latent directory
containing only the latent files whose image_keys survived filtering.

Usage:
    python3 data_pipeline/06_filter_dataset.py \
        --dataset_path /workspace/StableDiffusion/hf_dataset/train \
        --latent_dir   /workspace/StableDiffusion/latents/latents \
        --out_dataset  /workspace/StableDiffusion/hf_dataset_filtered/train \
        --out_latents  /workspace/StableDiffusion/latents_filtered/latents

Phase 1 result (705,890 raw samples):
    After filtering: 482,552 samples (68.4% kept), 44.8% face/human prompts.
"""

import argparse
import os
from pathlib import Path

from datasets import load_from_disk


# ── Filter rules ──────────────────────────────────────────────────────────────

CELEBRITIES = [
    "stephen colbert", "mark zuckerberg", "elon musk", "joe biden",
    "donald trump", "putin", "obama", "kanye", "taylor swift",
    "kim kardashian", "jeff bezos", "bill gates", "steve jobs",
    "lebron james", "cristiano ronaldo", "messi",
]

NSFW_TERMS = [
    "nude", "naked", "nsfw", "explicit", "xxx", "erotic",
    "porn", "topless", "lingerie", "underwear",
]


def keep(caption: str) -> bool:
    c = caption.lower().strip()
    if len(c) < 12:
        return False
    if len(c.split()) < 3:
        return False
    if any(name in c for name in CELEBRITIES):
        return False
    if any(w in c for w in NSFW_TERMS):
        return False
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--latent_dir",   required=True)
    parser.add_argument("--out_dataset",  required=True)
    parser.add_argument("--out_latents",  required=True)
    parser.add_argument("--workers",      type=int, default=8)
    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset_path}...")
    ds = load_from_disk(args.dataset_path)
    print(f"Total samples: {len(ds):,}")

    print("Filtering...")
    filtered = ds.filter(
        lambda batch: [keep(c) for c in batch["caption"]],
        batched=True,
        batch_size=10_000,
        num_proc=args.workers,
        desc="Filtering",
    )
    kept_pct = len(filtered) / len(ds) * 100
    print(f"After filter: {len(filtered):,} ({kept_pct:.1f}% kept)")

    face_keywords = [
        "portrait", "face", "woman", "man", "girl", "boy",
        "person", "human", "character", "anime", "figure", "lady", "guy",
    ]
    captions = filtered["caption"]
    face_count = sum(any(k in c.lower() for k in face_keywords) for c in captions)
    print(f"Face/human prompts: {face_count:,} ({face_count/len(filtered)*100:.1f}%)")

    print(f"Saving filtered dataset to {args.out_dataset}...")
    filtered.save_to_disk(args.out_dataset)
    print("Dataset saved.")

    # ── Hardlink latents matching filtered keys ────────────────────────────────
    latent_dir = Path(args.latent_dir)
    out_dir    = Path(args.out_latents)
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_keys = set(filtered["image_key"])
    print(f"\nFiltering latents: {len(valid_keys):,} valid keys")

    all_npy = list(latent_dir.glob("*.npy"))
    print(f"Total latent files: {len(all_npy):,}")

    kept = skipped = 0
    for i, f in enumerate(all_npy):
        dest = out_dir / f.name
        if dest.exists():
            kept += 1
            continue
        if f.stem in valid_keys:
            os.link(f, dest)
            kept += 1
        else:
            skipped += 1
        if (i + 1) % 50_000 == 0:
            print(f"  {i+1:,} processed | kept {kept:,} | skipped {skipped:,}")

    print(f"\nDone: {kept:,} latents kept, {skipped:,} removed")
    print(f"Output latents: {out_dir}")


if __name__ == "__main__":
    main()
