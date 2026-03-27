"""
STEP 5 — BUILD HUGGINGFACE DATASET (Hybrid Mode)
════════════════════════════════════════════════

Merges all parquet batches from Step 4 into a HuggingFace Dataset.

This version is **hybrid only**:
- Stores: image_key, caption, input_ids, attention_mask
- Images are loaded from the original .tar shards at training time (no pixel tensors stored)
- Disk usage: ~8–15 GB for the dataset + original shards

This is the recommended, production-grade setup.
"""

import os
import sys
import glob
import logging
import argparse
import time
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import Dataset, DatasetDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("/workspace/StableDiffusion/laion_logs/05_build_dataset.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Build HuggingFace Dataset (hybrid mode)")
    parser.add_argument("--input_dir",  type=str, default="/workspace/StableDiffusion/laion_cache")
    parser.add_argument("--output_dir", type=str, default="/workspace/StableDiffusion/laion_hf_dataset")
    parser.add_argument("--shard_dir",  type=str, default="/workspace/StableDiffusion/laion_shards")
    parser.add_argument("--val_size",   type=int, default=5000)
    parser.add_argument("--shuffle",    action="store_true", default=True)
    parser.add_argument("--seed",       type=int, default=42)
    return parser.parse_args()


def collect_parquet_batches(input_dir: str) -> list:
    batch_files = sorted(Path(input_dir).glob("batch_*.parquet"))
    if not batch_files:
        logger.error(f"❌ No batch_*.parquet files found in {input_dir}")
        logger.error("   Run Step 4 first: python3 04_preprocess_to_cache.py")
        sys.exit(1)
    logger.info(f"Found {len(batch_files)} parquet batches")
    return batch_files


def count_total_samples(batch_files: list) -> int:
    total = 0
    for bf in batch_files:
        try:
            total += pq.read_metadata(str(bf)).num_rows
        except Exception:
            pass
    return total


def build_hf_dataset(args):
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_files = collect_parquet_batches(str(input_dir))
    total = count_total_samples(batch_files)

    logger.info("═" * 60)
    logger.info(" Building HuggingFace Dataset (Hybrid Mode)")
    logger.info("═" * 60)
    logger.info(f"Batch files:     {len(batch_files)}")
    logger.info(f"Total samples:   {total:,}")
    logger.info(f"Validation size: {args.val_size:,}")
    logger.info(f"Output:          {output_dir}")
    logger.info("═" * 60)

    # ── Load all batches ──────────────────────────────────────────────────────
    logger.info("📥 Loading parquet batches...")
    chunks = []
    for bf in tqdm(batch_files, desc="Loading batches"):
        try:
            df = pd.read_parquet(bf)
            chunks.append(df)
        except Exception as e:
            logger.warning(f"⚠️ Skipping corrupt file {bf.name}: {e}")

    df_full = pd.concat(chunks, ignore_index=True)
    logger.info(f"Loaded {len(df_full):,} total rows")

    # ── Shuffle ───────────────────────────────────────────────────────────────
    if args.shuffle:
        logger.info("🔀 Shuffling dataset...")
        df_full = df_full.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # ── Convert list columns ──────────────────────────────────────────────────
    logger.info("🔧 Converting data types...")

    def convert_list_col(series, dtype=np.int32, expected_len=77):
        return [
            np.array(v, dtype=dtype)[:expected_len] if isinstance(v, (list, np.ndarray))
            else np.zeros(expected_len, dtype=dtype)
            for v in series
        ]

    df_full["input_ids"]      = convert_list_col(df_full["input_ids"])
    df_full["attention_mask"] = convert_list_col(df_full["attention_mask"])

    # ── Train / Val split ─────────────────────────────────────────────────────
    logger.info(f"✂️  Creating train/val split ({args.val_size} validation samples)...")
    n = len(df_full)
    val_size = min(args.val_size, n // 10)
    val_idx   = np.arange(val_size)
    train_idx = np.arange(val_size, n)

    df_train = df_full.iloc[train_idx].reset_index(drop=True)
    df_val   = df_full.iloc[val_idx].reset_index(drop=True)

    logger.info(f"   Train: {len(df_train):,}  |  Val: {len(df_val):,}")

    # ── Save as HuggingFace Dataset ───────────────────────────────────────────
    logger.info("💾 Saving HuggingFace Dataset to disk...")

    train_ds = Dataset.from_pandas(df_train)
    val_ds   = Dataset.from_pandas(df_val)

    train_dir = output_dir / "train"
    val_dir   = output_dir / "val"

    train_ds.save_to_disk(str(train_dir))
    val_ds.save_to_disk(str(val_dir))

    logger.info(f"   ✅ Train split saved: {train_dir}")
    logger.info(f"   ✅ Val split saved:   {val_dir}")

    # ── Save dataset config ───────────────────────────────────────────────────
    config = {
        "mode":           "standard",           # hybrid / image_key mode
        "train_size":     len(df_train),
        "val_size":       len(df_val),
        "shard_dir":      str(args.shard_dir),
        "image_size":     512,
        "columns":        list(df_train.columns),
        "seed":           args.seed,
        "created":        time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    config_path = output_dir / "dataset_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    logger.info(f"   ✅ Config saved: {config_path}")

    # ── Final report ──────────────────────────────────────────────────────────
    cache_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file()) / 1e9
    logger.info("\n" + "═" * 60)
    logger.info(" Dataset Build Complete (Hybrid Mode)")
    logger.info("═" * 60)
    logger.info(f" Train samples:   {len(df_train):,}")
    logger.info(f" Val samples:     {len(df_val):,}")
    logger.info(f" Dataset size:    {cache_size:.1f} GB")
    logger.info(f" Output:          {output_dir}")
    logger.info("═" * 60)
    logger.info("\n✅ Next step: python3 06_verify_dataset.py")

    return str(train_dir), str(val_dir)


if __name__ == "__main__":
    args = parse_args()
    build_hf_dataset(args)