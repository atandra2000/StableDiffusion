"""
STEP 4 — PREPROCESS SHARDS TO HYBRID CACHE (image_key only)
══════════════════════════════════════════════════════════════

Stores: image_key, caption, input_ids, attention_mask
Images remain in the original .tar shards and are decoded on-the-fly during training.

This is the recommended production setup (same as LAION pipelines).
Disk usage: ~8–12 GB cache + original shards (~1.8 TB for full LAION).
"""

import os
import io
import sys
import json
import tarfile
import logging
import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import CLIPTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("/workspace/StableDiffusion/laion_logs/04_preprocess.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ─── Tokenizer loaded once at module level ───────────────────────────────────
logger.info("Loading CLIP tokenizer once at startup...")
TOKENIZER = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
logger.info("Tokenizer loaded successfully.")


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess LAION shards → hybrid cache (image_key only)")
    parser.add_argument("--shard_dir",    type=str, default="/workspace/StableDiffusion/laion_shards")
    parser.add_argument("--output_dir",   type=str, default="/workspace/StableDiffusion/laion_cache")
    parser.add_argument("--workers",      type=int, default=4)          # safe default
    parser.add_argument("--batch_size",   type=int, default=5000)
    parser.add_argument("--max_samples",  type=int, default=None)
    parser.add_argument("--image_size",   type=int, default=512)
    parser.add_argument("--resume",       action="store_true", default=True)
    parser.add_argument("--validate_images", action="store_true", default=True)
    return parser.parse_args()


# ─── Image Processing ────────────────────────────────────────────────────────
def center_crop_and_resize(img: Image.Image, target_size: int = 512) -> Image.Image:
    w, h = img.size
    if w < h:
        new_w = target_size
        new_h = int(h * target_size / w)
    else:
        new_h = target_size
        new_w = int(w * target_size / h)

    resample = Image.LANCZOS if (new_w <= w and new_h <= h) else Image.BICUBIC
    img = img.resize((new_w, new_h), resample=resample)

    left = (new_w - target_size) // 2
    top  = (new_h - target_size) // 2
    img = img.crop((left, top, left + target_size, top + target_size))
    return img


def is_valid_image(img: Image.Image, target_size: int = 512) -> tuple:
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.width < target_size // 2 or img.height < target_size // 2:
        return False, f"too_small_{img.width}x{img.height}"

    arr = np.asarray(img, dtype=np.float32)
    if not np.isfinite(arr).all():
        return False, "nan_or_inf"
    if arr.std() < 5.0:
        return False, "near_blank"
    if arr.mean() < 2.0:
        return False, "too_dark"
    if arr.mean() > 253.0:
        return False, "too_bright"
    return True, "ok"


def tokenize_caption(caption: str) -> dict:
    result = TOKENIZER(
        caption,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="np"
    )
    return {
        "input_ids":      result.input_ids[0].astype(np.int32),
        "attention_mask": result.attention_mask[0].astype(np.int32)
    }


# ─── Single Shard Processing ─────────────────────────────────────────────────
def process_single_shard(args_tuple):
    shard_path, target_size, validate_images = args_tuple
    shard_name = Path(shard_path).name
    logger.info(f"Worker started → {shard_name}")

    results = {
        "image_keys":     [],
        "captions":       [],
        "input_ids":      [],
        "attention_mask": [],
    }
    stats = {"total": 0, "valid": 0, "corrupt": 0, "invalid": 0, "no_caption": 0}

    try:
        with tarfile.open(shard_path, "r") as tf:
            member_index = {}
            for m in tf.getmembers():
                if not m.isfile(): continue
                base = m.name.rsplit(".", 1)[0]
                ext  = m.name.rsplit(".", 1)[-1].lower()
                member_index.setdefault(base, {})[ext] = m

            for base_name, files in member_index.items():
                if "jpg" not in files and "jpeg" not in files: continue
                if "txt" not in files and "json" not in files:
                    stats["no_caption"] += 1
                    continue

                stats["total"] += 1
                img_ext = "jpg" if "jpg" in files else "jpeg"

                try:
                    # Caption
                    if "txt" in files:
                        caption = tf.extractfile(files["txt"]).read().decode("utf-8", errors="ignore").strip()
                    else:
                        meta = json.loads(tf.extractfile(files["json"]).read())
                        caption = meta.get("caption", meta.get("TEXT", "")).strip()

                    if len(caption) < 10:
                        stats["no_caption"] += 1
                        continue

                    # Image (just for validation, we don't store it)
                    img_bytes = tf.extractfile(files[img_ext]).read()
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    img = center_crop_and_resize(img, target_size)

                    if validate_images:
                        valid, _ = is_valid_image(img, target_size)
                        if not valid:
                            stats["invalid"] += 1
                            continue

                    tokens = tokenize_caption(caption)

                    image_key = f"{shard_name}::{base_name}"
                    results["image_keys"].append(image_key)
                    results["captions"].append(caption)
                    results["input_ids"].append(tokens["input_ids"].tolist())
                    results["attention_mask"].append(tokens["attention_mask"].tolist())

                    stats["valid"] += 1

                    if stats["valid"] % 500 == 0 and stats["valid"] > 0:
                        logger.info(f"{shard_name} — {stats['valid']:,} valid samples")

                except Exception:
                    stats["corrupt"] += 1
                    continue

    except Exception as e:
        logger.error(f"Shard failed: {shard_path} → {e}")
        return {"error": str(e), "shard": shard_path, "results": results, "stats": stats}

    logger.info(f"Finished {shard_name} → {stats['valid']:,} valid")
    return {"results": results, "stats": stats, "shard": shard_path}


# ─── Write Batch ─────────────────────────────────────────────────────────────
def write_batch_to_parquet(batch: dict, output_dir: str, batch_idx: int):
    if not batch["image_keys"]:
        return 0

    arrays = {
        "image_key":      pa.array(batch["image_keys"],    type=pa.string()),
        "caption":        pa.array(batch["captions"],      type=pa.string()),
        "input_ids":      pa.array(batch["input_ids"],     type=pa.list_(pa.int32(), list_size=77)),
        "attention_mask": pa.array(batch["attention_mask"], type=pa.list_(pa.int32(), list_size=77)),
    }

    table = pa.table(arrays)
    out_path = os.path.join(output_dir, f"batch_{batch_idx:06d}.parquet")
    pq.write_table(table, out_path, compression="snappy")
    return len(batch["image_keys"])


# ─── Main Preprocessing Pipeline ─────────────────────────────────────────────
def preprocess_all_shards(args):
    shard_dir  = Path(args.shard_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_files = sorted(shard_dir.rglob("*.tar"))
    if not shard_files:
        logger.error("No .tar files found. Run step 3 first.")
        sys.exit(1)

    logger.info("═" * 60)
    logger.info(" Preprocessing LAION Shards → Hybrid Cache (image_key only)")
    logger.info("═" * 60)
    logger.info(f"Shards:       {len(shard_files)}")
    logger.info(f"Workers:      {args.workers}")
    logger.info(f"Max samples:  {args.max_samples or 'all'}")
    logger.info(f"Output:       {output_dir}")
    logger.info("═" * 60)

    # Resume support
    processed_log = output_dir / "processed_shards.txt"
    done_shards = set()
    if args.resume and processed_log.exists():
        done_shards = set(processed_log.read_text().splitlines())
        logger.info(f"Resuming — {len(done_shards)} shards already done")

    pending_shards = [s for s in shard_files if str(s) not in done_shards]
    logger.info(f"Pending shards: {len(pending_shards)}")

    if not pending_shards:
        logger.info("All shards already processed.")
        return

    # ── Batch accumulation ───────────────────────────────────────────────────
    current_batch = {k: [] for k in ["image_keys", "captions", "input_ids", "attention_mask"]}
    batch_idx = len(list(output_dir.glob("batch_*.parquet")))
    total_valid = batch_idx * args.batch_size
    total_corrupt = total_invalid = 0
    start_time = time.time()

    task_args = [(str(s), args.image_size, args.validate_images) for s in pending_shards]

    with ProcessPoolExecutor(max_workers=args.workers) as executor, \
         open(processed_log, "a") as done_file:

        futures = {executor.submit(process_single_shard, ta): ta[0] for ta in task_args}

        pbar = tqdm(as_completed(futures), total=len(futures), desc="Processing shards")

        for future in pbar:
            shard_path = futures[future]
            try:
                result = future.result(timeout=900)  # 15 min per shard

                if "error" in result:
                    logger.warning(f"Shard error: {result['shard']} → {result['error']}")

                stats = result.get("stats", {})
                samples = result.get("results", {})

                total_corrupt += stats.get("corrupt", 0)
                total_invalid += stats.get("invalid", 0)
                added = len(samples.get("image_keys", []))
                total_valid += added

                for k in current_batch:
                    if k in samples:
                        current_batch[k].extend(samples[k])

                done_file.write(f"{shard_path}\n")
                done_file.flush()

                # Write batch when full
                if len(current_batch["image_keys"]) >= args.batch_size:
                    n = write_batch_to_parquet(current_batch, str(output_dir), batch_idx)
                    logger.info(f"→ Wrote batch {batch_idx:06d} ({n} samples)")
                    batch_idx += 1
                    current_batch = {k: [] for k in current_batch}

                # Progress
                elapsed = time.time() - start_time
                rate = total_valid / elapsed if elapsed > 0 else 0
                pbar.set_postfix(valid=f"{total_valid:,}", corrupt=total_corrupt, rate=f"{rate:.1f}/s")

                if args.max_samples and total_valid >= args.max_samples:
                    logger.info(f"Reached max_samples = {args.max_samples}")
                    for f in futures:
                        if not f.done():
                            f.cancel()
                    break

            except Exception as e:
                logger.warning(f"Future failed {shard_path}: {e}")

        # Final partial batch
        if current_batch["image_keys"]:
            write_batch_to_parquet(current_batch, str(output_dir), batch_idx)

    # ── Summary ──────────────────────────────────────────────────────────────
    all_batches = list(output_dir.glob("batch_*.parquet"))
    total_samples = sum(pq.read_metadata(str(bp)).num_rows for bp in all_batches)

    logger.info("\n" + "═" * 60)
    logger.info("Preprocessing Complete")
    logger.info("═" * 60)
    logger.info(f"Total valid samples: {total_samples:,}")
    logger.info(f"Parquet batches:     {len(all_batches)}")
    logger.info(f"Output directory:    {output_dir}")
    logger.info("═" * 60)
    logger.info("\nNext: python3 05_build_hf_dataset.py")


if __name__ == "__main__":
    args = parse_args()
    preprocess_all_shards(args)
