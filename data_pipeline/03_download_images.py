"""
STEP 3 — DOWNLOAD IMAGES WITH img2dataset
Simplified and stabilized version.
"""

import os
import sys
import json
import time
import argparse
import logging
import subprocess
from pathlib import Path
import shutil

# ---------- Logging ----------

LOG_DIR = "/workspace/StableDiffusion/laion_logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/03_download_images.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ---------- Args ----------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",
        default="/workspace/StableDiffusion/laion_filtered")
    parser.add_argument("--output_dir",
        default="/workspace/StableDiffusion/laion_shards")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--processes", type=int, default=16)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--shard_size", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--encode_quality", type=int, default=95)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()

# ---------- Helpers ----------

def count_existing_shards(output_dir):
    shard_files = list(Path(output_dir).rglob("*.tar"))
    total_images = 0

    for sf in shard_files:
        stats_file = sf.with_suffix(".json")
        if stats_file.exists():
            try:
                stats = json.loads(stats_file.read_text())
                total_images += stats.get("successes", 0)
            except Exception:
                pass

    return len(shard_files), total_images


def check_disk_space(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    free_gb = shutil.disk_usage(path).free / 1e9
    logger.info(f"💾 Free disk space: {free_gb:.1f} GB")

    if free_gb < 50:
        logger.warning("⚠️ Disk space looks low.")


def build_command(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    return [
        "img2dataset",
        "--url_list", args.input_dir,
        "--input_format", "parquet",
        "--url_col", "URL",
        "--caption_col", "TEXT",
        "--output_format", "webdataset",
        "--output_folder", args.output_dir,
        "--processes_count", str(args.processes),
        "--thread_count", str(args.threads),
        "--image_size", str(args.image_size),
        "--resize_mode", "center_crop",
        "--resize_only_if_bigger", "True",
        "--encode_format", "jpg",
        "--encode_quality", str(args.encode_quality),
        "--number_sample_per_shard", str(args.shard_size),
        "--timeout", str(args.timeout),
        "--retries", str(args.retries),
        "--incremental_mode", "incremental",
        "--distributor", "multiprocessing",
    ]


# ---------- Main ----------

def run_download(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not list(input_dir.glob("*.parquet")):
        logger.error("❌ No parquet files found.")
        sys.exit(1)

    check_disk_space(output_dir)

    shards, images = count_existing_shards(output_dir)
    if shards:
        logger.info(f"ℹ️ Resuming: {shards} shards ({images:,} images already downloaded)")

    cmd = build_command(args)

    logger.info("\n🚀 Running img2dataset:\n")
    logger.info(" ".join(cmd))

    if args.dry_run:
        logger.info("Dry run only.")
        return

    start = time.time()

    try:
        subprocess.run(cmd, check=True)
        success = True
    except subprocess.CalledProcessError:
        logger.error("Download failed.")
        success = False
    except KeyboardInterrupt:
        logger.info("Interrupted — resume supported.")
        success = False

    elapsed = time.time() - start
    shards, images = count_existing_shards(output_dir)

    logger.info("\n══════════════════════════════")
    logger.info("Download complete" if success else "Download interrupted")
    logger.info(f"Shards: {shards}")
    logger.info(f"Images: {images:,}")
    logger.info(f"Time: {elapsed/3600:.2f} hours")
    logger.info(f"Output: {output_dir}")
    logger.info("══════════════════════════════")


if __name__ == "__main__":
    args = parse_args()
    run_download(args)
