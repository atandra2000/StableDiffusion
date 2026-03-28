"""
STEP 2 — FILTER METADATA
═════════════════════════

Reads every parquet file downloaded in Step 1 and applies quality filters.
Output: a clean set of parquet files containing only the URLs+captions
we actually want to download — NO images involved yet.

WHY FILTER BEFORE DOWNLOADING:
  Downloading 12M images takes 12-48 hours. We don't want to waste that time
  on images that will be discarded during training anyway.
  Filtering first means we only download exactly what we'll use.

FILTER CRITERIA EXPLAINED:

  aesthetic_score >= 6.5:
    LAION's aesthetic predictor (trained on human preferences) assigns each
    image a score 0-10. Score >= 6.5 selects the top ~2% of LAION-2B-en —
    roughly 12M of the highest quality, visually pleasing images.
    Score 6.0 would give ~600M images (the "large dataset" option).

  similarity >= 0.28:
    CLIP image-text cosine similarity. Ensures the caption actually describes
    the image content. Low similarity = wrong captions (e.g., alt text that
    says "click here" or product SKU numbers).

  width >= 512, height >= 512:
    We train at 512×512. Upscaling smaller images introduces blur and artifacts
    that hurt training — the model learns to generate soft/blurry outputs.

  aspect ratio 0.5 to 2.0:
    Extreme panoramas or tall portraits crop very poorly to square.
    0.5 = 1:2 (portrait), 2.0 = 2:1 (landscape).

  caption length 20-300 chars:
    Too short = uninformative ("dog", "photo"). Too long = truncated at 77 tokens.
    The CLIP tokenizer processes max 77 tokens (~60-80 words).

  No watermark / NSFW:
    watermark_score < 0.5: LAION includes a watermark probability prediction.
    NSFW: removes adult content to keep training clean.

Usage:
  python3 02_filter_metadata.py
  python3 02_filter_metadata.py --aesthetic_min 6.0  # 600M images
  python3 02_filter_metadata.py --max_samples 5000000  # Cap at 5M
"""

import sys
import logging
import argparse
import time
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("/workspace/StableDiffusion/laion_logs/02_filter_metadata.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Filter LAION metadata for quality")
    parser.add_argument("--input_dir",     type=str,
                        default="/workspace/StableDiffusion/laion_metadata")
    parser.add_argument("--output_dir",    type=str,
                        default="/workspace/StableDiffusion/laion_filtered")
    parser.add_argument("--aesthetic_min", type=float, default=6.5,
                        help="Minimum aesthetic score (6.5=12M, 6.0=600M)")
    parser.add_argument("--clip_sim_min",  type=float, default=0.28,
                        help="Minimum CLIP image-text similarity")
    parser.add_argument("--min_width",     type=int,   default=512)
    parser.add_argument("--min_height",    type=int,   default=512)
    parser.add_argument("--min_aspect",    type=float, default=0.5,
                        help="Minimum aspect ratio (width/height)")
    parser.add_argument("--max_aspect",    type=float, default=2.0,
                        help="Maximum aspect ratio (width/height)")
    parser.add_argument("--min_caption",   type=int,   default=20,
                        help="Minimum caption character length")
    parser.add_argument("--max_caption",   type=int,   default=300,
                        help="Maximum caption character length")
    parser.add_argument("--max_watermark", type=float, default=0.5,
                        help="Maximum watermark probability (0-1)")
    parser.add_argument("--allow_nsfw",    action="store_true", default=False)
    parser.add_argument("--max_samples",   type=int,   default=None,
                        help="Stop after keeping this many samples (for dataset size capping)")
    parser.add_argument("--shuffle_seed",  type=int,   default=42,
                        help="Seed for shuffling files before processing (ensures random sample if max_samples set)")
    return parser.parse_args()


def inspect_parquet_schema(parquet_file: str) -> list:
    """
    Inspect column names of a parquet file to handle LAION's varying schemas.
    LAION files may have slightly different column names depending on the subset.
    """
    df = pd.read_parquet(parquet_file, columns=None).head(0)
    return list(df.columns)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names across different LAION parquet schemas.

    Known variations:
      URL/url → URL
      TEXT/text/caption → TEXT
      WIDTH/width → WIDTH
      HEIGHT/height → HEIGHT
      similarity/SIMILARITY → similarity
      aesthetic_score/AESTHETIC_SCORE → aesthetic_score
      punsafe/NSFW/nsfw → NSFW
      pwatermark/watermark → pwatermark
    """
    rename_map = {}
    col_lower = {c.lower(): c for c in df.columns}

    for target, variations in [
        ("URL",             ["url", "URL"]),
        ("TEXT",            ["text", "TEXT", "caption", "CAPTION"]),
        ("WIDTH",           ["width", "WIDTH"]),
        ("HEIGHT",          ["height", "HEIGHT"]),
        ("similarity",      ["similarity", "SIMILARITY", "clip_sim"]),
        ("aesthetic_score", ["aesthetic_score", "AESTHETIC_SCORE", "aesthetic"]),
        ("pwatermark",      ["pwatermark", "watermark", "WATERMARK", "watermark_score"]),
        ("NSFW",            ["nsfw", "NSFW", "punsafe", "is_nsfw"]),
    ]:
        for var in variations:
            if var in df.columns and var != target:
                rename_map[var] = target
            elif var.lower() in col_lower and col_lower[var.lower()] not in rename_map:
                original = col_lower[var.lower()]
                if original != target:
                    rename_map[original] = target

    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def apply_filters(df: pd.DataFrame, args) -> pd.DataFrame:
    """
    Apply all quality filters to a dataframe.
    Returns the filtered subset.
    """
    df = normalize_column_names(df)
    original_len = len(df)
    filter_stats = {}

    # 1. Require essential columns
    if "URL" not in df.columns or "TEXT" not in df.columns:
        logger.warning("   Missing URL or TEXT column — skipping file")
        return pd.DataFrame()

    # 2. Drop null rows
    df = df.dropna(subset=["URL", "TEXT"])
    filter_stats["null_drop"] = original_len - len(df)

    # 3. Aesthetic score
    if "aesthetic_score" in df.columns:
        before = len(df)
        df = df[df["aesthetic_score"] >= args.aesthetic_min]
        filter_stats["aesthetic"] = before - len(df)

    # 4. CLIP similarity
    if "similarity" in df.columns:
        before = len(df)
        df = df[df["similarity"] >= args.clip_sim_min]
        filter_stats["clip_sim"] = before - len(df)

    # 5. Image dimensions
    if "WIDTH" in df.columns and "HEIGHT" in df.columns:
        before = len(df)
        df = df[
            (df["WIDTH"] >= args.min_width) &
            (df["HEIGHT"] >= args.min_height)
        ]
        filter_stats["dimensions"] = before - len(df)

        # 6. Aspect ratio
        before = len(df)
        aspect = df["WIDTH"] / df["HEIGHT"]
        df = df[(aspect >= args.min_aspect) & (aspect <= args.max_aspect)]
        filter_stats["aspect"] = before - len(df)

    # 7. Caption length
    before = len(df)
    caption_len = df["TEXT"].str.len()
    df = df[
        (caption_len >= args.min_caption) &
        (caption_len <= args.max_caption)
    ]
    filter_stats["caption_len"] = before - len(df)

    # 8. Watermark filter
    if "pwatermark" in df.columns:
        before = len(df)
        df = df[df["pwatermark"] < args.max_watermark]
        filter_stats["watermark"] = before - len(df)

    # 9. NSFW filter
    if not args.allow_nsfw and "NSFW" in df.columns:
        before = len(df)
        # Values can be: "UNLIKELY", "POSSIBLE", "LIKELY", or numeric probability
        if df["NSFW"].dtype == object:
            df = df[df["NSFW"].isin(["UNLIKELY", "unlikely"])]
        else:
            df = df[df["NSFW"] < 0.2]
        filter_stats["nsfw"] = before - len(df)

    # 10. Remove duplicate URLs
    before = len(df)
    df = df.drop_duplicates(subset=["URL"])
    filter_stats["dedup"] = before - len(df)

    return df[["URL", "TEXT", "aesthetic_score", "similarity"]].copy()


def filter_all_metadata(args):
    """Main filtering pipeline across all parquet files."""
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(input_dir.glob("*.parquet"))
    if not parquet_files:
        # Some repos have subdirectories
        parquet_files = sorted(input_dir.rglob("*.parquet"))

    if not parquet_files:
        logger.error(f"❌ No parquet files found in {input_dir}")
        logger.error("   Run Step 1 first: python3 01_download_metadata.py")
        sys.exit(1)

    logger.info("═" * 60)
    logger.info(" LAION-Aesthetics Metadata Filtering")
    logger.info("═" * 60)
    logger.info(f" Input:           {input_dir}")
    logger.info(f" Output:          {output_dir}")
    logger.info(f" Parquet files:   {len(parquet_files)}")
    logger.info(f" aesthetic_min:   {args.aesthetic_min}")
    logger.info(f" clip_sim_min:    {args.clip_sim_min}")
    logger.info(f" dimensions:      >= {args.min_width}×{args.min_height}")
    logger.info(f" aspect ratio:    {args.min_aspect} - {args.max_aspect}")
    logger.info(f" caption chars:   {args.min_caption} - {args.max_caption}")
    logger.info(f" max watermark:   {args.max_watermark}")
    logger.info(f" allow_nsfw:      {args.allow_nsfw}")
    logger.info(f" max_samples:     {args.max_samples or 'unlimited'}")
    logger.info("═" * 60)

    # Inspect schema of first file to report columns
    try:
        first_cols = inspect_parquet_schema(str(parquet_files[0]))
        logger.info(f"\n📋 Detected columns: {first_cols}\n")
    except Exception as e:
        logger.warning(f"Could not inspect schema: {e}")

    # Shuffle files for random sampling (important if max_samples is set)
    np.random.seed(args.shuffle_seed)
    file_list = list(parquet_files)
    np.random.shuffle(file_list)

    total_seen  = 0
    total_kept  = 0
    file_count  = 0
    start_time  = time.time()

    for pq_file in tqdm(file_list, desc="Filtering parquet files"):

        # Check if already filtered (resume support)
        out_name = output_dir / pq_file.name
        if out_name.exists() and out_name.stat().st_size > 100:
            # Count rows to update our global counter
            try:
                n = len(pd.read_parquet(out_name))
                total_kept += n
                file_count += 1
                if args.max_samples and total_kept >= args.max_samples:
                    logger.info(f"   Reached max_samples={args.max_samples}. Stopping.")
                    break
                continue
            except Exception:
                pass  # Re-process if can't read

        try:
            df_raw = pd.read_parquet(pq_file)
            total_seen += len(df_raw)

            df_filtered = apply_filters(df_raw, args)
            kept = len(df_filtered)

            if kept > 0:
                # If max_samples: only keep as many as needed to reach the cap
                if args.max_samples and (total_kept + kept) > args.max_samples:
                    need = args.max_samples - total_kept
                    df_filtered = df_filtered.head(need)
                    kept = len(df_filtered)

                df_filtered.to_parquet(str(out_name), index=False, compression="snappy")
                total_kept += kept
                file_count += 1

            if total_seen % 5_000_000 == 0 and total_seen > 0:
                rate = total_kept / total_seen * 100
                elapsed = time.time() - start_time
                logger.info(f"   Progress: {total_seen/1e6:.1f}M seen | {total_kept/1e6:.2f}M kept ({rate:.1f}%) | {elapsed/60:.1f}min")

            if args.max_samples and total_kept >= args.max_samples:
                logger.info(f"   ✅ Reached max_samples={args.max_samples}. Stopping.")
                break

        except Exception as e:
            logger.warning(f"   ⚠️ Error in {pq_file.name}: {e}")
            continue

    elapsed = time.time() - start_time
    keep_rate = total_kept / max(total_seen, 1) * 100

    logger.info("\n" + "═" * 60)
    logger.info(" Filtering Complete")
    logger.info("═" * 60)
    logger.info(f" Total seen:   {total_seen:>12,}")
    logger.info(f" Total kept:   {total_kept:>12,}  ({keep_rate:.2f}%)")
    logger.info(f" Output files: {file_count}")
    logger.info(f" Time elapsed: {elapsed/60:.1f} minutes")

    # Disk usage of filtered metadata
    out_size_mb = sum(f.stat().st_size for f in output_dir.glob("*.parquet")) / 1e6
    logger.info(f" Filtered metadata size: {out_size_mb:.1f} MB")
    logger.info("═" * 60)
    logger.info("\n✅ Next step: bash 03_download_images.sh")


if __name__ == "__main__":
    args = parse_args()
    filter_all_metadata(args)
