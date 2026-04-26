"""
encode_latents.py
=================
Pre-encodes a LAION-style tar-shard dataset into VAE latents for fast training.
Optimised for 2× RTX 5090 with a network-mounted (RunPod MFS) shard directory.

Disk-space strategy
-------------------
Each float16 latent (4, 64, 64) is 32 KB.  One `.npy` file is written per
image directly to OUT_DIR:

    Storage = N_images × 32 KB
    1.5 M images → ~48 GB

Resume scans OUT_DIR with os.listdir() — O(1) lookup, no file reads needed.

Pipeline — three concurrent stages
-----------------------------------
Stage 1  Shard prefetch thread
    Copies tar files from the network mount → local NVMe (/tmp) ahead of the
    extractor.

Stage 2  Extractor + JPEG submit thread
    Opens each local tar, reads raw JPEG bytes, fans decode tasks to the pool.

Stage 3  Decode thread pool  (DECODE_WORKERS threads)
    Each worker runs decode_and_crop() on one JPEG: decode → resize →
    centre-crop → normalise to [-1, 1].  cv2 is ~3× faster than PIL.

Stage 4  GPU main thread
    Resolves decode futures → copies to GPU via pinned memory (DMA-direct) →
    runs VAE.encode() under bfloat16 autocast → np.saves float16 latents.

GPU idle time is minimised by:
    - Pinned-memory staging buffer  → page-locked H2D copy via DMA
    - Dedicated H2D CUDA stream     → transfer overlaps CPU future resolution
    - np.save at 32 KB/file         → negligible I/O cost, no async queue needed

Output format
-------------
One  .npy  file per image, stored in OUT_DIR.
Each file is a raw numpy float16 array of shape (4, 64, 64).

Usage
-----
    python encode_latents.py --shard_dir /data/laion_shards --out_dir /data/latents
    python encode_latents.py --batch_size 48 --decode_workers 24
"""

import argparse
import logging
import os
import shutil
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby
from pathlib import Path
from queue import Queue
from threading import Thread

import numpy as np
import torch
from tqdm import tqdm
import cv2 as _cv2

sys.path.insert(0, "/workspace/StableDiffusion")

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ── Default config (overridable via CLI) ──────────────────────────────────────
DATASET_PATH   = "/workspace/StableDiffusion/laion_hf_dataset/train"
SHARD_DIR      = Path("/workspace/StableDiffusion/laion_shards")
OUT_DIR        = Path("/workspace/StableDiffusion/laion_latents")
LOCAL_TMP      = Path("/tmp/shard_cache")    # local NVMe for tar staging

BATCH_SIZE     = 32      # images per VAE encode call (~8 GB VRAM @ bfloat16 on 5090)
TARGET_SIZE    = 512     # resize/centre-crop target resolution
RESUME         = True    # skip images whose .npy file already exists in OUT_DIR

PREFETCH_Q     = 64      # max decoded batches queued ahead of the GPU
DECODE_WORKERS = 16      # parallel JPEG decode threads (CPU-bound)
SHARD_PREFETCH = 3       # tar shards to keep pre-copied on local NVMe

# ── Sentinel objects for queue termination ────────────────────────────────────
_SHARD_SENTINEL = object()
_BATCH_SENTINEL = object()


# ═══════════════════════════════════════════════════════════════════════════════
# KEY / FILENAME HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def key_to_filename(k: str) -> str:
    """Convert a dataset image key to its .npy filename. Must match the identical function in SD_Train.py."""
    return k.replace("/", "_").replace("::", "__") + ".npy"


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def decode_and_crop(jpeg_bytes: bytes, target: int = TARGET_SIZE):
    """
    Decode a JPEG byte string, resize (preserving aspect ratio), and
    centre-crop to *target* × *target* pixels.  Normalise to [-1, 1].
    Returns a (3, H, W) float32 numpy array, or None on failure.
    """
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = _cv2.cvtColor(img, _cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if w < h:
        new_w, new_h = target, int(h * target / w)
    else:
        new_h, new_w = target, int(w * target / h)
    interp = _cv2.INTER_AREA if (new_w <= w and new_h <= h) else _cv2.INTER_CUBIC
    img = _cv2.resize(img, (new_w, new_h), interpolation=interp)
    x = (new_w - target) // 2
    y = (new_h - target) // 2
    img = img[y : y + target, x : x + target]
    return (img.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)


def read_all_keys(dataset_path: str) -> list:
    """
    Load every image_key from the HuggingFace Arrow dataset at *dataset_path*.
    Tries pyarrow's low-level IPC reader first; falls back to load_from_disk.
    """
    try:
        import pyarrow as pa
        arrow_files = sorted(Path(dataset_path).glob("data-*.arrow"))
        if not arrow_files:
            raise FileNotFoundError("No data-*.arrow files found")
        logger.info(f"Reading {len(arrow_files)} Arrow file(s) via pyarrow...")
        all_keys = []
        for f in arrow_files:
            raw = open(f, "rb").read()
            for opener in [pa.ipc.open_stream, pa.ipc.open_file]:
                try:
                    reader = opener(pa.BufferReader(raw))
                    batches = (
                        reader if hasattr(reader, "__iter__")
                        else [reader.get_batch(i) for i in range(reader.num_record_batches)]
                    )
                    for b in batches:
                        if "image_key" in b.schema.names:
                            all_keys.extend(b.column("image_key").to_pylist())
                    break
                except Exception:
                    pass
        if not all_keys:
            raise ValueError("image_key column empty or not found")
        logger.info(f"Loaded {len(all_keys):,} keys via pyarrow")
        return all_keys
    except Exception as e:
        logger.warning(f"pyarrow failed ({e}), falling back to HuggingFace datasets...")
        from datasets import load_from_disk
        return load_from_disk(dataset_path)["image_key"]


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — SHARD PREFETCH THREAD
# ═══════════════════════════════════════════════════════════════════════════════

def _shard_prefetcher(shard_names: list, local_q: Queue):
    """
    Copy tar files from the network mount → local NVMe ahead of the extractor.
    The bounded *local_q* provides backpressure so at most SHARD_PREFETCH
    shards occupy /tmp at once.
    """
    LOCAL_TMP.mkdir(parents=True, exist_ok=True)
    for shard_name in shard_names:
        src = SHARD_DIR / shard_name
        dst = LOCAL_TMP / shard_name
        if not src.exists():
            local_q.put(("missing", shard_name))
            continue
        if not dst.exists():
            shutil.copy2(src, dst)   # ~70 MB; done ahead-of-time so extractor never waits
        local_q.put(("ready", shard_name, dst))
    local_q.put(_SHARD_SENTINEL)


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — EXTRACTOR + DECODE SUBMIT THREAD
# ═══════════════════════════════════════════════════════════════════════════════

def _extractor(shard_groups: list, batch_q: Queue, pool: ThreadPoolExecutor):
    """
    For each shard (already on local NVMe via the prefetch thread):
      1. Open the tar, build a stem → TarMember index.
      2. Submit each JPEG's raw bytes to *pool* as a non-blocking future.
      3. Accumulate into BATCH_SIZE chunks and push to *batch_q*.
      4. Delete the local copy immediately to keep /tmp clean.
    """
    shard_map   = {sn: items for sn, items in shard_groups}
    shard_names = [sn for sn, _ in shard_groups]

    local_q    = Queue(maxsize=SHARD_PREFETCH)
    prefetch_t = Thread(target=_shard_prefetcher, args=(shard_names, local_q), daemon=True)
    prefetch_t.start()

    try:
        while True:
            item = local_q.get()
            if item is _SHARD_SENTINEL:
                break
            if item[0] == "missing":
                n = len(shard_map.get(item[1], []))
                batch_q.put(("missing", item[1], n))
                continue

            _, shard_name, local_path = item
            shard_items = shard_map.get(shard_name, [])

            with tarfile.open(local_path, "r") as tf:
                # Index by stem and full name for flexible lookup
                index = {}
                for m in tf.getmembers():
                    if m.isfile() and m.name.lower().endswith((".jpg", ".jpeg")):
                        index[Path(m.name).stem] = m
                        index[m.name] = m

                batch_paths, batch_futures = [], []
                for base_name, out_path_str in shard_items:
                    member = index.get(base_name) or next(
                        (m for k, m in index.items() if base_name in k), None
                    )
                    fut = None
                    if member is not None:
                        try:
                            fut = pool.submit(decode_and_crop, tf.extractfile(member).read())
                        except Exception:
                            pass
                    batch_paths.append(out_path_str)
                    batch_futures.append(fut)
                    if len(batch_paths) == BATCH_SIZE:
                        batch_q.put(("batch", batch_paths, batch_futures))
                        batch_paths, batch_futures = [], []

                if batch_paths:
                    batch_q.put(("batch", batch_paths, batch_futures))

            local_path.unlink(missing_ok=True)   # free NVMe immediately

        prefetch_t.join()
    finally:
        batch_q.put(_BATCH_SENTINEL)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main(args):
    global SHARD_DIR, OUT_DIR, BATCH_SIZE, DECODE_WORKERS, PREFETCH_Q
    global SHARD_PREFETCH, RESUME, DATASET_PATH, TARGET_SIZE

    DATASET_PATH   = args.dataset_path
    SHARD_DIR      = Path(args.shard_dir)
    OUT_DIR        = Path(args.out_dir)
    BATCH_SIZE     = args.batch_size
    DECODE_WORKERS = args.decode_workers
    PREFETCH_Q     = args.prefetch_q
    SHARD_PREFETCH = args.shard_prefetch
    RESUME         = not args.no_resume
    TARGET_SIZE    = args.target_size

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_TMP.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0")

    # ── CUDA performance flags ─────────────────────────────────────────────────
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Dedicated H2D stream — transfer overlaps CPU future resolution for next batch
    h2d_stream = torch.cuda.Stream(device=device)

    props = torch.cuda.get_device_properties(0)
    decode_backend = "cv2"
    logger.info("=" * 64)
    logger.info(f"  encode_latents.py  [{decode_backend} decode]")
    logger.info(f"  {props.name} | {props.total_memory / 1e9:.1f} GB VRAM")
    logger.info(f"  Batch: {BATCH_SIZE} | Decode workers: {DECODE_WORKERS} | "
                f"Shard prefetch: {SHARD_PREFETCH}")
    logger.info(f"  Output: one .npy per image → {OUT_DIR}")
    logger.info("=" * 64)

    # ── Load VAE ───────────────────────────────────────────────────────────────
    from model import PretrainedVAE
    vae = PretrainedVAE(model_id="stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    # torch.compile ~20–30 % throughput gain after first-batch graph capture
    vae = torch.compile(vae, mode="reduce-overhead", fullgraph=False)
    logger.info("VAE loaded and compiled\n")

    # ── Pre-allocate pinned-memory staging buffer ──────────────────────────────
    # Page-locked allocation: CPU → GPU copy goes via DMA, no software bounce buffer.
    # Allocated once to avoid per-batch malloc overhead in the hot loop.
    staging = torch.empty(
        BATCH_SIZE, 3, TARGET_SIZE, TARGET_SIZE,
        dtype=torch.float32,
        pin_memory=True,
    )

    # ── Build work list ────────────────────────────────────────────────────────
    all_keys = read_all_keys(DATASET_PATH)
    logger.info("Building work list...")

    # Resume: os.listdir is O(1) per lookup — no file reads, no chunk scanning.
    done_set = set(os.listdir(OUT_DIR)) if RESUME else set()

    items, skipped = [], 0
    for key in all_keys:
        if "::" not in key:
            continue
        fname = key_to_filename(key)          # e.g. "shard__stem.npy"
        if RESUME and fname in done_set:
            skipped += 1
            continue
        shard_name, base_name = key.split("::", 1)
        items.append((shard_name, base_name, str(OUT_DIR / fname)))

    items.sort(key=lambda x: x[0])           # group by shard for sequential tar reads
    total = len(items)
    logger.info(f"To encode: {total:,} | Already done: {skipped:,}")
    logger.info(f"Expected disk: ~{(total + skipped) * 32768 / 1e9:.1f} GB\n")

    if total == 0:
        logger.info("Nothing to do — all latents already encoded!")
        return

    shard_groups = [
        (sn, [(bn, op) for _, bn, op in grp])
        for sn, grp in groupby(items, key=lambda x: x[0])
    ]

    # ── Mutable counters ───────────────────────────────────────────────────────
    done   = 0
    failed = 0

    # ── GPU encode + save function ─────────────────────────────────────────────
    @torch.no_grad()
    def encode_and_save(paths: list, futures: list):
        """
        1. Resolve decode futures → filter valid arrays.
        2. Copy into pinned staging buffer → non-blocking DMA to GPU.
        3. VAE encode under bfloat16 autocast.
        4. Save each latent directly as a float16 .npy file.
        """
        nonlocal done, failed

        # ── 1. Collect valid decode results ───────────────────────────────────
        valid_paths, valid_arrs = [], []
        for path, fut in zip(paths, futures):
            arr = fut.result() if fut is not None else None
            if arr is not None:
                valid_paths.append(path)
                valid_arrs.append(arr)
            else:
                failed += 1

        if not valid_arrs:
            return

        n = len(valid_arrs)

        # ── 2. Pack into pinned staging buffer (CPU → pinned CPU, zero alloc) ──
        staging[:n].copy_(torch.from_numpy(np.stack(valid_arrs)))

        # ── 3. Non-blocking DMA via dedicated H2D stream ──────────────────────
        with torch.cuda.stream(h2d_stream):
            pixel = staging[:n].to(device, non_blocking=True)
        torch.cuda.current_stream(device).wait_stream(h2d_stream)

        # ── 4. VAE encode (bfloat16 for ~2× throughput on Blackwell) ──────────
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latents = vae.encode(pixel)                # (n, 4, H/8, W/8)

        # ── 5. Save each latent as an individual float16 .npy file ────────────
        # At 32 KB/file, np.save is nearly instantaneous — no async queue needed.
        latents_np = latents.to(dtype=torch.float16).cpu().numpy()
        for path, arr in zip(valid_paths, latents_np):
            np.save(path, arr)
            done += 1

    # ── Pipeline start ─────────────────────────────────────────────────────────
    with ThreadPoolExecutor(max_workers=DECODE_WORKERS) as pool:
        batch_q = Queue(maxsize=PREFETCH_Q)
        ext_t   = Thread(target=_extractor, args=(shard_groups, batch_q, pool), daemon=True)
        ext_t.start()

        pbar = tqdm(total=total, unit="img", dynamic_ncols=True, smoothing=0.05)
        while True:
            item = batch_q.get()
            if item is _BATCH_SENTINEL:
                break
            if item[0] == "missing":
                _, shard_name, n = item
                logger.warning(f"Missing shard: {shard_name} ({n} items skipped)")
                failed += n
                pbar.update(n)
                continue

            _, paths, futures = item
            encode_and_save(paths, futures)
            pbar.update(len(paths))
            pbar.set_postfix(
                done=f"{done:,}",
                fail=failed,
                vram=f"{torch.cuda.memory_reserved(device) / 1e9:.1f}GB",
            )

        ext_t.join()
        pbar.close()

    shutil.rmtree(LOCAL_TMP, ignore_errors=True)

    # ── Final report ──────────────────────────────────────────────────────────
    npy_files = list(OUT_DIR.glob("*.npy"))
    n_files   = len(npy_files)
    disk_gb   = sum(f.stat().st_size for f in npy_files) / 1e9

    logger.info("=" * 64)
    logger.info("  ENCODING COMPLETE")
    logger.info(f"  Encoded : {done:,}")
    logger.info(f"  Failed  : {failed:,}")
    logger.info(f"  Skipped : {skipped:,}")
    logger.info(f"  Files   : {n_files:,}")
    logger.info(f"  Disk    : {disk_gb:.1f} GB")
    logger.info("=" * 64)


# ═══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-encode LAION tar shards → individual float16 .npy VAE latents."
    )

    # ── Paths ─────────────────────────────────────────────────────────────────
    parser.add_argument("--dataset_path",   type=str, default=DATASET_PATH,
                        help="HuggingFace Arrow dataset with image_key column.")
    parser.add_argument("--shard_dir",      type=str, default=str(SHARD_DIR),
                        help="Source tar shard directory (may be a network mount).")
    parser.add_argument("--out_dir",        type=str, default=str(OUT_DIR),
                        help="Output directory for .npy latent files.")

    # ── Storage ───────────────────────────────────────────────────────────────
    parser.add_argument("--target_size",    type=int, default=TARGET_SIZE,
                        help="Resize/crop resolution (default 512). Use 256 to quarter disk usage.")

    # ── Performance ───────────────────────────────────────────────────────────
    parser.add_argument("--batch_size",     type=int, default=BATCH_SIZE,
                        help="Images per VAE encode call.")
    parser.add_argument("--decode_workers", type=int, default=DECODE_WORKERS,
                        help="CPU threads for parallel JPEG decode.")
    parser.add_argument("--prefetch_q",     type=int, default=PREFETCH_Q,
                        help="Max decoded batches queued ahead of the GPU.")
    parser.add_argument("--shard_prefetch", type=int, default=SHARD_PREFETCH,
                        help="Tar shards to keep pre-copied on local NVMe.")

    # ── Behaviour ─────────────────────────────────────────────────────────────
    parser.add_argument("--no_resume",  action="store_true",
                        help="Re-encode all images even if .npy files already exist.")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this script.")

    main(args)
