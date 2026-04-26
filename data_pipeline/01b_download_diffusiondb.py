import io
import json
import logging
import zipfile
import tarfile
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
)

OUT_META   = Path("/workspace/StableDiffusion/metadata")
OUT_SHARDS = Path("/workspace/StableDiffusion/shards/diffusiondb")
OUT_META.mkdir(parents=True, exist_ok=True)
OUT_SHARDS.mkdir(parents=True, exist_ok=True)

SAVE_PATH  = OUT_META / "diffusiondb_raw.parquet"
DONE_LOG   = OUT_META / "diffusiondb_done_shards.txt"

BASE_URL   = "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/images"
MAX_SHARDS = 500
WORKERS    = 10
IMAGE_SIZE = 512


def load_done() -> set:
    if DONE_LOG.exists():
        return set(DONE_LOG.read_text().splitlines())
    return set()


def mark_done(shard_name: str):
    with open(DONE_LOG, "a") as f:
        f.write(shard_name + "\n")


def resize_to_512(img_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = img.size
    scale = IMAGE_SIZE / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - IMAGE_SIZE) // 2
    top  = (new_h - IMAGE_SIZE) // 2
    img  = img.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))
    buf  = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def process_shard(shard_num: int) -> list:
    shard_name = f"part-{shard_num:06d}"
    url        = f"{BASE_URL}/{shard_name}.zip"
    tar_path   = OUT_SHARDS / f"{shard_name}.tar"
    rows       = []

    try:
        resp = requests.get(url, timeout=120)
        if resp.status_code == 404:
            return rows
        resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            json_files = [f for f in z.namelist() if f.endswith(".json")]
            if not json_files:
                return rows

            meta = json.loads(z.read(json_files[0]))

            with tarfile.open(tar_path, "w") as tar:
                for img_filename, attrs in meta.items():
                    prompt = attrs.get("p", "") or attrs.get("prompt", "")
                    if not prompt or img_filename not in z.namelist():
                        continue
                    try:
                        img_bytes     = z.read(img_filename)
                        resized_bytes = resize_to_512(img_bytes)
                        key           = f"{shard_name}/{img_filename.replace('.png', '')}"

                        # Write JPEG
                        jpg_info      = tarfile.TarInfo(name=f"{key}.jpg")
                        jpg_info.size = len(resized_bytes)
                        tar.addfile(jpg_info, io.BytesIO(resized_bytes))

                        # Write caption txt
                        txt_bytes     = prompt.encode("utf-8")
                        txt_info      = tarfile.TarInfo(name=f"{key}.txt")
                        txt_info.size = len(txt_bytes)
                        tar.addfile(txt_info, io.BytesIO(txt_bytes))

                        rows.append({"image_key": key, "URL": url, "TEXT": prompt, "source": "diffusiondb"})

                    except Exception:
                        continue

        mark_done(shard_name)

    except Exception as e:
        logging.warning(f"Shard {shard_name} failed: {e}")

    return rows


def main():
    done = load_done()
    pending = [n for n in range(1, MAX_SHARDS + 1)
               if f"part-{n:06d}" not in done]

    if SAVE_PATH.exists() and not pending:
        logging.info(f"Already complete: {len(pd.read_parquet(SAVE_PATH)):,} rows")
        return

    logging.info("════════════════════════════════════════════════════════════")
    logging.info(f" DiffusionDB Download + Extract | Shards: {len(pending)} remaining")
    logging.info(f" Output shards: {OUT_SHARDS}")
    logging.info("════════════════════════════════════════════════════════════")

    # Load existing rows if resuming
    all_rows = []
    if SAVE_PATH.exists():
        all_rows = pd.read_parquet(SAVE_PATH).to_dict("records")
        logging.info(f"Resuming with {len(all_rows):,} existing rows")

    done_count = MAX_SHARDS - len(pending)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(process_shard, n): n for n in pending}
        for fut in as_completed(futures):
            rows = fut.result()
            all_rows.extend(rows)
            done_count += 1
            if done_count % 50 == 0:
                logging.info(f"  Shards done: {done_count}/{MAX_SHARDS} | Rows: {len(all_rows):,}")
                # Incremental save every 50 shards
                pd.DataFrame(all_rows).to_parquet(SAVE_PATH, index=False)

    df = pd.DataFrame(all_rows)
    df["source"] = "diffusiondb"
    df = df.drop_duplicates(subset=["image_key"])
    df.to_parquet(SAVE_PATH, index=False)

    jdb_count = len(pd.read_parquet(OUT_META / "journeydb_raw.parquet")) \
        if (OUT_META / "journeydb_raw.parquet").exists() else 0

    logging.info("════════════════════════════════════════════════════════════")
    logging.info(f" JourneyDB rows:   {jdb_count:,}")
    logging.info(f" DiffusionDB rows: {len(df):,}")
    logging.info(f" Tar shards:       {len(list(OUT_SHARDS.glob('*.tar')))}")
    logging.info(" Next: python3 02_filter_phase1_metadata.py")
    logging.info("════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
