import io
import logging
import tarfile
import pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
)

OUT_SHARDS = Path("/workspace/StableDiffusion/shards/journeydb")
OUT_META   = Path("/workspace/StableDiffusion/metadata")
OUT_SHARDS.mkdir(parents=True, exist_ok=True)

DONE_LOG   = OUT_META / "journeydb_done_archives.txt"
PROMPT_MAP = OUT_META / "journeydb_raw.parquet"
IMAGE_SIZE = 512
ARCHIVES   = [f"{i:03d}" for i in range(10)]  # 000 through 009, ~160 GB, ~210K images


def load_done() -> set:
    if DONE_LOG.exists():
        return set(DONE_LOG.read_text().splitlines())
    return set()


def mark_done(archive_id: str):
    with open(DONE_LOG, "a") as f:
        f.write(archive_id + "\n")


def resize_to_512(img_bytes: bytes) -> bytes:
    img  = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = img.size
    scale = IMAGE_SIZE / min(w, h)
    nw, nh = int(w * scale), int(h * scale)
    img  = img.resize((nw, nh), Image.LANCZOS)
    left = (nw - IMAGE_SIZE) // 2
    top  = (nh - IMAGE_SIZE) // 2
    img  = img.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))
    buf  = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def process_archive(archive_id: str, prompt_map: dict) -> list:
    filename   = f"data/train/imgs/{archive_id}.tgz"
    out_tar    = OUT_SHARDS / f"journeydb_{archive_id}.tar"
    rows       = []

    logging.info(f"Downloading archive {archive_id}.tgz ...")
    hf_hub_download(
        repo_id="JourneyDB/JourneyDB",
        filename=filename,
        repo_type="dataset",
        local_dir=str(OUT_META),
    )
    actual_tgz = OUT_META / filename  # placed by hf_hub_download

    logging.info(f"Extracting + resizing {archive_id}.tgz → {out_tar.name} ...")
    with tarfile.open(actual_tgz, "r:gz") as src, tarfile.open(out_tar, "w") as dst:
        for member in src.getmembers():
            if not member.name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            try:
                img_bytes     = src.extractfile(member).read()
                resized_bytes = resize_to_512(img_bytes)
                img_name      = Path(member.name).stem
                key           = f"journeydb_{archive_id}/{img_name}"
                prompt        = prompt_map.get(img_name, "")

                # Write JPEG
                jpg_info      = tarfile.TarInfo(name=f"{key}.jpg")
                jpg_info.size = len(resized_bytes)
                dst.addfile(jpg_info, io.BytesIO(resized_bytes))

                # Write caption
                txt_bytes     = prompt.encode("utf-8") if prompt else img_name.encode()
                txt_info      = tarfile.TarInfo(name=f"{key}.txt")
                txt_info.size = len(txt_bytes)
                dst.addfile(txt_info, io.BytesIO(txt_bytes))

                rows.append({"image_key": key, "TEXT": prompt or img_name, "source": "journeydb"})

            except Exception:
                continue

    # Delete source tgz to free space
    actual_tgz.unlink(missing_ok=True)
    logging.info(f"Archive {archive_id} done: {len(rows):,} images → deleted source tgz")
    mark_done(archive_id)
    return rows


def main():
    done = load_done()
    pending = [a for a in ARCHIVES if a not in done]

    if not pending:
        logging.info("All archives already processed.")
        return

    logging.info("════════════════════════════════════════════════════════════")
    logging.info(f" JourneyDB Subset Download | Archives: {len(pending)} pending")
    logging.info(f" Output: {OUT_SHARDS}")
    logging.info("════════════════════════════════════════════════════════════")

    # Build prompt lookup: image filename stem → prompt
    logging.info("Loading prompt map from journeydb_raw.parquet ...")
    df_prompts  = pd.read_parquet(PROMPT_MAP)
    prompt_map  = {}
    for _, row in df_prompts.iterrows():
        stem = Path(str(row.get("URL", ""))).stem
        if stem:
            prompt_map[stem] = row.get("TEXT", "")
    logging.info(f"Prompt map loaded: {len(prompt_map):,} entries")

    all_rows = []
    for archive_id in pending:
        rows = process_archive(archive_id, prompt_map)
        all_rows.extend(rows)

    out_parquet = OUT_META / "journeydb_images_meta.parquet"
    existing    = pd.read_parquet(out_parquet).to_dict("records") if out_parquet.exists() else []
    combined    = pd.DataFrame(existing + all_rows).drop_duplicates(subset=["image_key"])
    combined.to_parquet(out_parquet, index=False)

    logging.info("════════════════════════════════════════════════════════════")
    logging.info(f" JourneyDB images processed: {len(combined):,}")
    logging.info(f" Tar shards: {len(list(OUT_SHARDS.glob('*.tar')))}")
    logging.info(" Next: python3 02_filter_phase1_metadata.py")
    logging.info("════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
