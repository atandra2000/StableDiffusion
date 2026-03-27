import os
import logging
from huggingface_hub import snapshot_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
)

REPO_ID = "laion/laion2B-en-aesthetic"
OUTPUT_DIR = "/workspace/StableDiffusion/laion_metadata"

def main():
    logging.info("════════════════════════════════════════════════════════════")
    logging.info(" LAION-Aesthetics Metadata Download")
    logging.info("════════════════════════════════════════════════════════════")
    logging.info(f" Repo:       {REPO_ID}")
    logging.info(f" Output:     {OUTPUT_DIR}")
    logging.info(" Mode:       snapshot download")
    logging.info("════════════════════════════════════════════════════════════")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logging.info("📥 Downloading parquet shards via snapshot...")

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=OUTPUT_DIR,
        allow_patterns="*.parquet",
        resume_download=True,
    )

    logging.info("✅ Metadata download completed successfully.")


if __name__ == "__main__":
    main()
