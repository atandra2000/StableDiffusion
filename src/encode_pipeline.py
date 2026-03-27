#!/usr/bin/env python3
"""
Dual-GPU VAE Encoding with Process Isolation
"""

import os
import sys
import time
import tarfile
import multiprocessing as mp
from pathlib import Path
import cv2 as cv
import numpy as np

# ========== CONFIGURATION ==========
SHARD_SRC = Path("/workspace/StableDiffusion/laion_shards")
OUT_DIR = Path("/workspace/StableDiffusion/laion_latents")
BATCH_SHARDS = 80
TARGET = 512
BATCH = 32  # Batch size per GPU

# ========== GPU WORKER (Runs in separate process) ==========
def gpu_worker(gpu_physical_id, image_paths, output_dir, worker_id):
    """
    Worker that runs on a specific physical GPU.
    This runs in its own process with isolated CUDA context.
    """
    # Set CUDA to use ONLY this physical GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_physical_id)
    
    # Now import torch (after setting CUDA_VISIBLE_DEVICES)
    import torch
    from diffusers import AutoencoderKL
    
    device = torch.device("cuda:0")  # This will be the physical GPU we set
    
    print(f"[Worker {worker_id}] Running on physical GPU {gpu_physical_id}", flush=True)
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float16
    ).to(device)
    vae.eval()
    
    for param in vae.parameters():
        param.requires_grad = False
    
    print(f"[Worker {worker_id}] VAE loaded, encoding {len(image_paths)} images", flush=True)
    
    # Decode and encode function
    def decode_jpeg(path):
        try:
            img = cv.imdecode(np.fromfile(path, dtype=np.uint8), cv.IMREAD_COLOR)
            if img is None:
                return None
            
            img = cv.resize(img, (TARGET, TARGET), interpolation=cv.INTER_LINEAR)
            img = img.astype(np.float32)
            img = img[..., ::-1]  # BGR to RGB
            img = (img / 127.5) - 1.0
            img = img.transpose(2, 0, 1)
            return img
        except:
            return None
    
    def encode_batch(images):
        with torch.no_grad():
            tensor = torch.from_numpy(np.stack(images)).to(device, dtype=torch.float16)
            moments = vae.encoder(tensor)
            mean, _ = torch.chunk(moments, 2, dim=1)
            latents = mean * 0.18215
            return latents.to(torch.float16).cpu().numpy()
    
    # Process images in streaming fashion
    encoded = 0
    batch_buffer = []
    batch_paths = []
    start_time = time.time()
    
    for i, path in enumerate(image_paths):
        img = decode_jpeg(path)
        if img is None:
            continue
        
        batch_buffer.append(img)
        batch_paths.append(path)
        
        if len(batch_buffer) >= BATCH:
            latents = encode_batch(batch_buffer)
            
            for p, latent in zip(batch_paths, latents):
                out_path = output_dir / (p.stem + ".npy")
                np.save(out_path, latent)
            
            encoded += len(batch_buffer)
            batch_buffer = []
            batch_paths = []
            
            if encoded % 5000 == 0:
                elapsed = time.time() - start_time
                rate = encoded / elapsed
                print(f"[Worker {worker_id}] {encoded:,} encoded | {rate:.0f} img/s", flush=True)
    
    # Final batch
    if batch_buffer:
        latents = encode_batch(batch_buffer)
        for p, latent in zip(batch_paths, latents):
            out_path = output_dir / (p.stem + ".npy")
            np.save(out_path, latent)
        encoded += len(batch_buffer)
    
    elapsed = time.time() - start_time
    rate = encoded / elapsed if elapsed > 0 else 0
    print(f"[Worker {worker_id}] Done: {encoded:,} in {elapsed:.0f}s ({rate:.0f} img/s)", flush=True)
    
    return encoded

# ========== EXTRACTION ==========
def extract_batch(batch_shards, jpeg_dir):
    """Extract shards to directory."""
    print(f"Extracting {len(batch_shards)} shards...", flush=True)
    
    # Clear
    for f in jpeg_dir.glob("*.jpg"):
        try:
            f.unlink()
        except:
            pass
    
    extracted = 0
    for shard in batch_shards:
        try:
            with tarfile.open(shard, "r") as tf:
                for m in tf.getmembers():
                    if m.isfile() and m.name.lower().endswith((".jpg", ".jpeg")):
                        out_name = f"{shard.name}__{Path(m.name).stem}.jpg"
                        out_path = jpeg_dir / out_name
                        if not out_path.exists():
                            data = tf.extractfile(m).read()
                            out_path.write_bytes(data)
                            extracted += 1
        except Exception as e:
            print(f"  Error: {shard.name}: {e}")
    
    print(f"Extracted {extracted:,} images", flush=True)
    return extracted

# ========== MAIN ==========
def main():
    print("\n" + "="*60)
    print("DUAL-GPU VAE ENCODER (Process Isolation)")
    print("="*60 + "\n")
    
    # Set start method for multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # Get shards
    all_shards = sorted(SHARD_SRC.glob("*.tar"))
    batches = [all_shards[i:i+BATCH_SHARDS] for i in range(0, len(all_shards), BATCH_SHARDS)]
    print(f"Total: {len(all_shards)} shards → {len(batches)} batches\n")
    
    # Setup slot
    slot = Path("/dev/shm/current_batch")
    slot.mkdir(exist_ok=True)
    
    total_encoded = 0
    start_time = time.time()
    
    # Get existing files
    existing = {f.stem for f in OUT_DIR.glob("*.npy")}
    print(f"Found {len(existing)} existing latents\n")
    
    # Process each batch
    for idx, batch in enumerate(batches):
        print(f"\n{'='*60}")
        print(f"BATCH {idx+1}/{len(batches)}")
        print(f"{'='*60}")
        
        # Extract
        extract_start = time.time()
        extract_batch(batch, slot)
        extract_time = time.time() - extract_start
        
        # Get images to encode
        jpegs = sorted(slot.glob("*.jpg"))
        todo = [f for f in jpegs if f.stem not in existing]
        
        print(f"\n{len(todo)}/{len(jpegs)} images to encode (extract: {extract_time:.0f}s)")
        
        if todo:
            # Split between physical GPUs
            mid = len(todo) // 2
            gpu0_images = todo[:mid]
            gpu1_images = todo[mid:]
            
            print(f"Physical GPU 0: {len(gpu0_images):,} images")
            print(f"Physical GPU 1: {len(gpu1_images):,} images\n")
            
            # Run on both GPUs in parallel using separate processes
            encode_start = time.time()
            
            with mp.Pool(processes=2) as pool:
                # Submit tasks to both physical GPUs
                result0 = pool.apply_async(gpu_worker, (0, gpu0_images, OUT_DIR, 0))
                result1 = pool.apply_async(gpu_worker, (1, gpu1_images, OUT_DIR, 1))
                
                count0 = result0.get()
                count1 = result1.get()
                
                batch_encoded = count0 + count1
                total_encoded += batch_encoded
                
                # Update existing
                for f in todo:
                    existing.add(f.stem)
            
            encode_time = time.time() - encode_start
            print(f"\n✅ Batch {idx+1}: +{batch_encoded:,} in {encode_time:.0f}s")
        
        # Clean slot
        for f in slot.glob("*.jpg"):
            f.unlink()
        
        # Progress report
        elapsed = time.time() - start_time
        rate = total_encoded / elapsed if elapsed > 0 else 0
        remaining = (len(existing) + total_encoded)
        eta = (remaining - total_encoded) / rate / 3600 if rate > 0 else 0
        print(f"\n📊 SPEED: {rate:.0f} img/s | TOTAL: {total_encoded:,} | ETA: {eta:.1f}h")
    
    # Final
    total_time = time.time() - start_time
    final_count = sum(1 for _ in OUT_DIR.glob("*.npy"))
    
    print("\n" + "="*60)
    print(f"✅ COMPLETE: {final_count:,} images in {total_time/3600:.1f}h")
    print(f"Average: {final_count/total_time:.0f} img/s")
    print("="*60)

if __name__ == "__main__":
    main()