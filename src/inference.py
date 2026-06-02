"""
inference.py — SD Inference on Apple Silicon (M1/M2/M3)
========================================================
Loads EMA weights from a training checkpoint and runs DDIM sampling.

Requirements:
    pip install torch torchvision diffusers transformers Pillow

Usage:
    python3 inference.py --prompt "a beautiful sunset over mountain peaks"
    python3 inference.py --prompt "your prompt" --steps 50 --guidance 7.5 --seed 42
    python3 inference.py --prompt "your prompt" --output my_image.png
    python3 inference.py --batch "prompts.txt" --output_dir ./outputs

Setup:
    1. Copy SD_Model.py to the same directory as this script
    2. Copy your checkpoint (sd_epoch_021.pt) to the same directory
    3. Run: python3 inference.py --prompt "your prompt"
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPTokenizer


# ── Device ────────────────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        print("Warning: no GPU found, running on CPU — will be slow")
        return torch.device("cpu")


# ── Fixed DDIM step (removes destructive latent-space clamp) ─────────────────

def _fixed_ddim_step(self, noise_pred, t, x_t, eta=0.0):
    """
    Corrected DDIM step. The original SD_Model.py clamps pred_x0 to [-1, 1]
    which is wrong — SD latents have std ~4, not 1. Clamping destroys signal
    and causes broken outputs at inference time. This version removes the clamp.
    """
    device    = x_t.device
    t_int     = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
    step_size = self.num_train_timesteps // self.num_inference_steps
    prev_t    = t_int - step_size

    alpha_t    = self.alphas_cumprod[t_int].to(device)
    alpha_prev = (
        self.alphas_cumprod[prev_t].to(device)
        if prev_t >= 0
        else torch.ones(1, device=device)
    )

    # Estimate clean latent — no clamp, latents are NOT in [-1, 1]
    pred_x0 = (x_t - (1.0 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()

    # Direction from x̂_0 towards x_t
    dir_xt = (1.0 - alpha_prev).sqrt() * noise_pred

    # Interpolate
    x_prev = alpha_prev.sqrt() * pred_x0 + dir_xt

    if eta > 0.0:
        sigma_t = eta * (
            (1.0 - alpha_prev) / (1.0 - alpha_t)
            * (1.0 - alpha_t / alpha_prev)
        ).clamp(min=0.0).sqrt()
        x_prev = x_prev + sigma_t * torch.randn_like(x_t)

    return x_prev


# ── EMA weight loader ─────────────────────────────────────────────────────────

def load_ema_unet(checkpoint_path: str, unet, device: torch.device):
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "ema_state_dict" not in ckpt:
        raise KeyError("No 'ema_state_dict' found. Make sure this checkpoint was saved by SD_Train.py.")

    shadow      = ckpt["ema_state_dict"]["shadow"]
    epoch       = ckpt.get("epoch", "?")
    global_step = ckpt.get("global_step", "?")
    best_loss   = ckpt.get("best_loss", "?")
    print(f"  Epoch: {epoch} | Step: {global_step} | Best loss: {best_loss}")
    print(f"  EMA shadow keys: {len(shadow):,}")

    cleaned = {}
    for k, v in shadow.items():
        key = k
        for prefix in ("module.", "unet.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = v

    missing, unexpected = unet.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys ignored")

    print("  EMA weights loaded")
    unet.to(device)
    unet.eval()
    return unet


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device):
    sys.path.insert(0, str(Path(__file__).parent))
    import SD_Model
    from SD_Model import (
        PretrainedVAE,
        PretrainedCLIPTextEncoder,
        UNetModel,
        DDIMScheduler,
    )

    # Patch DDIMScheduler to remove destructive latent-space clamp
    SD_Model.DDIMScheduler.step = _fixed_ddim_step

    print("Loading VAE...")
    vae = PretrainedVAE(model_id="stabilityai/sd-vae-ft-mse", use_fp16=False)
    vae.to(device)
    vae.eval()

    print("Loading CLIP text encoder...")
    text_encoder = PretrainedCLIPTextEncoder(model_id="openai/clip-vit-large-patch14")
    text_encoder.to(device)
    text_encoder.eval()

    print("Building UNet...")
    unet = UNetModel(
        in_ch=4, out_ch=4, ch=320,
        ch_mults=(1, 2, 4, 4), res_blks=2,
        attn_lvls=(1, 2, 3), heads=8, ctx_dim=768,
    )
    unet = load_ema_unet(checkpoint_path, unet, device)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    scheduler = DDIMScheduler()
    scheduler.to(device)

    return vae, text_encoder, unet, tokenizer, scheduler


# ── Inference loop ────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(
    prompts:         list,
    vae,
    text_encoder,
    unet,
    tokenizer,
    scheduler,
    device:          torch.device,
    num_steps:       int   = 50,
    guidance_scale:  float = 7.5,
    seed:            int   = 42,
    height:          int   = 512,
    width:           int   = 512,
    negative_prompts: list = None,
) -> list:
    assert height % 8 == 0 and width % 8 == 0
    batch_size = len(prompts)
    latent_h   = height // 8
    latent_w   = width  // 8

    if negative_prompts is None:
        negative_prompts = [""] * batch_size
    elif len(negative_prompts) == 1 and batch_size > 1:
        negative_prompts = negative_prompts * batch_size

    # Encode text
    def encode(texts):
        tok = tokenizer(
            texts, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt",
        ).to(device)
        emb, _ = text_encoder(tok.input_ids, tok.attention_mask)
        return emb.float()

    cond_emb   = encode(prompts)
    uncond_emb = encode(negative_prompts)
    ctx = torch.cat([uncond_emb, cond_emb], dim=0)

    # Initial noise — generate on CPU then move (MPS generator workaround)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latents   = torch.randn(
        (batch_size, 4, latent_h, latent_w),
        generator=generator,
    ).to(device).float()

    # Denoising loop
    scheduler.set_timesteps(num_steps, device)

    for i, t in enumerate(scheduler.timesteps):
        latent_in = torch.cat([latents, latents], dim=0)
        t_batch   = torch.full(
            (latent_in.shape[0],), t, dtype=torch.long, device=device
        )
        noise_pred  = unet(latent_in.float(), t_batch, ctx)
        uncond, cond = noise_pred.chunk(2)
        guided      = uncond + guidance_scale * (cond - uncond)
        latents     = scheduler.step(guided, t, latents)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Step {i+1:3d}/{num_steps}", end="\r")

    print()

    # Decode
    images_tensor = vae.decode(latents.float()).clamp(-1.0, 1.0)
    images_tensor = (images_tensor + 1.0) / 2.0

    pil_images = []
    for i in range(batch_size):
        arr = (images_tensor[i].cpu().float().permute(1, 2, 0).numpy() * 255).astype("uint8")
        pil_images.append(Image.fromarray(arr))

    return pil_images


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SD Inference — Apple Silicon")
    p.add_argument("--prompt",     type=str,   default=None)
    p.add_argument("--batch",      type=str,   default=None,  help=".txt file, one prompt per line")
    p.add_argument("--checkpoint", type=str,   default="sd_epoch_042.pt")
    p.add_argument("--output",     type=str,   default="output.png")
    p.add_argument("--output_dir", type=str,   default="./outputs")
    p.add_argument("--steps",      type=int,   default=50,    help="DDIM steps (25=fast, 50=good, 100=best)")
    p.add_argument("--guidance",   type=float, default=7.5,   help="CFG scale")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--width",      type=int,   default=512)
    p.add_argument("--height",     type=int,   default=512)
    p.add_argument("--batch_size", type=int,   default=1)
    p.add_argument("--negative",   type=str,   default="",    help="Negative prompt")
    return p.parse_args()


def main():
    args = parse_args()

    if args.prompt is None and args.batch is None:
        print("Error: provide --prompt or --batch")
        sys.exit(1)

    if not Path(args.checkpoint).exists():
        print(f"Error: checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    if not Path("SD_Model.py").exists():
        print("Error: SD_Model.py not found in current directory")
        sys.exit(1)

    device = get_device()
    print(f"Device: {device}")

    t0 = time.time()
    vae, text_encoder, unet, tokenizer, scheduler = load_model(args.checkpoint, device)
    print(f"Model loaded in {time.time()-t0:.1f}s\n")

    if args.batch:
        prompts = [l.strip() for l in Path(args.batch).read_text().splitlines() if l.strip()]
        print(f"Batch mode: {len(prompts)} prompts")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    else:
        prompts = [args.prompt]

    all_images = []
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i:i + args.batch_size]
        print(f"Generating: {batch[0][:80]}")
        t0 = time.time()
        imgs = generate(
            prompts          = batch,
            vae              = vae,
            text_encoder     = text_encoder,
            unet             = unet,
            tokenizer        = tokenizer,
            scheduler        = scheduler,
            device           = device,
            num_steps        = args.steps,
            guidance_scale   = args.guidance,
            seed             = args.seed + i,
            height           = args.height,
            width            = args.width,
            negative_prompts = [args.negative] if args.negative else None,
        )
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s ({elapsed/args.steps:.2f}s/step)")
        all_images.extend(zip(batch, imgs))

    if args.batch:
        for j, (prompt, img) in enumerate(all_images):
            name = "".join(c if c.isalnum() or c in " _-" else "_" for c in prompt[:50]).strip()
            path = Path(args.output_dir) / f"{j:04d}_{name}.png"
            img.save(path)
            print(f"Saved: {path}")
    else:
        all_images[0][1].save(args.output)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()