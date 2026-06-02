"""
CPU smoke test: UNet forward pass, parameter count, DDIM step.

Run:
    pip install torch transformers
    python -m pytest tests/test_unet_forward.py -v
    # or directly:
    python tests/test_unet_forward.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch


def test_ddpm_add_noise():
    from model import DDPMScheduler

    sched = DDPMScheduler(steps=1000)
    x = torch.randn(2, 4, 8, 8)
    t = torch.randint(0, 1000, (2,))
    x_t, noise = sched.add_noise(x, t)
    assert x_t.shape == x.shape, f"DDPM add_noise shape mismatch: {x_t.shape} != {x.shape}"
    print("  ✓ DDPMScheduler.add_noise")


def test_ddim_timesteps():
    from model import DDIMScheduler

    ddim = DDIMScheduler(steps=1000)
    ddim.set_timesteps(4, device="cpu")
    assert ddim.timesteps.shape[0] == 4, f"DDIM timesteps count wrong: {ddim.timesteps.shape[0]} != 4"
    print("  ✓ DDIMScheduler.set_timesteps")


def test_ddim_step():
    from model import DDIMScheduler

    ddim = DDIMScheduler(steps=1000, clamp_pred_x0=False)
    ddim.set_timesteps(4, device="cpu")
    noise = torch.randn(1, 4, 8, 8)
    x_t = torch.randn(1, 4, 8, 8)
    t = ddim.timesteps[0]
    x_prev = ddim.step(noise, t, x_t, eta=0.0)
    assert x_prev.shape == x_t.shape, f"DDIM step shape mismatch: {x_prev.shape} != {x_t.shape}"
    assert not torch.isnan(x_prev).any(), "DDIM step produced NaN"
    print("  ✓ DDIMScheduler.step")


def test_ddim_clamp_opt_in():
    from model import DDIMScheduler

    ddim_clamp = DDIMScheduler(steps=1000, clamp_pred_x0=True)
    ddim_noclamp = DDIMScheduler(steps=1000, clamp_pred_x0=False)
    ddim_clamp.set_timesteps(4, device="cpu")
    ddim_noclamp.set_timesteps(4, device="cpu")

    noise = torch.randn(1, 4, 8, 8) * 10  # extreme noise
    x_t = torch.randn(1, 4, 8, 8) * 10
    t = ddim_clamp.timesteps[0]

    out_clamp = ddim_clamp.step(noise, t, x_t, eta=0.0)
    out_noclamp = ddim_noclamp.step(noise, t, x_t, eta=0.0)
    assert out_clamp.shape == out_noclamp.shape
    assert not torch.equal(out_clamp, out_noclamp), "clamp=True vs False should differ with extreme inputs"
    print("  ✓ DDIMScheduler clamp opt-in")


def test_unet_forward():
    from model import UNetModel

    unet = UNetModel(
        in_ch=4, out_ch=4, ch=32,
        res_blks=1, attn_lvls=(1,),
        ch_mults=(1, 2), heads=2,
        t_dim=32, ctx_dim=64,
    )
    unet.eval()
    with torch.no_grad():
        latent = torch.randn(1, 4, 8, 8)
        t = torch.randint(0, 1000, (1,))
        ctx = torch.randn(1, 4, 64)
        out = unet(latent, t, ctx)

    assert out.shape == latent.shape, f"UNet output shape {out.shape} != {latent.shape}"
    total_params = sum(p.numel() for p in unet.parameters())
    print(f"  ✓ UNet forward pass ({total_params:,} params)")


def test_config_import():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))
    from config import SDConfig
    cfg = SDConfig()
    assert cfg.model.ch == 320
    assert cfg.training.ema_decay == 0.9999
    assert cfg.scheduler.guidance_scale == 7.5
    print("  ✓ Config import")


if __name__ == "__main__":
    print("Running smoke tests...\n")
    test_ddpm_add_noise()
    test_ddim_timesteps()
    test_ddim_step()
    test_ddim_clamp_opt_in()
    test_unet_forward()
    test_config_import()
    print("\n✓ All smoke tests passed.")
