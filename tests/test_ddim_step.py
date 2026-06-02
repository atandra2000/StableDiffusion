"""
Test that DDIM with eta=0 preserves latent variance through the denoising
trajectory (i.e., the denoising process doesn't collapse or explode).

Run:
    python -m pytest tests/test_ddim_step.py -v
    # or directly:
    python tests/test_ddim_step.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch


def test_ddim_variance_preservation():
    """
    Verify that DDIM doesn't produce extreme latent shifts.
    The std of latents should stay within reasonable bounds throughout
    the denoising trajectory (not collapse to 0 or explode).
    """
    from model import DDIMScheduler

    ddim = DDIMScheduler(steps=1000, clamp_pred_x0=False)
    num_steps = 50
    ddim.set_timesteps(num_steps, device="cpu")

    # Start from realistic noise (pure Gaussian)
    x_t = torch.randn(2, 4, 16, 16)  # half res for speed
    initial_std = x_t.std().item()

    stds = []
    for i, t in enumerate(ddim.timesteps):
        noise_pred = torch.randn_like(x_t)  # simulate UNet predicting near-Gaussian
        x_t = ddim.step(noise_pred, t, x_t, eta=0.0)
        stds.append(x_t.std().item())

    final_std = stds[-1]
    std_ratio = final_std / initial_std

    # With random noise predictions, the latents shouldn't collapse:
    # a perfectly denoised image would have much lower std than noise,
    # but with random predictions the variance should stay bounded.
    assert 0.1 < std_ratio < 3.0, (
        f"DDIM variance collapsed/extreme: initial_std={initial_std:.4f}, "
        f"final_std={final_std:.4f}, ratio={std_ratio:.4f}"
    )
    print(f"  ✓ DDIM variance preservation (std ratio: {std_ratio:.4f})")


def test_ddim_determinism():
    """Same seed + same inputs should produce identical outputs."""
    from model import DDIMScheduler

    ddim = DDIMScheduler(steps=1000, clamp_pred_x0=False)
    ddim.set_timesteps(10, device="cpu")

    torch.manual_seed(0)
    x_t = torch.randn(1, 4, 8, 8)
    noise = torch.randn_like(x_t)

    out1 = ddim.step(noise.clone(), ddim.timesteps[0], x_t.clone(), eta=0.0)
    out2 = ddim.step(noise.clone(), ddim.timesteps[0], x_t.clone(), eta=0.0)

    assert torch.equal(out1, out2), "DDIM should be deterministic with eta=0"
    print("  ✓ DDIM determinism")


def test_ddim_stochasticity():
    """eta > 0 should produce different (non-deterministic) outputs."""
    from model import DDIMScheduler

    ddim = DDIMScheduler(steps=1000, clamp_pred_x0=False)
    ddim.set_timesteps(10, device="cpu")

    torch.manual_seed(42)
    x_t = torch.randn(1, 4, 8, 8)
    noise = torch.randn_like(x_t)

    out_det = ddim.step(noise.clone(), ddim.timesteps[0], x_t.clone(), eta=0.0)

    # eta=1.0 should differ due to random noise injection
    out_stoch = ddim.step(noise.clone(), ddim.timesteps[0], x_t.clone(), eta=1.0)

    assert not torch.equal(out_det, out_stoch), "eta=1.0 should differ from eta=0.0"
    print("  ✓ DDIM stochasticity (eta)")


if __name__ == "__main__":
    print("Running DDIM tests...\n")
    test_ddim_variance_preservation()
    test_ddim_determinism()
    test_ddim_stochasticity()
    print("\n✓ All DDIM tests passed.")
