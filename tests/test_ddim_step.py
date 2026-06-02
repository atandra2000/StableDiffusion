"""
DDIM scheduler tests: determinism, stochasticity, NaN-free operation.

Run:
    python -m pytest tests/test_ddim_step.py -v
    # or directly:
    python tests/test_ddim_step.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch


def test_ddim_no_nan():
    """DDIM step should never produce NaN regardless of noise inputs."""
    from model import DDIMScheduler

    ddim = DDIMScheduler(steps=1000, clamp_pred_x0=False)
    ddim.set_timesteps(50, device="cpu")

    x_t = torch.randn(2, 4, 8, 8)
    for i, t in enumerate(ddim.timesteps):
        noise_pred = torch.randn_like(x_t)
        x_t = ddim.step(noise_pred, t, x_t, eta=0.0)
        assert not torch.isnan(x_t).any(), f"NaN at step {i}"

    final_std = x_t.std().item()
    assert 0.01 < final_std < 100.0, f"DDIM produced extreme std: {final_std:.4f}"
    print(f"  ✓ DDIM no NaN (final std: {final_std:.4f})")


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
    test_ddim_no_nan()
    test_ddim_determinism()
    test_ddim_stochasticity()
    print("\n✓ All DDIM tests passed.")
