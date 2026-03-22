#!/usr/bin/env python3
"""Generate a sequence of Lenia 2D frames showing the Orbium glider.

Uses the exact Orbium pattern from Bert Chan's Lenia paper (2019).
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt
from pathlib import Path

# --- Lenia 2D parameters (Orbium, Chan 2019) ---
N = 256          # grid size
R = 13           # kernel radius
T = 10           # time resolution (dt = 1/T)
mu = 0.15        # growth center
sigma = 0.015    # growth width
dt = 1.0 / T

# Exact Orbium pattern from Bert Chan (20x20 cells, values 0-255 mapped to 0-1)
ORBIUM_CELLS = np.array([
    [0,0,0,0,0,0,0.1,0.14,0.1,0,0,0.03,0.03,0,0,0.3,0,0,0,0],
    [0,0,0,0,0,0.08,0.24,0.3,0.3,0.18,0.14,0.15,0.16,0.15,0.09,0.2,0,0,0,0],
    [0,0,0,0,0,0.15,0.34,0.44,0.46,0.38,0.18,0.14,0.11,0.13,0.19,0.18,0.45,0,0,0],
    [0,0,0,0,0.06,0.13,0.39,0.5,0.5,0.37,0.06,0,0,0,0.02,0.16,0.68,0,0,0],
    [0,0,0,0.11,0.17,0.17,0.33,0.4,0.38,0.28,0.14,0,0,0,0,0,0.18,0.42,0,0],
    [0,0,0.09,0.18,0.13,0.06,0.08,0.26,0.32,0.32,0.27,0,0,0,0,0,0,0.82,0,0],
    [0.27,0,0.16,0.12,0,0,0,0.25,0.38,0.44,0.45,0.34,0,0,0,0,0,0.22,0.17,0],
    [0,0.07,0.2,0.02,0,0,0,0.31,0.48,0.57,0.6,0.57,0,0,0,0,0,0,0.49,0],
    [0,0.59,0.19,0,0,0,0,0.2,0.57,0.69,0.76,0.76,0.49,0,0,0,0,0,0.36,0],
    [0,0.58,0.19,0,0,0,0,0,0.67,0.83,0.9,0.92,0.87,0.12,0,0,0,0,0.22,0.07],
    [0,0,0.46,0,0,0,0,0,0.7,0.93,1,1,1,0.61,0,0,0,0,0.18,0.11],
    [0,0,0.82,0,0,0,0,0,0.47,1,1,0.98,1,0.96,0.27,0,0,0,0.19,0.1],
    [0,0,0.46,0,0,0,0,0,0.25,1,1,0.84,0.92,0.97,0.54,0.14,0.04,0.1,0.21,0.05],
    [0,0,0,0.4,0,0,0,0,0.09,0.8,1,0.82,0.8,0.85,0.63,0.31,0.18,0.19,0.2,0.01],
    [0,0,0,0.36,0.1,0,0,0,0.05,0.54,0.86,0.79,0.74,0.72,0.6,0.39,0.28,0.24,0.13,0],
    [0,0,0,0.01,0.3,0.07,0,0,0.08,0.36,0.64,0.7,0.64,0.6,0.51,0.39,0.29,0.19,0.04,0],
    [0,0,0,0,0.1,0.24,0.14,0.1,0.15,0.29,0.45,0.53,0.52,0.46,0.4,0.31,0.21,0.08,0,0],
    [0,0,0,0,0,0.08,0.21,0.21,0.22,0.29,0.36,0.39,0.37,0.33,0.26,0.18,0.09,0,0,0],
    [0,0,0,0,0,0,0.03,0.13,0.19,0.22,0.24,0.24,0.23,0.18,0.13,0.05,0,0,0,0],
    [0,0,0,0,0,0,0,0,0.02,0.06,0.08,0.09,0.07,0.05,0.01,0,0,0,0,0],
])


def bell(x: np.ndarray, m: float, s: float) -> np.ndarray:
    """Smooth bell-shaped function."""
    return np.exp(-((x - m) ** 2) / (2 * s**2))


def make_kernel(N: int, R: int) -> np.ndarray:
    """Annular kernel with single ring, bell cross-section."""
    mid = N // 2
    y, x = np.ogrid[-mid:mid, -mid:mid]
    dist = np.sqrt(x.astype(float)**2 + y.astype(float)**2) / R
    # Single ring at distance 0.5 with width 0.15
    kernel = bell(dist, 0.5, 0.15)
    kernel[dist > 1] = 0
    kernel /= kernel.sum()
    return kernel


def growth(u: np.ndarray) -> np.ndarray:
    """Growth function G(u) = 2*exp(-((u-mu)^2)/(2*sigma^2)) - 1."""
    return 2.0 * np.exp(-((u - mu) ** 2) / (2 * sigma**2)) - 1.0


def place_pattern(grid: np.ndarray, pattern: np.ndarray, cy: int, cx: int) -> None:
    """Place a pattern centered at (cy, cx) on the grid with wrapping."""
    ph, pw = pattern.shape
    for i in range(ph):
        for j in range(pw):
            grid[(cy - ph // 2 + i) % N, (cx - pw // 2 + j) % N] = pattern[i, j]


def run_lenia_2d(steps: int, save_every: int) -> list[tuple[int, np.ndarray]]:
    """Run Lenia 2D and return frames at regular intervals."""
    grid = np.zeros((N, N))
    place_pattern(grid, ORBIUM_CELLS, N // 2, N // 2)

    # Build kernel and pre-compute FFT
    kernel = make_kernel(N, R)
    # Shift kernel center to (0,0) for FFT convolution
    kernel_fft = fft2(fftshift(kernel))

    frames: list[tuple[int, np.ndarray]] = [(0, grid.copy())]

    for step in range(1, steps + 1):
        A_hat = fft2(grid)
        U = np.real(ifft2(A_hat * kernel_fft))
        grid = np.clip(grid + dt * growth(U), 0.0, 1.0)
        if step % save_every == 0:
            frames.append((step, grid.copy()))

    return frames


def main() -> None:
    out_dir = Path(__file__).parent.parent / "plots" / "lenia2d_seq"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running Lenia 2D simulation (Orbium)...")
    frames = run_lenia_2d(steps=600, save_every=50)

    print(f"Saving {len(frames)} frames to {out_dir}/")
    for step, grid in frames:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(grid, cmap="magma", vmin=0, vmax=1, interpolation="bilinear")
        ax.set_title(f"Lenia 2D  —  step {step}", fontsize=13, fontweight="bold")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"frame_{step:04d}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  frame_{step:04d}.png  (mass={grid.sum():.1f})")

    print("Done!")


if __name__ == "__main__":
    main()
