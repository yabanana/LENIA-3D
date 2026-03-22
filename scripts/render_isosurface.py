"""
Reusable 3D isosurface rendering module for Lenia thesis figures.

Uses skimage.measure.marching_cubes + mpl_toolkits.mplot3d for
publication-quality isosurface visualizations.
"""

import struct
import numpy as np
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource


def load_grid_bin(path: str) -> np.ndarray:
    """Load a 3D grid from binary file (3×int32 header + float32 data)."""
    with open(path, "rb") as f:
        nz, nr, nc = struct.unpack("iii", f.read(12))
        data = np.frombuffer(f.read(nz * nr * nc * 4), dtype=np.float32)
    return data.reshape((nz, nr, nc)).copy()  # writable copy for marching_cubes


def render_isosurface(
    grid: np.ndarray,
    iso_level: float,
    ax,
    azim: float = -60,
    elev: float = 30,
    color: str = "#2196F3",
    alpha: float = 0.85,
    edgecolor: str = "none",
) -> None:
    """Render a 3D isosurface onto an existing Axes3D."""
    if grid.max() < iso_level:
        return

    verts, faces, normals, _ = marching_cubes(grid, level=iso_level)

    # Build polygon collection
    mesh = Poly3DCollection(
        verts[faces],
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=0.1,
    )

    # Phong-like shading using LightSource
    ls = LightSource(azdeg=315, altdeg=45)
    face_normals = normals[faces].mean(axis=1)
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-10

    # Compute diffuse shading
    light_dir = np.array(
        [
            np.cos(np.radians(ls.altdeg)) * np.cos(np.radians(ls.azdeg)),
            np.cos(np.radians(ls.altdeg)) * np.sin(np.radians(ls.azdeg)),
            np.sin(np.radians(ls.altdeg)),
        ]
    )
    shade = np.clip(face_normals @ light_dir, 0, 1)
    shade = 0.3 + 0.7 * shade  # ambient + diffuse

    # Convert base color to RGB and apply shading
    base_rgb = np.array(plt.cm.colors.to_rgb(color))
    face_colors = np.outer(shade, base_rgb)
    face_colors = np.clip(face_colors, 0, 1)
    face_colors = np.column_stack([face_colors, np.full(len(shade), alpha)])

    mesh.set_facecolor(face_colors)
    ax.add_collection3d(mesh)

    # Set axis limits
    N = grid.shape[0]
    margin = N * 0.05
    ax.set_xlim([-margin, N + margin])
    ax.set_ylim([-margin, N + margin])
    ax.set_zlim([-margin, N + margin])
    ax.view_init(elev=elev, azim=azim)

    # Clean up axes for publication
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1])


def render_pattern_card(
    grid: np.ndarray,
    name: str,
    iso_level: float = 0.15,
    angles: list[tuple[float, float]] | None = None,
    color: str = "#2196F3",
    figsize: tuple[float, float] = (10, 5),
) -> plt.Figure:
    """Render a multi-angle figure for one pattern.

    Returns a figure with one subplot per viewing angle.
    """
    if angles is None:
        angles = [(-60, 30), (30, 20)]

    n_angles = len(angles)
    fig = plt.figure(figsize=figsize, facecolor="white")

    for i, (azim, elev) in enumerate(angles):
        ax = fig.add_subplot(1, n_angles, i + 1, projection="3d", facecolor="white")
        render_isosurface(grid, iso_level, ax, azim=azim, elev=elev, color=color)

    fig.suptitle(name, fontsize=14, fontweight="bold", y=0.95)
    fig.tight_layout()
    return fig


def render_gallery(
    grids: dict[str, np.ndarray],
    iso_level: float = 0.15,
    colors: list[str] | None = None,
    ncols: int = 3,
    figsize: tuple[float, float] = (14, 9),
) -> plt.Figure:
    """Render a gallery grid of multiple patterns.

    Args:
        grids: dict mapping pattern name to 3D grid array.
        iso_level: isosurface threshold.
        colors: list of colors, one per pattern.
        ncols: number of columns.
        figsize: figure size.

    Returns:
        matplotlib Figure.
    """
    names = list(grids.keys())
    n = len(names)
    nrows = (n + ncols - 1) // ncols

    if colors is None:
        cmap = plt.cm.Set2
        colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    fig = plt.figure(figsize=figsize, facecolor="white")

    for i, name in enumerate(names):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d", facecolor="white")
        c = colors[i] if isinstance(colors[i], str) else colors[i]
        render_isosurface(grids[name], iso_level, ax, azim=-60, elev=25, color=c)
        ax.set_title(name, fontsize=11, fontweight="bold", pad=-10)

    fig.tight_layout()
    return fig
