"""Matplotlib figures: AA vs CG distributions and the score history."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .core.distributions import Dist, common_edges

AA_COLOR = "#4a6fa5"
CG_COLOR = "#d1495b"


def draw_distributions(fig, ref: dict[str, Dist], cg: dict[str, Dist], title: str = "",
                       terms: dict | None = None, bond_bin: float = 0.1, angle_bin: float = 2.5) -> None:
    """Fill ``fig`` with one panel per bond/angle group, drawn as in Swarm-CG.

    Each curve is the histogram at the requested bin width (Angstrom, degrees), normalised
    to probability per bin (sum 1), plotted through the bin centres with the area filled;
    a diamond on the axis marks the mean. Nothing is smoothed.
    """
    fig.clear()
    keys = list(ref)
    if not keys:
        return
    n = len(keys)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    axes = fig.subplots(nrow, ncol, squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    for ax, k in zip(axes.ravel(), keys):
        r = ref[k]
        c = cg.get(k)
        edges = common_edges([r] + ([c] if c is not None else []), r.kind,
                             bond_bin if r.kind == "bond" else angle_bin)
        x = 0.5 * (edges[1:] + edges[:-1])
        w = np.diff(edges)
        series = [(r, AA_COLOR, f"AA-mapped (n={len(r.samples)})")]
        if c is not None and len(c.samples):
            series.append((c, CG_COLOR, "CG"))
        for d, color, label in series:
            y = d.histogram(edges) * w                  # probability per bin
            ax.plot(x, y, color=color, lw=1.4, alpha=0.9, label=label)
            ax.fill_between(x, y, color=color, alpha=0.25, lw=0)
            ax.plot(d.mean, 0, marker="D", color=color, ms=5, clip_on=False, zorder=5)
        ax.set_ylim(bottom=0)
        label = f"{r.group}"
        if terms and k in terms:
            label += f"   ({terms[k]:.2f})"
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("distance (Å)" if r.kind == "bond" else "angle (°)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_yticks([])
        if r.kind == "angle":
            lo = max(0.0, min(r.mean - 4 * r.std, (c.mean - 4 * c.std) if c is not None and len(c.samples) else 180))
            ax.set_xlim(lo, 180)
    axes.ravel()[0].legend(fontsize=7, frameon=False)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()


def plot_distributions(ref, cg, path: str | Path, title: str = "", terms=None,
                       bond_bin: float = 0.1, angle_bin: float = 2.5) -> None:
    import matplotlib
    matplotlib.use("Agg", force=False)
    from matplotlib.figure import Figure
    n = len(ref)
    ncol = min(4, max(n, 1))
    nrow = int(np.ceil(n / ncol)) if n else 1
    fig = Figure(figsize=(3.2 * ncol, 2.5 * nrow + 0.4))
    draw_distributions(fig, ref, cg, title, terms, bond_bin, angle_bin)
    fig.savefig(path, dpi=130)


def draw_history(fig, history) -> None:
    fig.clear()
    ax = fig.subplots()
    idx = [r.index for r in history if r.ok and np.isfinite(r.score)]
    sc = [r.score for r in history if r.ok and np.isfinite(r.score)]
    if sc:
        ax.plot(idx, sc, ".", color="#999999", ms=4, label="simulation")
        ax.plot(idx, np.minimum.accumulate(sc), "-", color=CG_COLOR, lw=1.5, label="best so far")
        ax.set_yscale("log")
        ax.legend(fontsize=8, frameon=False)
    ax.set_xlabel("simulation", fontsize=8)
    ax.set_ylabel("score", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()


def plot_history(history, path: str | Path) -> None:
    from matplotlib.figure import Figure
    fig = Figure(figsize=(6, 3.5))
    draw_history(fig, history)
    fig.savefig(path, dpi=130)
