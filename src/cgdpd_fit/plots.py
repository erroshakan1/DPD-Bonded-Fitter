"""Matplotlib figures: AA vs CG distributions and the score history."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .core.distributions import Dist, common_edges

AA_COLOR = "#4a6fa5"
CG_COLOR = "#d1495b"


def draw_distributions(fig, ref: dict[str, Dist], cg: dict[str, Dist], title: str = "",
                       terms: dict | None = None) -> None:
    """Fill ``fig`` with one panel per bond/angle group."""
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
        edges = common_edges([r] + ([c] if c is not None else []), r.kind)
        x = 0.5 * (edges[1:] + edges[:-1])
        ax.fill_between(x, r.histogram(edges), step="mid", color=AA_COLOR, alpha=0.35, lw=0)
        ax.plot(x, r.histogram(edges), drawstyle="steps-mid", color=AA_COLOR, lw=1.2, label="AA")
        if c is not None and len(c.samples):
            ax.plot(x, c.histogram(edges), drawstyle="steps-mid", color=CG_COLOR, lw=1.4, label="CG")
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


def plot_distributions(ref, cg, path: str | Path, title: str = "", terms=None) -> None:
    import matplotlib
    matplotlib.use("Agg", force=False)
    from matplotlib.figure import Figure
    n = len(ref)
    ncol = min(4, max(n, 1))
    nrow = int(np.ceil(n / ncol)) if n else 1
    fig = Figure(figsize=(3.2 * ncol, 2.5 * nrow + 0.4))
    draw_distributions(fig, ref, cg, title, terms)
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
