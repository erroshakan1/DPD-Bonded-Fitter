"""Bond-length and angle distributions of mapped (or CG) trajectories, and their comparison.

All lengths here are in Angstrom (CG trajectories are converted with the DPD length unit
before they are compared), angles in degrees. Every group pools all its bonds/angles of all
molecules and frames.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import wasserstein_distance

from .model import CGModel

ANGLE_BIN = 2.0          # degrees
BOND_BINS = 120


@dataclass
class Dist:
    kind: str              # "bond" or "angle"
    group: str
    samples: np.ndarray    # Angstrom or degrees
    n_terms: int = 1       # bonds/angles per molecule in this group

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.group}"

    @property
    def mean(self) -> float:
        return float(np.mean(self.samples)) if len(self.samples) else float("nan")

    @property
    def std(self) -> float:
        return float(np.std(self.samples)) if len(self.samples) else float("nan")

    def histogram(self, edges: np.ndarray) -> np.ndarray:
        """Probability density on ``edges``."""
        h, _ = np.histogram(self.samples, bins=edges, density=False)
        w = np.diff(edges)
        tot = h.sum()
        return h / (tot * w) if tot else h.astype(float)

    def thin(self, max_samples: int, seed: int = 0) -> "Dist":
        if len(self.samples) <= max_samples:
            return self
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.samples), max_samples, replace=False)
        return Dist(self.kind, self.group, self.samples[idx], self.n_terms)


def bond_lengths(x: np.ndarray, pairs) -> np.ndarray:
    """x: (..., n_beads, 3); returns all lengths of ``pairs`` flattened."""
    p = np.asarray(pairs, dtype=int).reshape(-1, 2)
    d = x[..., p[:, 1], :] - x[..., p[:, 0], :]
    return np.linalg.norm(d, axis=-1).ravel()


def angles_deg(x: np.ndarray, triples) -> np.ndarray:
    """x: (..., n_beads, 3); angles (degrees) of ``triples`` (centre bead in the middle)."""
    t = np.asarray(triples, dtype=int).reshape(-1, 3)
    u = x[..., t[:, 0], :] - x[..., t[:, 1], :]
    v = x[..., t[:, 2], :] - x[..., t[:, 1], :]
    nu = np.linalg.norm(u, axis=-1)
    nv = np.linalg.norm(v, axis=-1)
    c = np.einsum("...k,...k->...", u, v) / np.maximum(nu * nv, 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0))).ravel()


def compute_distributions(x: np.ndarray, model: CGModel) -> dict[str, Dist]:
    """Distributions of every bond and angle group of ``model`` for bead coordinates ``x``.

    ``x`` has shape (n_frames, n_molecules, n_beads, 3).
    """
    out: dict[str, Dist] = {}
    for g in model.bond_group_names():
        pairs = model.bonds_in(g)
        d = Dist("bond", g, bond_lengths(x, pairs).astype(float), len(pairs))
        out[d.key] = d
    for g in model.angle_group_names():
        tr = model.angles_in(g)
        d = Dist("angle", g, angles_deg(x, tr).astype(float), len(tr))
        out[d.key] = d
    return out


def emd(a: Dist, b: Dist) -> float:
    """Earth mover's (1-Wasserstein) distance between the two sample sets."""
    if len(a.samples) == 0 or len(b.samples) == 0:
        return float("inf")
    return float(wasserstein_distance(a.samples, b.samples))


# floors for the normalisation, so that extremely narrow reference distributions do not
# dominate the score
_STD_FLOOR = {"bond": 0.02, "angle": 1.0}


@dataclass
class ScoreSettings:
    mode: str = "normalized"     # "normalized": EMD / sigma_AA ; "absolute": Swarm-CG like
    bond_scale: float = 10.0     # absolute mode: degrees per Angstrom of bond EMD
    weights: dict | None = None  # key -> weight (default 1)

    def to_dict(self) -> dict:
        return {"mode": self.mode, "bond_scale": self.bond_scale, "weights": self.weights or {}}

    @classmethod
    def from_dict(cls, d: dict) -> "ScoreSettings":
        return cls(d.get("mode", "normalized"), float(d.get("bond_scale", 10.0)), d.get("weights") or None)


def score(ref: dict[str, Dist], cg: dict[str, Dist], st: ScoreSettings | None = None,
          keys=None) -> tuple[float, dict[str, float]]:
    """Total score (lower is better) and the per-group terms.

    ``normalized``: term = EMD / sigma_ref, i.e. the mismatch in units of the width of the
    all-atom distribution, which puts bonds and angles on the same footing.
    ``absolute``: bonds EMD [A] * bond_scale, angles EMD [deg] (as in Swarm-CG, which mixes
    nm and degrees with a fixed factor).
    The total is the weighted root-mean-square of the terms.
    """
    st = st or ScoreSettings()
    keys = list(ref) if keys is None else list(keys)
    terms: dict[str, float] = {}
    num = den = 0.0
    for k in keys:
        r = ref[k]
        c = cg.get(k)
        e = emd(r, c) if c is not None else float("inf")
        if st.mode == "absolute":
            t = e * st.bond_scale if r.kind == "bond" else e
        else:
            t = e / max(r.std, _STD_FLOOR[r.kind])
        terms[k] = t
        w = (st.weights or {}).get(k, 1.0)
        num += w * t * t
        den += w
    total = float(np.sqrt(num / den)) if den > 0 else float("inf")
    return total, terms


def common_edges(dists: list[Dist], kind: str, width: float) -> np.ndarray:
    """Histogram edges with bin ``width`` (Angstrom for bonds, degrees for angles) covering
    all given distributions; bonds use multiples of ``width`` as edges, angles 0-180."""
    width = float(width)
    if width <= 0:
        raise ValueError("bin width must be positive")
    if kind == "angle":
        return np.append(np.arange(0.0, 180.0, width), 180.0)
    allv = np.concatenate([d.samples for d in dists if len(d.samples)]) if dists else np.zeros(1)
    lo = np.floor(allv.min() / width) * width - width
    hi = np.ceil(allv.max() / width) * width + width
    return np.arange(max(0.0, lo), hi + 0.5 * width, width)
