"""Bonded parameters, Boltzmann-inversion initial guesses and iterative updates.

Conventions follow LAMMPS (no factor 1/2):

* ``bond_style harmonic``         E = K (r - r0)^2            K in kT / r_c^2, r0 in r_c
* ``angle_style harmonic``        E = K (theta - theta0)^2    K in kT / rad^2, theta0 in degrees
* ``angle_style cosine/squared``  E = K (cos theta - cos theta0)^2

For a harmonic well the Boltzmann distribution is Gaussian with variance
sigma^2 = kT / (2 K), hence the Boltzmann-inversion estimate K = kT / (2 sigma^2) with sigma
measured in the units of the potential (DPD lengths, radians or cosines). This first guess
neglects the Jacobian (r^2, sin theta) and the non-bonded forces that act between bonded
beads in DPD; the optimiser corrects both.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .distributions import Dist
from .units import DPDUnits

ANGLE_STYLES = ("harmonic", "cosine/squared")


@dataclass
class BondedParams:
    """K and equilibrium value per group. Bonds: (K, r0[r_c]); angles: (K, theta0[deg])."""

    bonds: dict[str, tuple[float, float]] = field(default_factory=dict)
    angles: dict[str, tuple[float, float]] = field(default_factory=dict)
    angle_style: str = "harmonic"

    def copy(self) -> "BondedParams":
        return BondedParams(dict(self.bonds), dict(self.angles), self.angle_style)

    def to_dict(self) -> dict:
        return {"angle_style": self.angle_style,
                "bonds": {g: list(v) for g, v in self.bonds.items()},
                "angles": {g: list(v) for g, v in self.angles.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "BondedParams":
        return cls({g: tuple(v) for g, v in d.get("bonds", {}).items()},
                   {g: tuple(v) for g, v in d.get("angles", {}).items()},
                   d.get("angle_style", "harmonic"))

    # ------------------------------------------------- vector representation
    def keys(self) -> list[tuple[str, str, int]]:
        """(kind, group, index) for every scalar; index 0 = K, 1 = equilibrium value."""
        out = []
        for g in self.bonds:
            out += [("bond", g, 0), ("bond", g, 1)]
        for g in self.angles:
            out += [("angle", g, 0), ("angle", g, 1)]
        return out

    def get(self, key) -> float:
        kind, g, i = key
        return (self.bonds if kind == "bond" else self.angles)[g][i]

    def set(self, key, value: float) -> None:
        kind, g, i = key
        d = self.bonds if kind == "bond" else self.angles
        v = list(d[g])
        v[i] = float(value)
        d[g] = tuple(v)


def initial_guess(ref: dict[str, Dist], units: DPDUnits, angle_style: str = "harmonic",
                  k_max_bond: float = 5000.0, k_max_angle: float = 500.0) -> BondedParams:
    """Boltzmann inversion of the mapped all-atom distributions (Gaussian approximation)."""
    p = BondedParams(angle_style=angle_style)
    kT = units.kT
    for d in ref.values():
        if d.kind == "bond":
            r = units.to_dpd(d.samples)
            var = max(float(np.var(r)), 1e-8)
            p.bonds[d.group] = (min(kT / (2.0 * var), k_max_bond), float(np.mean(r)))
        elif d.kind == "angle":
            th = np.radians(d.samples)
            if angle_style == "cosine/squared":
                c = np.cos(th)
                var = max(float(np.var(c)), 1e-8)
                th0 = float(np.degrees(np.arccos(np.clip(np.mean(c), -1, 1))))
            else:
                var = max(float(np.var(th)), 1e-8)
                th0 = float(np.degrees(np.mean(th)))
            p.angles[d.group] = (min(kT / (2.0 * var), k_max_angle), th0)
    return p


def ibi_update(p: BondedParams, ref: dict[str, Dist], cg: dict[str, Dist], units: DPDUnits,
               alpha: float = 0.7) -> BondedParams:
    """One iterative-Boltzmann-like update of the Gaussian moments.

    K <- K * (sigma_CG^2 / sigma_AA^2)^alpha   (CG too broad -> stiffer)
    x0 <- x0 + alpha * (<x>_AA - <x>_CG)       (shift the equilibrium value)
    """
    q = p.copy()
    for d in ref.values():
        c = cg.get(d.key)
        if c is None or len(c.samples) < 2:
            continue
        if d.kind == "bond":
            K, r0 = p.bonds[d.group]
            ratio = max(np.var(c.samples), 1e-12) / max(np.var(d.samples), 1e-12)
            K = K * ratio ** alpha
            r0 = r0 + alpha * units.to_dpd(d.mean - c.mean)
            q.bonds[d.group] = (float(K), float(max(r0, 0.01)))
        elif d.kind == "angle":
            K, th0 = p.angles[d.group]
            if p.angle_style == "cosine/squared":
                va, vc = np.var(np.cos(np.radians(d.samples))), np.var(np.cos(np.radians(c.samples)))
            else:
                va, vc = np.var(d.samples), np.var(c.samples)
            K = K * (max(vc, 1e-12) / max(va, 1e-12)) ** alpha
            th0 = float(np.clip(th0 + alpha * (d.mean - c.mean), 0.0, 180.0))
            q.angles[d.group] = (float(K), th0)
    return q


def pmf(d: Dist, edges: np.ndarray, kT: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Potential of mean force with the Jacobian removed (r^2 for bonds, sin(theta) for angles).

    Returns (bin centres, U) with min(U) = 0; empty bins are NaN.
    """
    h = d.histogram(edges)
    x = 0.5 * (edges[1:] + edges[:-1])
    jac = x ** 2 if d.kind == "bond" else np.sin(np.radians(x))
    with np.errstate(divide="ignore", invalid="ignore"):
        p = h / np.maximum(jac, 1e-12)
        u = -kT * np.log(p)
    u[~np.isfinite(u)] = np.nan
    if np.any(np.isfinite(u)):
        u -= np.nanmin(u)
    return x, u
