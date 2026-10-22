"""Bonded parameters, Boltzmann-inversion initial guesses and iterative updates.

Conventions follow LAMMPS (no factor 1/2):

* ``bond_style harmonic``         E = K (r - r0)^2            K in kT / r_c^2, r0 in r_c
* ``angle_style harmonic``        E = K (theta - theta0)^2    K in kT / rad^2, theta0 in degrees
* ``angle_style cosine/squared``  E = K (cos theta - cos theta0)^2
* ``angle_style table``           U(theta) tabulated on a 1 degree grid, in kT; starts from the
  Boltzmann inversion U = -kT ln(P(theta)/sin theta) of the all-atom distribution and is refined
  by iterative Boltzmann inversion, U <- U + alpha kT ln(P_CG/P_AA). Reproduces multi-modal
  angle distributions that no single harmonic well can.

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

ANGLE_STYLES = ("harmonic", "cosine/squared", "table")
TABLE_GRID = np.linspace(0.0, 180.0, 181)       # degrees; LAMMPS tables must span 0-180
TABLE_U_MAX = 25.0                              # kT, cap of the extrapolated walls
TABLE_SMOOTH = 2.0                              # degrees, Gaussian smoothing of the densities


@dataclass
class BondedParams:
    """K and equilibrium value per group. Bonds: (K, r0[r_c]); angles: (K, theta0[deg]).

    With ``angle_style == "table"`` the angles are not in ``angles`` (so the swarm only sees
    the bonds) but in ``tables``: U [kT] on ``TABLE_GRID`` per group, updated by IBI."""

    bonds: dict[str, tuple[float, float]] = field(default_factory=dict)
    angles: dict[str, tuple[float, float]] = field(default_factory=dict)
    angle_style: str = "harmonic"
    tables: dict[str, np.ndarray] = field(default_factory=dict)

    def copy(self) -> "BondedParams":
        return BondedParams(dict(self.bonds), dict(self.angles), self.angle_style,
                            {g: u.copy() for g, u in self.tables.items()})

    def to_dict(self) -> dict:
        d = {"angle_style": self.angle_style,
             "bonds": {g: list(v) for g, v in self.bonds.items()},
             "angles": {g: list(v) for g, v in self.angles.items()}}
        if self.tables:
            d["tables"] = {g: [round(float(x), 5) for x in u] for g, u in self.tables.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BondedParams":
        return cls({g: tuple(v) for g, v in d.get("bonds", {}).items()},
                   {g: tuple(v) for g, v in d.get("angles", {}).items()},
                   d.get("angle_style", "harmonic"),
                   {g: np.asarray(u, float) for g, u in d.get("tables", {}).items()})

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
        elif d.kind == "angle" and angle_style == "table":
            p.tables[d.group] = table_from_pmf(angle_density(d.samples), kT)
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
        elif d.kind == "angle" and p.angle_style == "table":
            q.tables[d.group] = table_ibi_update(p.tables[d.group], d.samples, c.samples,
                                                 units.kT, alpha)
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


# ------------------------------------------------------------- tabulated angles
def angle_density(samples: np.ndarray) -> np.ndarray:
    """Smoothed probability density of angles (degrees) on ``TABLE_GRID``."""
    edges = np.append(TABLE_GRID - 0.5, 180.5)
    h, _ = np.histogram(np.asarray(samples, float), bins=edges)
    x = np.arange(-4 * TABLE_SMOOTH, 4 * TABLE_SMOOTH + 1)
    k = np.exp(-0.5 * (x / TABLE_SMOOTH) ** 2)
    h = np.convolve(np.pad(h.astype(float), len(x) // 2, mode="reflect"), k / k.sum(), mode="valid")
    return h / max(h.sum(), 1e-300)


def table_from_pmf(dens: np.ndarray, kT: float = 1.0, rel_floor: float = 1e-3) -> np.ndarray:
    """U(theta) = -kT ln(P/sin theta) where the angle is sampled; outside, linear walls
    that rise towards 0 and 180 degrees (capped at ``TABLE_U_MAX``)."""
    sin = np.maximum(np.sin(np.radians(TABLE_GRID)), 1e-3)
    ok = dens > rel_floor * dens.max()
    u = np.full(len(TABLE_GRID), np.nan)
    u[ok] = -kT * np.log(dens[ok] / sin[ok])
    i0, i1 = np.flatnonzero(ok)[[0, -1]]
    u[i0:i1 + 1] = np.interp(TABLE_GRID[i0:i1 + 1], TABLE_GRID[ok], u[ok])   # interior gaps
    u -= np.nanmin(u)
    n = 4                                   # slope of the edge, at least 0.2 kT per degree
    lo = max(0.2 * kT, -np.polyfit(TABLE_GRID[i0:i0 + n], u[i0:i0 + n], 1)[0]) if i1 - i0 >= n else 0.2 * kT
    hi = max(0.2 * kT, np.polyfit(TABLE_GRID[i1 - n + 1:i1 + 1], u[i1 - n + 1:i1 + 1], 1)[0]) if i1 - i0 >= n else 0.2 * kT
    u[:i0] = u[i0] + lo * (TABLE_GRID[i0] - TABLE_GRID[:i0])
    u[i1 + 1:] = u[i1] + hi * (TABLE_GRID[i1 + 1:] - TABLE_GRID[i1])
    return np.minimum(u, TABLE_U_MAX * kT)


def table_ibi_update(u: np.ndarray, aa: np.ndarray, cg: np.ndarray, kT: float = 1.0,
                     alpha: float = 0.7, max_step: float = 2.0, rel_floor: float = 1e-3) -> np.ndarray:
    """U <- U + alpha kT ln(P_CG / P_AA). The floor keeps the correction finite and lets it
    fade where neither system samples the angle; a step is limited to ``max_step`` kT."""
    pa, pc = angle_density(aa), angle_density(cg)
    eps = rel_floor * pa.max()
    du = alpha * kT * np.log((pc + eps) / (pa + eps))
    u = u + np.clip(du, -max_step * kT, max_step * kT)
    u -= u.min()
    return np.minimum(u, TABLE_U_MAX * kT)


def table_text(tables: dict[str, np.ndarray], names: dict[str, str]) -> str:
    """LAMMPS ``angle_style table`` file: one section per group (keyword ``names[group]``);
    the 4th column is -dU/dtheta per degree, as LAMMPS expects."""
    out = ["# tabulated angle potentials written by cgdpd-fit (energy in kT, angle in degrees)"]
    for g, u in tables.items():
        f = -np.gradient(u, TABLE_GRID)
        out += ["", f"{names[g]}  # {g}", f"N {len(TABLE_GRID)}", ""]
        out += [f"{i} {t:.1f} {e:.6f} {fi:.6f}" for i, (t, e, fi) in enumerate(zip(TABLE_GRID, u, f), 1)]
    return "\n".join(out) + "\n"
