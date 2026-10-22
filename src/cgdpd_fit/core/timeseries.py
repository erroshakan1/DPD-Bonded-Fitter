"""Time series of the radius of gyration and the solvent accessible surface (SASA).

Both are computed from bead positions, for the mapped all-atom trajectory and for the CG
runs alike, so the two curves compare conformations and not resolutions. Every value is
the mean over the molecules of a frame.

The SASA is the Shrake-Rupley surface of every molecule on its own (as ``gmx sasa`` of one
molecule): beads are spheres of radius ``bead_radius`` (by default that of a sphere with the
volume of one bead, ``r_c^3 / rho``) and the probe is a water molecule (1.4 A).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .distributions import radius_of_gyration

PROBE_RADIUS = 1.4          # Angstrom
SPHERE_POINTS = 96
MAX_SASA_MOLECULES = 20     # SASA of at most this many molecules per frame (cost)
MAX_FRAMES = 1000           # longer trajectories are thinned evenly


@dataclass
class TimeSeries:
    time: np.ndarray        # ps for all-atom, tau for CG
    rg: np.ndarray          # Angstrom, mean over molecules
    sasa: np.ndarray        # nm^2, mean over molecules
    unit: str = "ps"

    def to_csv(self, path) -> None:
        np.savetxt(path, np.column_stack([self.time, self.rg, self.sasa]), fmt="%.6g", delimiter=",",
                   header=f"time [{self.unit}],Rg [A],SASA [nm^2]", comments="")

    def thin(self, n: int) -> "TimeSeries":
        if len(self.time) <= n:
            return self
        i = np.linspace(0, len(self.time) - 1, n).round().astype(int)
        return TimeSeries(self.time[i], self.rg[i], self.sasa[i], self.unit)


def bead_radius(rc: float, density: float) -> float:
    """Radius (A) of a sphere with the volume of one bead, r_c^3 / rho."""
    return (3.0 * rc ** 3 / (4.0 * np.pi * density)) ** (1.0 / 3.0)


def _sphere(n: int = SPHERE_POINTS) -> np.ndarray:
    """``n`` nearly uniform unit vectors (golden spiral)."""
    k = np.arange(n) + 0.5
    z = 1.0 - 2.0 * k / n
    r = np.sqrt(1.0 - z * z)
    phi = np.pi * (3.0 - np.sqrt(5.0)) * k
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def sasa(x: np.ndarray, radius: float, probe: float = PROBE_RADIUS) -> np.ndarray:
    """Shrake-Rupley SASA (A^2) of every molecule in ``x`` (..., n_beads, 3), equal radii."""
    shape = x.shape[:-2]
    x = x.reshape((-1,) + x.shape[-2:]).astype(float)
    R = radius + probe
    pts = _sphere()
    out = np.empty(len(x))
    nb = x.shape[1]
    chunk = max(1, int(1e6 // (nb * nb * len(pts))))
    for s in range(0, len(x), chunk):
        xc = x[s:s + chunk]                                         # (m, b, 3)
        p = xc[:, :, None, :] + R * pts[None, None]                 # (m, b, P, 3)
        d2 = ((p[:, :, :, None, :] - xc[:, None, None, :, :]) ** 2).sum(-1)   # (m, b, P, b)
        d2[:, np.arange(nb), :, np.arange(nb)] = np.inf             # a bead does not bury itself
        free = (d2 >= R * R).all(-1)                                # (m, b, P)
        out[s:s + chunk] = free.sum((1, 2)) * (4.0 * np.pi * R * R / len(pts))
    return out.reshape(shape)


def time_series(x: np.ndarray, time: np.ndarray, radius: float, unit: str = "ps",
                max_molecules: int = MAX_SASA_MOLECULES) -> TimeSeries:
    """x: (n_frames, n_molecules, n_beads, 3) Angstrom."""
    if len(x) > MAX_FRAMES:
        i = np.linspace(0, len(x) - 1, MAX_FRAMES).round().astype(int)
        x, time = x[i], np.asarray(time)[i]
    rg = radius_of_gyration(x).reshape(x.shape[:2]).mean(1)
    xs = x[:, :max_molecules] if max_molecules else x
    area = sasa(xs, radius).mean(1) / 100.0
    return TimeSeries(np.asarray(time, float), rg, area, unit)
