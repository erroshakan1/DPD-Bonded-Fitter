"""Optional GROMACS input (``.tpr`` + ``.xtc``/``.trr``) through MDAnalysis.

MDAnalysis returns Angstrom, like LAMMPS ``real`` units, so the rest of the pipeline is the same.
"""
from __future__ import annotations

import numpy as np

from ..core.topology import AATopology, element_from_mass
from .lammps_data import Box


def _universe(tpr: str, traj: str | None = None):
    try:
        import MDAnalysis as mda
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("GROMACS input needs MDAnalysis: pip install MDAnalysis") from exc
    return mda.Universe(tpr, traj) if traj else mda.Universe(tpr)


def _box(dimensions) -> Box:
    from MDAnalysis.lib.mdamath import triclinic_vectors
    if dimensions is None:                     # topology without coordinates
        return Box(np.zeros(3), np.ones(3))
    v = triclinic_vectors(dimensions)          # rows a, b, c (LAMMPS-compatible lower triangle)
    lengths = np.array([v[0, 0], v[1, 1], v[2, 2]])
    tilt = np.array([v[1, 0], v[2, 0], v[2, 1]])
    return Box(np.zeros(3), lengths, tilt)


def load_topology(tpr: str, traj: str | None = None) -> AATopology:
    u = _universe(tpr, traj)
    a = u.atoms
    n = len(a)
    try:
        mol = np.asarray(a.molnums, dtype=np.int64) + 1
    except (AttributeError, Exception):  # noqa: BLE001 - not every topology has molnums
        mol = np.asarray(a.fragindices, dtype=np.int64) + 1
    names = list(a.types) if hasattr(a, "types") else list(a.names)
    tmap: dict[str, int] = {}
    types = np.array([tmap.setdefault(t, len(tmap) + 1) for t in names], dtype=np.int64)
    masses = np.asarray(a.masses, float)
    try:
        elements = [e if e else element_from_mass(m) for e, m in zip(a.elements, masses)]
    except Exception:  # noqa: BLE001
        elements = [element_from_mass(m) for m in masses]
    elements = [e.capitalize() if len(e) > 1 else e.upper() for e in elements]
    bonds = np.asarray(u.bonds.indices, dtype=np.int64).reshape(-1, 2) if hasattr(u, "bonds") else np.zeros((0, 2), int)
    charges = np.asarray(a.charges, float) if hasattr(a, "charges") else np.zeros(n)
    try:
        pos = a.positions.astype(float)
        dims = u.dimensions
    except Exception:  # noqa: BLE001 - MDAnalysis NoDataError: tpr read without a trajectory
        pos, dims = np.zeros((n, 3)), None
    return AATopology(ids=np.arange(1, n + 1), mol=mol, types=types,
                      type_names={v: k for k, v in tmap.items()}, masses=masses, charges=charges,
                      elements=elements, bonds=bonds, pos=pos, box=_box(dims), source=tpr,
                      names=[str(x) for x in a.names])


def count_frames(tpr: str, traj: str) -> int:
    return len(_universe(tpr, traj).trajectory)


def iter_frames(tpr: str, traj: str, rows: np.ndarray, start: int = 0, stop: int | None = None,
                stride: int = 1):
    """Yield (box, positions (n_mol, n_atoms, 3), frame index, time in ps) for the atom ``rows``."""
    u = _universe(tpr, traj)
    shape = rows.shape
    flat = rows.ravel()
    for ts in u.trajectory[start:stop:max(1, stride)]:
        yield _box(ts.dimensions), u.atoms.positions[flat].reshape(shape + (3,)).astype(float), ts.frame, float(ts.time)
