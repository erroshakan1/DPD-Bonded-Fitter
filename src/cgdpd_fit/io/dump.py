"""Streaming reader for LAMMPS text dump files (``*.lammpstrj``, optionally gzipped).

Reference: https://docs.lammps.org/dump.html

Positions are taken from the best available columns, in this order:

* ``xu yu zu``   unwrapped Cartesian
* ``x y z``      wrapped Cartesian (+ ``ix iy iz`` image flags if present)
* ``xsu ysu zsu`` / ``xs ys zs``  scaled (fractional) coordinates

Molecules are made whole later along their bonds (see :mod:`cgdpd_fit.core.topology`), so
wrapped coordinates without image flags work as well.
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .lammps_data import Box


class DumpError(ValueError):
    pass


@dataclass
class Frame:
    step: int
    box: Box
    ids: np.ndarray        # (n,) atom IDs, in the order of ``pos``
    pos: np.ndarray        # (n, 3) Cartesian coordinates
    unwrapped: bool        # True if ``pos`` is already unwrapped (xu or x + image flags)


@dataclass
class DumpInfo:
    path: str
    n_frames: int
    n_atoms: int
    columns: list[str]
    first_step: int
    last_step: int

    @property
    def position_kind(self) -> str:
        c = set(self.columns)
        if {"xu", "yu", "zu"} <= c:
            return "unwrapped (xu yu zu)"
        if {"x", "y", "z"} <= c:
            return "wrapped + image flags" if {"ix", "iy", "iz"} <= c else "wrapped (x y z)"
        if {"xsu", "ysu", "zsu"} <= c:
            return "scaled unwrapped (xsu)"
        if {"xs", "ys", "zs"} <= c:
            return "scaled (xs ys zs)"
        return "no coordinates"


def _open(path: str | Path):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def _parse_box(header: str, lines: list[str]) -> Box:
    """``ITEM: BOX BOUNDS [xy xz yz] pp pp pp`` followed by three lines."""
    vals = [[float(t) for t in ln.split()] for ln in lines]
    triclinic = "xy" in header.split()
    if not triclinic:
        lo = np.array([v[0] for v in vals])
        hi = np.array([v[1] for v in vals])
        return Box(lo, hi)
    (xlob, xhib, xy), (ylob, yhib, xz), (zlo, zhi, yz) = vals
    xlo = xlob - min(0.0, xy, xz, xy + xz)
    xhi = xhib - max(0.0, xy, xz, xy + xz)
    ylo = ylob - min(0.0, yz)
    yhi = yhib - max(0.0, yz)
    return Box(np.array([xlo, ylo, zlo]), np.array([xhi, yhi, zhi]), np.array([xy, xz, yz]))


def _frames_raw(path: str | Path) -> Iterator[tuple[int, str, list[str], list[str], callable]]:
    """Yield (step, box header, box lines, columns, read_atoms) per frame.

    ``read_atoms(parse: bool)`` must be called exactly once; it returns the atom lines
    (or None when ``parse`` is False, in which case they are skipped cheaply).
    """
    with _open(path) as fh:
        while True:
            line = fh.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                if line.strip():
                    raise DumpError(f"{path}: expected 'ITEM: TIMESTEP', got {line.strip()!r}")
                continue
            step = int(fh.readline().split()[0])
            if not fh.readline().startswith("ITEM: NUMBER OF ATOMS"):
                raise DumpError(f"{path}: malformed frame at step {step}")
            n = int(fh.readline())
            box_header = fh.readline()
            if not box_header.startswith("ITEM: BOX BOUNDS"):
                raise DumpError(f"{path}: missing BOX BOUNDS at step {step}")
            box_lines = [fh.readline() for _ in range(3)]
            atoms_header = fh.readline()
            if not atoms_header.startswith("ITEM: ATOMS"):
                raise DumpError(f"{path}: missing ATOMS at step {step}")
            columns = atoms_header.split()[2:]
            done = [False]

            def read_atoms(parse: bool, _n=n, _done=done):
                _done[0] = True
                if parse:
                    return [fh.readline() for _ in range(_n)]
                for _ in range(_n):
                    fh.readline()
                return None

            yield step, box_header, box_lines, columns, read_atoms
            if not done[0]:
                read_atoms(False)


def scan_dump(path: str | Path) -> DumpInfo:
    """Count frames and read the column layout without parsing coordinates."""
    n_frames = 0
    n_atoms = 0
    columns: list[str] = []
    first = last = 0
    for step, _bh, _bl, cols, read_atoms in _frames_raw(path):
        if n_frames == 0:
            columns = cols
            first = step
            lines = read_atoms(True)
            n_atoms = len(lines)
        n_frames += 1
        last = step
    if n_frames == 0:
        raise DumpError(f"{path}: no frames found")
    return DumpInfo(str(path), n_frames, n_atoms, columns, first, last)


def _position_columns(columns: list[str]) -> tuple[list[int], str]:
    idx = {c: k for k, c in enumerate(columns)}

    def have(*names):
        return all(n in idx for n in names)

    if have("xu", "yu", "zu"):
        return [idx["xu"], idx["yu"], idx["zu"]], "unwrapped"
    if have("x", "y", "z"):
        if have("ix", "iy", "iz"):
            return [idx["x"], idx["y"], idx["z"], idx["ix"], idx["iy"], idx["iz"]], "image"
        return [idx["x"], idx["y"], idx["z"]], "wrapped"
    if have("xsu", "ysu", "zsu"):
        return [idx["xsu"], idx["ysu"], idx["zsu"]], "scaled_unwrapped"
    if have("xs", "ys", "zs"):
        return [idx["xs"], idx["ys"], idx["zs"]], "scaled"
    raise DumpError(f"dump has no coordinate columns (columns: {' '.join(columns)})")


def iter_dump(path: str | Path, ids: np.ndarray | None = None, start: int = 0,
              stop: int | None = None, stride: int = 1) -> Iterator[Frame]:
    """Iterate over frames ``start:stop:stride``.

    If ``ids`` is given, only these atoms are returned, in exactly that order; a frame
    missing any of them raises :class:`DumpError`.
    """
    stride = max(1, int(stride))
    want = None if ids is None else np.asarray(ids, dtype=np.int64)
    for k, (step, box_header, box_lines, columns, read_atoms) in enumerate(_frames_raw(path)):
        if stop is not None and k >= stop:
            return
        if k < start or (k - start) % stride:
            continue
        box = _parse_box(box_header, box_lines)
        lines = read_atoms(True)
        if "id" not in columns:
            raise DumpError(f"{path}: dump needs an 'id' column")
        pcols, kind = _position_columns(columns)
        tok = np.array("".join(lines).split())
        try:
            tok = tok.reshape(len(lines), len(columns))
        except ValueError as exc:
            raise DumpError(f"{path}: inconsistent number of columns at step {step}") from exc
        fids = tok[:, columns.index("id")].astype(np.int64)
        if want is not None:
            order = np.argsort(fids)
            sorted_ids = fids[order]
            loc = np.searchsorted(sorted_ids, want)
            loc = np.clip(loc, 0, len(sorted_ids) - 1)
            if not np.array_equal(sorted_ids[loc], want):
                missing = want[sorted_ids[loc] != want][:5]
                raise DumpError(f"{path}: step {step} is missing atom IDs {missing.tolist()} ...")
            rows = order[loc]
            tok = tok[rows]
            fids = want
        xyz = tok[:, pcols[:3]].astype(float)
        h = box.h_matrix()
        if kind in ("scaled", "scaled_unwrapped"):
            xyz = box.lo + xyz @ h.T
        if kind == "image":
            img = tok[:, pcols[3:6]].astype(float)
            xyz = xyz + img @ h.T
        unwrapped = kind in ("unwrapped", "image", "scaled_unwrapped")
        yield Frame(step=step, box=box, ids=fids, pos=xyz, unwrapped=unwrapped)
