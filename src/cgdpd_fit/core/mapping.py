"""Map an all-atom trajectory onto CG bead positions."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..io.dump import iter_dump
from ..io.lammps_data import Box
from .model import CGModel
from .runner import Cancelled
from .topology import AATopology, Species, WholeMaker


@dataclass
class MappedTrajectory:
    x: np.ndarray                    # (n_frames, n_molecules, n_beads, 3) Angstrom
    steps: list[int] = field(default_factory=list)
    volumes: list[float] = field(default_factory=list)
    lengths: list[np.ndarray] = field(default_factory=list)     # box edge lengths per frame (A)

    @property
    def n_frames(self) -> int:
        return self.x.shape[0]

    @property
    def n_molecules(self) -> int:
        return self.x.shape[1]


@dataclass
class FrameSelection:
    start: int = 0
    stop: int | None = None
    stride: int = 1
    max_molecules: int = 0           # 0 = all molecules of the species

    def to_dict(self) -> dict:
        return {"start": self.start, "stop": self.stop, "stride": self.stride,
                "max_molecules": self.max_molecules}

    @classmethod
    def from_dict(cls, d: dict) -> "FrameSelection":
        return cls(int(d.get("start", 0)), d.get("stop"), int(d.get("stride", 1)),
                   int(d.get("max_molecules", 0)))


def species_rows(sp: Species, max_molecules: int = 0, seed: int = 0) -> np.ndarray:
    rows = sp.rows()
    if max_molecules and max_molecules < len(rows):
        rng = np.random.default_rng(seed)
        rows = rows[np.sort(rng.choice(len(rows), max_molecules, replace=False))]
    return rows


def map_frames(frames, topo: AATopology, sp: Species, model: CGModel, rows: np.ndarray,
               progress=None, should_stop=None, n_total: int | None = None) -> MappedTrajectory:
    """Map an iterable of (box, positions (n_mol, n_atoms, 3), step) tuples."""
    W = model.weights(sp, topo.masses[rows[0]])
    whole = WholeMaker.for_species(sp)
    xs, steps, vols, lens = [], [], [], []
    for k, (box, pos, step) in enumerate(frames):
        if should_stop and should_stop():
            raise Cancelled()
        x = whole.apply(pos, box)
        xs.append(np.einsum("ba,mak->mbk", W, x).astype(np.float32))
        steps.append(int(step))
        vols.append(float(abs(np.linalg.det(box.h_matrix()))))
        lens.append(np.array(box.lengths, dtype=float))
        if progress:
            progress(k + 1, n_total)
    if not xs:
        raise ValueError("no frames selected")
    return MappedTrajectory(np.stack(xs), steps, vols, lens)


def map_lammps_dump(path: str, topo: AATopology, sp: Species, model: CGModel,
                    sel: FrameSelection, progress=None, should_stop=None,
                    n_total: int | None = None) -> MappedTrajectory:
    rows = species_rows(sp, sel.max_molecules)
    ids = topo.ids[rows.ravel()]
    shape = rows.shape

    def frames():
        for fr in iter_dump(path, ids=ids, start=sel.start, stop=sel.stop, stride=sel.stride):
            yield fr.box, fr.pos.reshape(shape + (3,)), fr.step

    return map_frames(frames(), topo, sp, model, rows, progress, should_stop, n_total)


def reference_beads(topo: AATopology, sp: Species, model: CGModel, instance: int = 0,
                    box: Box | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(whole atom coordinates, bead coordinates) of one molecule from the topology file."""
    rows = sp.instances[instance]
    x = WholeMaker.for_species(sp).apply(topo.pos[rows][None], box or topo.box)[0]
    W = model.weights(sp, topo.masses[rows])
    return x, W @ x
