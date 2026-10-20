"""All-atom topology: split the system into molecules and group identical molecules into species.

The topology is independent of the input format: it is built from a LAMMPS data file
(:func:`from_lammps_data`) or, optionally, from a GROMACS run input via MDAnalysis
(:mod:`cgdpd_fit.io.gromacs`).
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from ..io.lammps_data import Box, LammpsData

# element guess from the atomic mass (united-atom CH2/CH3/CH groups count as carbon)
_ELEMENTS = [(1.008, "H"), (2.014, "H"), (12.011, "C"), (13.019, "C"), (14.027, "C"),
             (15.035, "C"), (16.043, "C"), (14.007, "N"), (15.999, "O"), (18.998, "F"),
             (22.990, "Na"), (24.305, "Mg"), (30.974, "P"), (32.06, "S"), (35.45, "Cl"),
             (39.098, "K"), (40.078, "Ca"), (79.904, "Br"), (126.90, "I")]


def element_from_mass(m: float) -> str:
    if m <= 0:
        return "X"
    best = min(_ELEMENTS, key=lambda e: abs(e[0] - m))
    return best[1] if abs(best[0] - m) < 0.6 else "X"


@dataclass
class Species:
    """A set of topologically identical molecules."""

    key: tuple
    types: list[int]
    elements: list[str]
    bonds: list[tuple[int, int]]                       # local atom indices
    instances: list[np.ndarray] = field(default_factory=list)   # global atom rows per molecule
    mol_ids: list[int] = field(default_factory=list)
    name: str = ""

    @property
    def n_atoms(self) -> int:
        return len(self.types)

    @property
    def count(self) -> int:
        return len(self.instances)

    @property
    def formula(self) -> str:
        c = Counter(self.elements)
        order = [e for e in ("C", "H") if e in c] + sorted(e for e in c if e not in ("C", "H"))
        return "".join(f"{e}{c[e] if c[e] > 1 else ''}" for e in order)

    @property
    def is_water(self) -> bool:
        return sorted(self.elements) == ["H", "H", "O"]

    def rows(self) -> np.ndarray:
        """All global atom rows, molecule by molecule: shape (count, n_atoms)."""
        return np.array(self.instances, dtype=np.int64).reshape(self.count, self.n_atoms)


@dataclass
class AATopology:
    ids: np.ndarray            # (N,) atom IDs (sorted)
    mol: np.ndarray            # (N,) molecule IDs (0 = none)
    types: np.ndarray          # (N,)
    type_names: dict[int, str]
    masses: np.ndarray         # (N,)
    charges: np.ndarray        # (N,)
    elements: list[str]
    bonds: np.ndarray          # (M, 2) atom rows
    pos: np.ndarray            # (N, 3) reference coordinates
    box: Box
    source: str = ""
    names: list[str] = field(default_factory=list)      # atom names (GROMACS) or type labels
    species: list[Species] = field(default_factory=list)
    adj: list[list[int]] = field(default_factory=list)

    @property
    def n_atoms(self) -> int:
        return len(self.ids)

    def type_name(self, t: int) -> str:
        return self.type_names.get(int(t), str(int(t)))

    # -------------------------------------------------------------- species
    def build_species(self, split: str = "auto") -> list[Species]:
        n = self.n_atoms
        self.adj = [[] for _ in range(n)]
        for a, b in self.bonds:
            self.adj[a].append(int(b))
            self.adj[b].append(int(a))
        if split == "auto":
            split = "molid" if np.any(self.mol != 0) else "bonds"
        groups = self._groups_by_molid() if split == "molid" else self._groups_by_bonds()
        by_key: dict[tuple, Species] = {}
        for g in groups:
            local = {int(r): k for k, r in enumerate(g)}
            types = tuple(int(self.types[r]) for r in g)
            lb = sorted((local[int(r)], local[b]) for r in g for b in self.adj[r]
                        if b in local and local[int(r)] < local[b])
            key = (types, tuple(lb))
            sp = by_key.get(key)
            if sp is None:
                sp = Species(key=key, types=list(types), elements=[self.elements[r] for r in g],
                             bonds=[tuple(b) for b in lb])
                by_key[key] = sp
            sp.instances.append(np.asarray(g, dtype=np.int64))
            sp.mol_ids.append(int(self.mol[g[0]]))
        species = sorted(by_key.values(), key=lambda s: (-s.n_atoms * s.count, -s.n_atoms))
        for k, sp in enumerate(species, 1):
            sp.name = f"S{k}: {sp.formula}" + (" (water)" if sp.is_water else "")
        self.species = species
        return species

    def _groups_by_molid(self) -> list[np.ndarray]:
        zero = self.mol == 0
        order = np.lexsort((self.ids, self.mol))
        order = order[~zero[order]]
        cut = np.flatnonzero(np.diff(self.mol[order])) + 1
        groups = [g for g in np.split(order, cut) if len(g)]
        if zero.any():
            groups += [g for g in self._groups_by_bonds() if zero[g[0]]]
        return groups

    def _groups_by_bonds(self) -> list[np.ndarray]:
        seen = np.zeros(self.n_atoms, dtype=bool)
        out = []
        for s in range(self.n_atoms):
            if seen[s]:
                continue
            comp = []
            dq = deque([s])
            seen[s] = True
            while dq:
                a = dq.popleft()
                comp.append(a)
                for b in self.adj[a]:
                    if not seen[b]:
                        seen[b] = True
                        dq.append(b)
            out.append(np.array(sorted(comp, key=lambda r: self.ids[r]), dtype=np.int64))
        return out


def from_lammps_data(d: LammpsData) -> AATopology:
    row_of = {int(a): k for k, a in enumerate(d.ids)}
    b = d.bonds()
    bonds = np.array([[row_of[int(x)], row_of[int(y)]] for x, y in b[:, 1:3]],
                     dtype=np.int64).reshape(-1, 2)
    masses = np.array([d.masses.get(int(t), 0.0) for t in d.types])
    if np.any(masses <= 0):
        missing = sorted({int(t) for t, m in zip(d.types, masses) if m <= 0})
        raise ValueError(f"{d.path}: no mass for atom types {missing} (Masses section missing?)")
    elements = [element_from_mass(m) for m in masses]
    return AATopology(ids=d.ids, mol=d.mol, types=d.types, type_names=dict(d.type_labels),
                      masses=masses, charges=d.charges, elements=elements, bonds=bonds,
                      pos=d.pos, box=d.box, source=d.path,
                      names=[d.type_name(t) for t in d.types])


# ------------------------------------------------------------------ whole molecules
@dataclass
class WholeMaker:
    """Makes molecules of one species whole across periodic boundaries.

    The bond graph of a molecule is traversed breadth-first once; the edges are grouped by
    depth so that unwrapping all molecules of a frame needs one vectorised step per level.
    Atoms that cannot be reached through bonds are attached to atom 0 by the minimum image.
    """

    levels: list[tuple[np.ndarray, np.ndarray]]    # (parents, children) per BFS depth

    @classmethod
    def for_species(cls, sp: Species) -> "WholeMaker":
        return cls.from_bonds(sp.n_atoms, sp.bonds)

    @classmethod
    def from_bonds(cls, n: int, bonds) -> "WholeMaker":
        adj = defaultdict(list)
        for a, b in bonds:
            adj[a].append(b)
            adj[b].append(a)
        depth = np.full(n, -1)
        parent = np.full(n, -1)
        depth[0] = 0
        for s in range(n):
            if depth[s] >= 0 and s != 0:
                continue
            if s != 0:
                depth[s] = 1
                parent[s] = 0        # disconnected component: attach to atom 0
            dq = deque([s])
            while dq:
                a = dq.popleft()
                for b in adj[a]:
                    if depth[b] < 0:
                        depth[b] = depth[a] + 1
                        parent[b] = a
                        dq.append(b)
        levels = []
        for d in range(1, int(depth.max()) + 1 if n > 1 else 1):
            ch = np.flatnonzero(depth == d)
            if len(ch):
                levels.append((parent[ch], ch))
        return cls(levels)

    def apply(self, x: np.ndarray, box: Box) -> np.ndarray:
        """x: (n_mol, n_atoms, 3) -> whole molecules (atom 0 stays in place)."""
        x = np.array(x, dtype=float, copy=True)
        h = box.h_matrix()
        hinv = np.linalg.inv(h)
        for par, ch in self.levels:
            d = x[:, ch] - x[:, par]
            f = d @ hinv.T
            f -= np.round(f)
            x[:, ch] = x[:, par] + f @ h.T
        return x
