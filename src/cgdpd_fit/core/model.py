"""Coarse-grained model of one molecule species: beads, bonds, angles and their type groups.

Bonds and angles that share a *group* label share one LAMMPS bond/angle type and are fitted
together (all their instances are pooled into one distribution). The default group label
is built from the bead types, so chemically equivalent bonds (e.g. the two tails of a lipid)
end up in the same group; edit the labels to split or merge groups.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .topology import Species

CENTERS = ("com", "com_heavy", "cog_heavy")
CENTER_LABELS = {"com": "center of mass (incl. H)", "com_heavy": "heavy-atom center of mass",
                 "cog_heavy": "heavy-atom centroid"}


@dataclass
class Bead:
    name: str
    type: str
    atoms: list[int] = field(default_factory=list)     # local atom indices (incl. hydrogens)


@dataclass
class CGModel:
    beads: list[Bead] = field(default_factory=list)
    bonds: list[tuple[int, int]] = field(default_factory=list)
    bond_groups: list[str] = field(default_factory=list)
    angles: list[tuple[int, int, int]] = field(default_factory=list)
    angle_groups: list[str] = field(default_factory=list)
    center: str = "com"

    @property
    def n_beads(self) -> int:
        return len(self.beads)

    @property
    def bead_types(self) -> list[str]:
        """Distinct bead types in order of first appearance."""
        out: list[str] = []
        for b in self.beads:
            if b.type not in out:
                out.append(b.type)
        return out

    # ------------------------------------------------------------- mapping
    def owner(self, n_atoms: int) -> np.ndarray:
        """Bead index of every local atom (-1 = unmapped)."""
        own = np.full(n_atoms, -1, dtype=int)
        for k, b in enumerate(self.beads):
            own[b.atoms] = k
        return own

    def assign(self, bead: int, atoms, sp: Species, with_hydrogens: bool = True) -> None:
        """Move ``atoms`` (local indices) to ``bead``; hydrogens follow their heavy atom."""
        atoms = set(int(a) for a in atoms)
        if with_hydrogens:
            atoms |= {h for a in list(atoms) for h in _hydrogens_of(sp, a)}
        for b in self.beads:
            b.atoms = [a for a in b.atoms if a not in atoms]
        self.beads[bead].atoms = sorted(set(self.beads[bead].atoms) | atoms)

    def toggle(self, bead: int, atom: int, sp: Species) -> None:
        group = {atom, *_hydrogens_of(sp, atom)}
        if atom in self.beads[bead].atoms:
            self.beads[bead].atoms = [a for a in self.beads[bead].atoms if a not in group]
        else:
            self.assign(bead, group, sp, with_hydrogens=False)

    def weights(self, sp: Species, masses: np.ndarray) -> np.ndarray:
        """(n_beads, n_atoms) matrix W so that bead positions are W @ atom positions.

        ``masses`` are the masses of the local atoms of the species.
        """
        W = np.zeros((self.n_beads, sp.n_atoms))
        heavy = np.array([e != "H" for e in sp.elements])
        for k, b in enumerate(self.beads):
            idx = np.array(b.atoms, dtype=int)
            if len(idx) == 0:
                continue
            if self.center == "com":
                w = masses[idx]
            elif self.center == "com_heavy":
                w = masses[idx] * heavy[idx]
            else:
                w = heavy[idx].astype(float)
            if w.sum() <= 0:            # bead of hydrogens only: fall back to its centroid
                w = np.ones(len(idx))
            W[k, idx] = w / w.sum()
        return W

    def bead_charges(self, sp: Species, charges: np.ndarray) -> np.ndarray:
        """Bead charges; unmapped atoms (e.g. hydrogens left out of a heavy-atom mapping)
        give their charge to the bead they are bonded to, so the molecule keeps its net charge."""
        own = self.owner(sp.n_atoms)
        adj = defaultdict(list)
        for a, b in sp.bonds:
            adj[a].append(b)
            adj[b].append(a)
        queue = [a for a in range(sp.n_atoms) if own[a] >= 0]
        for a in queue:                         # breadth-first growth of the beads
            for b in adj[a]:
                if own[b] < 0:
                    own[b] = own[a]
                    queue.append(b)
        q = np.zeros(self.n_beads)
        for a, k in enumerate(own):
            if k >= 0:
                q[k] += charges[a]
        return q

    # ------------------------------------------------------------ topology
    def bonds_from_atoms(self, sp: Species) -> None:
        """CG bonds wherever an all-atom bond connects two different beads."""
        own = self.owner(sp.n_atoms)
        pairs = set()
        for a, b in sp.bonds:
            i, j = own[a], own[b]
            if i >= 0 and j >= 0 and i != j:
                pairs.add((min(i, j), max(i, j)))
        self.bonds = sorted((int(i), int(j)) for i, j in pairs)
        self.regroup_bonds()

    def angles_from_bonds(self) -> None:
        nb = defaultdict(set)
        for i, j in self.bonds:
            nb[i].add(j)
            nb[j].add(i)
        out = []
        for c in range(self.n_beads):
            ns = sorted(nb[c])
            for x in range(len(ns)):
                for y in range(x + 1, len(ns)):
                    out.append((ns[x], c, ns[y]))
        self.angles = out
        self.regroup_angles()

    def bond_label(self, bond: tuple[int, int]) -> str:
        t = sorted((self.beads[bond[0]].type, self.beads[bond[1]].type))
        return "-".join(t)

    def angle_label(self, angle: tuple[int, int, int]) -> str:
        a, c, b = (self.beads[i].type for i in angle)
        a, b = sorted((a, b))
        return f"{a}-{c}-{b}"

    def regroup_bonds(self) -> None:
        self.bond_groups = [self.bond_label(b) for b in self.bonds]

    def regroup_angles(self) -> None:
        self.angle_groups = [self.angle_label(a) for a in self.angles]

    def bond_group_names(self) -> list[str]:
        return list(dict.fromkeys(self.bond_groups))

    def angle_group_names(self) -> list[str]:
        return list(dict.fromkeys(self.angle_groups))

    def bonds_in(self, group: str) -> list[tuple[int, int]]:
        return [b for b, g in zip(self.bonds, self.bond_groups) if g == group]

    def angles_in(self, group: str) -> list[tuple[int, int, int]]:
        return [a for a, g in zip(self.angles, self.angle_groups) if g == group]

    # ---------------------------------------------------------- validation
    def validate(self, sp: Species) -> list[str]:
        msgs = []
        if not self.beads:
            return ["no beads defined"]
        empty = [b.name for b in self.beads if not b.atoms]
        if empty:
            msgs.append("empty beads: " + ", ".join(empty))
        names = [b.name for b in self.beads]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            msgs.append("duplicate bead names: " + ", ".join(dup))
        own = self.owner(sp.n_atoms)
        un = [k for k in np.flatnonzero(own < 0) if sp.elements[k] != "H"]
        if un:
            msgs.append(f"{len(un)} heavy atoms are not mapped to any bead")
        if len(self.beads) > 1 and not self.bonds:
            msgs.append("no CG bonds")
        if len(self.bond_groups) != len(self.bonds) or len(self.angle_groups) != len(self.angles):
            msgs.append("group labels out of date")
        return msgs

    # ------------------------------------------------------------------ IO
    def to_dict(self) -> dict:
        return {
            "center": self.center,
            "beads": [{"name": b.name, "type": b.type, "atoms": list(map(int, b.atoms))} for b in self.beads],
            "bonds": [list(map(int, b)) for b in self.bonds],
            "bond_groups": list(self.bond_groups),
            "angles": [list(map(int, a)) for a in self.angles],
            "angle_groups": list(self.angle_groups),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CGModel":
        m = cls(
            beads=[Bead(b["name"], b.get("type", b["name"]), list(b.get("atoms", []))) for b in d.get("beads", [])],
            bonds=[tuple(b) for b in d.get("bonds", [])],
            bond_groups=list(d.get("bond_groups", [])),
            angles=[tuple(a) for a in d.get("angles", [])],
            angle_groups=list(d.get("angle_groups", [])),
            center=d.get("center", "com"),
        )
        if len(m.bond_groups) != len(m.bonds):
            m.regroup_bonds()
        if len(m.angle_groups) != len(m.angles):
            m.regroup_angles()
        return m


def _hydrogens_of(sp: Species, atom: int) -> list[int]:
    if sp.elements[atom] == "H":
        return []
    out = []
    for a, b in sp.bonds:
        if a == atom and sp.elements[b] == "H":
            out.append(b)
        elif b == atom and sp.elements[a] == "H":
            out.append(a)
    return out


def auto_chain_model(sp: Species, n_beads: int, prefix: str = "B") -> CGModel:
    """Split the longest heavy-atom path into ``n_beads`` consecutive groups.

    Side branches are assigned to the bead of the backbone atom they hang from.
    A starting proposal for chain-like molecules; edit it afterwards.
    """
    heavy = [k for k, e in enumerate(sp.elements) if e != "H"]
    hs = set(heavy)
    adj = defaultdict(list)
    for a, b in sp.bonds:
        if a in hs and b in hs:
            adj[a].append(b)
            adj[b].append(a)

    def bfs(src):
        dist = {src: 0}
        prev = {src: None}
        q = [src]
        for a in q:
            for b in adj[a]:
                if b not in dist:
                    dist[b] = dist[a] + 1
                    prev[b] = a
                    q.append(b)
        far = max(dist, key=dist.get)
        return far, prev, dist

    if not heavy:
        heavy = list(range(sp.n_atoms))
        hs = set(heavy)
    a, _, _ = bfs(heavy[0])
    b, prev, _ = bfs(a)
    path = []
    x = b
    while x is not None:
        path.append(x)
        x = prev[x]
    n_beads = max(1, min(n_beads, len(path)))
    chunks = np.array_split(np.arange(len(path)), n_beads)
    model = CGModel(beads=[Bead(f"{prefix}{k + 1}", f"{prefix}{k + 1}") for k in range(n_beads)])
    bead_of = {}
    for k, ch in enumerate(chunks):
        for i in ch:
            bead_of[path[i]] = k
    # side branches: BFS outwards from the backbone
    q = list(bead_of)
    for x in q:
        for y in adj[x]:
            if y not in bead_of:
                bead_of[y] = bead_of[x]
                q.append(y)
    for atom, k in bead_of.items():
        model.beads[k].atoms.append(atom)
    for b in model.beads:
        b.atoms = sorted(set(b.atoms) | {h for a in b.atoms for h in _hydrogens_of(sp, a)})
    model.bonds_from_atoms(sp)
    model.angles_from_bonds()
    return model
