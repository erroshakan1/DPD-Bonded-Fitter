"""Import the mapping files of MD2CG v3 / cgdpd-fit 0.x.

* ``cg_map.ndx``    one ``[ bead ]`` group per bead with 1-based global atom indices of one molecule
* ``bonds_map.txt`` ``[type]`` followed by ``bead1 bead2`` lines (1-based bead numbers)
* ``angles_map.txt`` ``[type]`` followed by ``bead1 bead2 bead3`` lines (centre in the middle)

Bead names such as ``C_2`` give the bead type ``C``. Bond/angle groups keep the old type
numbers as labels (``b3``, ``a5``).
"""
from __future__ import annotations

import re
from pathlib import Path

from ..core.model import Bead, CGModel
from ..core.topology import AATopology, Species


def _sections(path: str | Path) -> list[tuple[str, list[list[str]]]]:
    out: list[tuple[str, list[list[str]]]] = []
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#")[0].split(";")[0].strip()
        if not line:
            continue
        m = re.fullmatch(r"\[\s*(.+?)\s*\]", line)
        if m:
            out.append((m.group(1), []))
        elif out:
            out[-1][1].append(line.split())
    return out


def read_legacy(topo: AATopology, ndx: str | Path, bonds_map: str | Path | None = None,
                angles_map: str | Path | None = None) -> tuple[int, CGModel]:
    """Return (species index, model). The NDX atoms must belong to one molecule."""
    groups = _sections(ndx)
    if not groups:
        raise ValueError(f"{ndx}: no [ group ] sections")
    beads_atoms = [[int(t) - 1 for row in rows for t in row] for _, rows in groups]
    first = beads_atoms[0][0]
    where = None
    for si, sp in enumerate(topo.species):
        for inst in sp.instances:
            pos = {int(r): k for k, r in enumerate(inst)}
            if first in pos:
                where = (si, sp, pos)
                break
        if where:
            break
    if where is None:
        raise ValueError(f"{ndx}: atom {first + 1} is not part of any molecule")
    si, sp, pos = where
    model = CGModel()
    for (name, _), atoms in zip(groups, beads_atoms):
        missing = [a + 1 for a in atoms if a not in pos]
        if missing:
            raise ValueError(f"{ndx}: bead {name} has atoms outside the molecule: {missing[:5]}")
        btype = re.sub(r"_\d+$", "", name)
        model.beads.append(Bead(name, btype, sorted(pos[a] for a in atoms)))
    if bonds_map:
        for tid, rows in _sections(bonds_map):
            for r in rows:
                model.bonds.append((int(r[0]) - 1, int(r[1]) - 1))
                model.bond_groups.append(f"b{tid}")
    else:
        model.bonds_from_atoms(sp)
    if angles_map:
        for tid, rows in _sections(angles_map):
            for r in rows:
                model.angles.append((int(r[0]) - 1, int(r[1]) - 1, int(r[2]) - 1))
                model.angle_groups.append(f"a{tid}")
    elif not bonds_map:
        model.angles_from_bonds()
    return si, model


def species_of(topo: AATopology, sp: Species) -> int:
    return topo.species.index(sp)
