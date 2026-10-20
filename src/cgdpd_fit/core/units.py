"""DPD reduced units.

Lengths in the all-atom trajectory are Angstrom (LAMMPS ``real``/``metal`` units, or
GROMACS read through MDAnalysis). The DPD length unit is the cutoff ``r_c``; energies are
in units of ``kT`` (= 1 in the generated LAMMPS input).

With the usual Groot-Rabone mapping, one DPD bead holds ``N_m`` water molecules and the
bead number density is ``rho`` (typically 3), so

    r_c = (rho * N_m * V_w) ** (1/3),   V_w ~ 30 A^3 (volume of one water molecule)

e.g. rho = 3, N_m = 3 gives r_c = 6.46 A.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

WATER_VOLUME_A3 = 30.0


def rc_from_mapping(n_m: float, rho: float = 3.0, v_w: float = WATER_VOLUME_A3) -> float:
    """DPD length unit in Angstrom for ``n_m`` water molecules per bead at density ``rho``."""
    return (rho * n_m * v_w) ** (1.0 / 3.0)


@dataclass
class DPDUnits:
    rc: float = 6.46          # Angstrom per DPD length unit
    kT: float = 1.0           # DPD energy unit (the thermostat temperature)

    def to_dpd(self, x_angstrom):
        return x_angstrom / self.rc

    def to_angstrom(self, x_dpd):
        return x_dpd * self.rc

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DPDUnits":
        return cls(**{k: d[k] for k in ("rc", "kT") if k in d})
