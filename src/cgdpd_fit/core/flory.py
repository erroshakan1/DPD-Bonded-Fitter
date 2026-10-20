"""DPD repulsion parameters from Hildebrand solubility parameters (Flory-Huggins route).

    chi_ij = V_bead (delta_i - delta_j)^2 / (R T)
    a_ij   = a_ii + f * chi_ij

* ``delta`` in (J/cm^3)^1/2 = MPa^1/2 (water ~ 47.8, alkanes ~ 15-16).
* ``V_bead`` is the molar volume of one bead, r_c^3 / rho (in cm^3/mol).
* ``a_ii`` follows from the compressibility of the solvent: 25 kT for N_m = 1 at rho = 3
  (Groot & Warren, J. Chem. Phys. 107, 4423 (1997)); about 78 kT for N_m = 3 (Groot & Rabone,
  Biophys. J. 81, 725 (2001)).
* ``f`` = 3.27 for rho = 3 (Groot & Madden, J. Chem. Phys. 108, 8713 (1998)); Groot & Warren
  found chi = 0.286 Delta a, i.e. f = 3.50, and the factor depends on the density.
"""
from __future__ import annotations

R_GAS = 8.314462618          # J / (mol K)
AVOGADRO = 6.02214076e23
DEFAULT_FACTOR = 3.27


def bead_molar_volume(rc_angstrom: float, rho: float) -> float:
    """Molar volume of one bead in cm^3/mol for a bead number density ``rho`` (per r_c^3)."""
    v_a3 = rc_angstrom ** 3 / rho
    return v_a3 * 1e-24 * AVOGADRO


def chi(delta_i: float, delta_j: float, v_cm3: float, temperature: float = 298.15) -> float:
    return v_cm3 * (delta_i - delta_j) ** 2 / (R_GAS * temperature)


def aij_from_solubility(delta: dict[str, float], v_cm3: float, temperature: float = 298.15,
                        a_ii: float = 25.0, factor: float = DEFAULT_FACTOR) -> tuple[dict, dict]:
    """(a_ij, chi_ij) for all pairs of types with a solubility parameter; keys ``"T1 T2"``."""
    types = list(delta)
    a, x = {}, {}
    for i, t1 in enumerate(types):
        for t2 in types[i:]:
            c = chi(delta[t1], delta[t2], v_cm3, temperature)
            a[f"{t1} {t2}"] = a_ii + factor * c
            x[f"{t1} {t2}"] = c
    return a, x
