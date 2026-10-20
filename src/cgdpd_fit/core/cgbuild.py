"""Build the CG LAMMPS system (data + input) used to evaluate bonded parameters.

Three interaction styles are supported:

``dpd``      ``atom_style angle``, ``pair_style dpd`` (Groot-Warren).
``dpd_q``    ``atom_style full``, ``pair_style dpd/coul/slater/long`` + Ewald, for charged
             beads (bead charge = sum of its atomic charges). Requires LAMMPS >= 27Jun2024.
``mdpd``     ``atom_style hybrid angle mdpd``, ``pair_style hybrid/overlay mdpd/rhosum mdpd``
             and ``fix mvv/dpd`` (many-body DPD, LAMMPS package DPD-MESO).

The molecules either keep the arrangement of the all-atom system (``box_mode = "match"``:
bead positions of one mapped frame scaled by 1/r_c, optionally replicated n x n x n), or are
dropped at random positions with random orientations into a cubic box (``"custom"``).
Solvent beads fill the box up to the bead number density.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .model import CGModel
from .params import BondedParams
from .units import DPDUnits

STYLES = {"dpd": "DPD (atom_style angle)", "dpd_q": "charged DPD (atom_style full)",
          "mdpd": "many-body DPD (mDPD)"}
BJERRUM_WATER_A = 7.1        # Bjerrum length of water at 298 K, Angstrom


@dataclass
class CGSystemSettings:
    style: str = "dpd"
    box_mode: str = "match"          # "match" | "custom"
    replicate: int = 1               # match: n x n x n copies of the all-atom box
    frame: int = -1                  # match: mapped frame used for the start structure
    n_molecules: int = 50            # custom
    box_length: float = 10.0         # custom, DPD units
    density: float = 3.0             # beads per r_c^3 (incl. solvent)
    solvent: bool = True
    solvent_name: str = "W"
    neutralize: bool = True          # dpd_q: counter-ion beads replace solvent beads
    bead_mass: float = 1.0
    # DPD
    a_default: float = 25.0
    a_ij: dict = field(default_factory=dict)       # "T1 T2" -> a
    gamma: float = 4.5
    # Flory-Huggins helper of the pair table (core.flory): solubility parameter per type
    delta: dict = field(default_factory=dict)      # type -> MPa^1/2
    fh_temperature: float = 298.15
    fh_factor: float = 3.27
    # mDPD (A < 0 attraction, B > 0 density-dependent repulsion, rd its cutoff)
    A_default: float = -40.0
    B_default: float = 25.0
    A_ij: dict = field(default_factory=dict)
    B_ij: dict = field(default_factory=dict)
    rd: float = 0.75
    # electrostatics (dpd_q)
    slater_lambda: float = 0.25
    coul_cut: float = 3.0
    bjerrum: float = BJERRUM_WATER_A
    kspace_acc: float = 1e-4
    special_lj: tuple = (1.0, 1.0, 1.0)
    # run
    dt: float = 0.02
    n_start: int = 1000              # short run at dt/4 to relax the start structure
    n_equil: int = 5000
    n_prod: int = 50000
    dump_every: int = 100
    seed: int = 4928459

    def to_dict(self) -> dict:
        d = asdict(self)
        d["special_lj"] = list(self.special_lj)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CGSystemSettings":
        s = cls()
        for k, v in d.items():
            if hasattr(s, k):
                setattr(s, k, tuple(v) if k == "special_lj" else v)
        return s

    def pair_value(self, table: dict, default: float, t1: str, t2: str) -> float:
        for key in (f"{t1} {t2}", f"{t2} {t1}"):
            if key in table:
                return float(table[key])
        return float(default)


@dataclass
class BuiltSystem:
    """What the builder wrote; the analysis needs the atom IDs of the solute beads."""

    workdir: Path
    data_file: Path
    input_file: Path
    pair_file: Path
    n_molecules: int
    n_solvent: int
    n_ions: int
    box_length: np.ndarray
    solute_ids: np.ndarray           # (n_molecules, n_beads) atom IDs
    type_names: list[str]
    total_charge: float = 0.0
    warnings: list[str] = field(default_factory=list)


def _random_rotation(rng) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
                     [2*(b*c + a*d), a*a - b*b + c*c - d*d, 2*(c*d - a*b)],
                     [2*(b*d - a*c), 2*(c*d + a*b), a*a - b*b - c*c + d*d]])


def start_structure(mapped_x: np.ndarray, aa_box_lengths: np.ndarray, units: DPDUnits,
                    st: CGSystemSettings, n_beads: int) -> tuple[np.ndarray, np.ndarray]:
    """Bead coordinates (n_mol, n_beads, 3) in DPD units and the box edge lengths."""
    rng = np.random.default_rng(st.seed)
    if st.box_mode == "match":
        x = units.to_dpd(np.asarray(mapped_x[st.frame], dtype=float))     # (n_mol, n_beads, 3)
        L = units.to_dpd(np.asarray(aa_box_lengths, dtype=float))
        n = max(1, int(st.replicate))
        if n > 1:
            shifts = np.array([[i, j, k] for i in range(n) for j in range(n) for k in range(n)], float)
            x = np.concatenate([x + s * L for s in shifts])
            L = L * n
        return x, L
    # custom: random conformations from the mapped trajectory, random orientation and place
    L = np.full(3, float(st.box_length))
    nf, nm = mapped_x.shape[:2]
    out = np.empty((st.n_molecules, n_beads, 3))
    for k in range(st.n_molecules):
        conf = units.to_dpd(np.asarray(mapped_x[rng.integers(nf), rng.integers(nm)], dtype=float))
        conf = (conf - conf.mean(0)) @ _random_rotation(rng).T
        out[k] = conf + rng.uniform(0, 1, 3) * L
    return out, L


def build_system(workdir: str | Path, model: CGModel, bead_charges: np.ndarray,
                 mapped_x: np.ndarray, aa_box_lengths: np.ndarray, units: DPDUnits,
                 st: CGSystemSettings, params: BondedParams) -> BuiltSystem:
    """Write ``system.data``, ``pair.lmp``, ``params.lmp`` and ``in.cg`` into ``workdir``."""
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(st.seed + 1)
    warnings: list[str] = []
    x, L = start_structure(mapped_x, aa_box_lengths, units, st, model.n_beads)
    n_mol = len(x)
    if np.any(L < 2.5):
        warnings.append(f"box edge {L.min():.2f} r_c is very small; increase 'replicate' or the box size")

    types = model.bead_types
    charged = st.style == "dpd_q"
    q_bead = np.round(np.asarray(bead_charges, float), 4) if charged else np.zeros(model.n_beads)
    q_total = float(q_bead.sum() * n_mol)
    n_beads_total = n_mol * model.n_beads
    n_target = int(round(st.density * float(np.prod(L))))
    n_free = n_target - n_beads_total
    if st.solvent and n_free < 0:
        warnings.append(f"the solute alone exceeds density {st.density}: no solvent added")
    n_free = max(0, n_free) if st.solvent else 0

    n_ions = 0
    ion_q = 0.0
    if charged and st.neutralize and abs(q_total) > 1e-3:
        n_ions = int(round(abs(q_total)))
        ion_q = -np.sign(q_total)
        if abs(n_ions - abs(q_total)) > 0.01:
            warnings.append(f"net charge {q_total:.3f} is not an integer; system stays charged by "
                            f"{q_total + n_ions * ion_q:.3f}")
        n_free = max(0, n_free - n_ions)
    n_solv = n_free

    type_names = list(types)
    if n_solv:
        if st.solvent_name in type_names:
            warnings.append(f"solvent name {st.solvent_name!r} is also a bead type; solvent shares its type")
        else:
            type_names.append(st.solvent_name)
    ion_name = "CIp" if ion_q > 0 else "CIm"
    if n_ions:
        type_names.append(ion_name)
    tid = {t: k + 1 for k, t in enumerate(type_names)}

    bgroups = model.bond_group_names()
    agroups = model.angle_group_names()
    bid = {g: k + 1 for k, g in enumerate(bgroups)}
    aid = {g: k + 1 for k, g in enumerate(agroups)}

    # ---- atoms
    lines_atoms = []
    solute_ids = np.zeros((n_mol, model.n_beads), dtype=np.int64)
    a = 0
    wrap = lambda p: np.mod(p, L)  # noqa: E731
    style = st.style

    def atom_line(i, mol, t, q, p):
        p = wrap(p)
        if style == "dpd":
            return f"{i} {mol} {t} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}"
        if style == "dpd_q":
            return f"{i} {mol} {t} {q:.6f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}"
        # hybrid angle mdpd: id type x y z molecule-ID rho
        return f"{i} {t} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {mol} {st.density:.4f}"

    for m in range(n_mol):
        for b, bead in enumerate(model.beads):
            a += 1
            solute_ids[m, b] = a
            lines_atoms.append(atom_line(a, m + 1, tid[bead.type], q_bead[b], x[m, b]))
    mol = n_mol
    for _ in range(n_solv):
        a += 1
        mol += 1
        lines_atoms.append(atom_line(a, mol, tid[st.solvent_name], 0.0, rng.uniform(0, 1, 3) * L))
    for _ in range(n_ions):
        a += 1
        mol += 1
        lines_atoms.append(atom_line(a, mol, tid[ion_name], ion_q, rng.uniform(0, 1, 3) * L))
    n_atoms = a

    # ---- bonds / angles (unwrapped bead coordinates are not needed: LAMMPS uses the
    # minimum image for bonded interactions, all CG bonds are shorter than half the box)
    lines_bonds, lines_angles = [], []
    for m in range(n_mol):
        ids = solute_ids[m]
        for (i, j), g in zip(model.bonds, model.bond_groups):
            lines_bonds.append(f"{len(lines_bonds) + 1} {bid[g]} {ids[i]} {ids[j]}")
        for (i, j, k), g in zip(model.angles, model.angle_groups):
            lines_angles.append(f"{len(lines_angles) + 1} {aid[g]} {ids[i]} {ids[j]} {ids[k]}")

    hdr = ["LAMMPS data file written by cgdpd-fit", "",
           f"{n_atoms} atoms", f"{len(type_names)} atom types"]
    if lines_bonds:
        hdr += [f"{len(lines_bonds)} bonds", f"{len(bgroups)} bond types"]
    if lines_angles:
        hdr += [f"{len(lines_angles)} angles", f"{len(agroups)} angle types"]
    hdr += ["", f"0.0 {L[0]:.6f} xlo xhi", f"0.0 {L[1]:.6f} ylo yhi", f"0.0 {L[2]:.6f} zlo zhi", "",
            "Masses", ""]
    hdr += [f"{tid[t]} {st.bead_mass:g}  # {t}" for t in type_names]
    style_tag = {"dpd": "angle", "dpd_q": "full", "mdpd": "hybrid"}[style]
    body = hdr + ["", f"Atoms  # {style_tag}", ""] + lines_atoms
    if lines_bonds:
        body += ["", "Bonds", ""] + lines_bonds
    if lines_angles:
        body += ["", "Angles", ""] + lines_angles
    data_file = wd / "system.data"
    data_file.write_text("\n".join(body) + "\n")

    # ---- pair coefficients (fixed during the bonded fit)
    is_q = {t: False for t in type_names}
    for b, bead in enumerate(model.beads):
        if abs(q_bead[b]) > 1e-6:
            is_q[bead.type] = True
    if n_ions:
        is_q[ion_name] = True
    pair = [f"# pair coefficients ({STYLES[style]}); kept fixed during the bonded fit"]
    for i, t1 in enumerate(type_names):
        for t2 in type_names[i:]:
            ii, jj = sorted((tid[t1], tid[t2]))
            if style == "mdpd":
                A = st.pair_value(st.A_ij, st.A_default, t1, t2)
                B = st.pair_value(st.B_ij, st.B_default, t1, t2)
                pair.append(f"pair_coeff {ii} {jj} mdpd/rhosum {st.rd:g}")
                pair.append(f"pair_coeff {ii} {jj} mdpd {A:g} {B:g} {st.gamma:g} 1.0 {st.rd:g}  # {t1} {t2}")
            else:
                aij = st.pair_value(st.a_ij, st.a_default, t1, t2)
                extra = " yes" if style == "dpd_q" and is_q[t1] and is_q[t2] else ""
                pair.append(f"pair_coeff {ii} {jj} {aij:g} {st.gamma:g}{extra}  # {t1} {t2}")
    pair_file = wd / "pair.lmp"
    pair_file.write_text("\n".join(pair) + "\n")

    write_params(wd / "params.lmp", params, model)

    n_sol_types = len(types)
    input_file = wd / "in.cg"
    input_file.write_text(input_script(st, units, n_sol_types, bool(lines_bonds), bool(lines_angles),
                                       params.angle_style))
    return BuiltSystem(wd, data_file, input_file, pair_file, n_mol, n_solv, n_ions, L, solute_ids,
                       type_names, q_total + n_ions * ion_q, warnings)


def write_params(path: Path, params: BondedParams, model: CGModel) -> None:
    lines = ["# bonded coefficients (LAMMPS convention, no factor 1/2): bonds K r0 [kT/rc^2, rc]",
             f"# angles ({params.angle_style}): K theta0 [kT, degrees]"]
    for k, g in enumerate(model.bond_group_names(), 1):
        K, r0 = params.bonds[g]
        lines.append(f"bond_coeff {k} {K:.6g} {r0:.6g}  # {g}")
    for k, g in enumerate(model.angle_group_names(), 1):
        K, th = params.angles[g]
        lines.append(f"angle_coeff {k} {K:.6g} {th:.6g}  # {g}")
    Path(path).write_text("\n".join(lines) + "\n")


def input_script(st: CGSystemSettings, units: DPDUnits, n_solute_types: int, bonds: bool,
                 angles: bool, angle_style: str, dump: bool = True, system_dir: str = "") -> str:
    """LAMMPS input; ``system_dir`` is the folder of ``system.data``/``pair.lmp`` relative
    to the run directory (``params.lmp`` is always read from the run directory)."""
    sl = " ".join(f"{v:g}" for v in st.special_lj)
    atom_style = {"dpd": "angle", "dpd_q": "full", "mdpd": "hybrid angle mdpd"}[st.style]
    L = ["# CG simulation written by cgdpd-fit",
         "units           lj",
         f"atom_style      {atom_style}",
         "boundary        p p p",
         "comm_modify     vel yes cutoff 3.5",
         "newton          on"]
    if bonds:
        L.append("bond_style      harmonic")
    if angles:
        L.append(f"angle_style     {angle_style}")
    if st.style == "dpd_q":
        L.append(f"special_bonds   lj {sl} coul {sl}")
    else:
        L.append(f"special_bonds   lj {sl}")
    L.append(f"read_data       {system_dir}system.data")
    kT = units.kT
    if st.style == "dpd":
        L.append(f"pair_style      dpd {kT:g} 1.0 {st.seed}")
    elif st.style == "dpd_q":
        lb = units.to_dpd(st.bjerrum)          # Bjerrum length in r_c
        L.append(f"pair_style      dpd/coul/slater/long {kT:g} 1.0 {st.seed} {st.slater_lambda:g} {st.coul_cut:g}")
        L.append(f"dielectric      {1.0 / lb:.6g}   # Bjerrum length {st.bjerrum:g} A = {lb:.4f} r_c")
        L.append(f"kspace_style    ewald {st.kspace_acc:g}")
    else:
        L.append(f"pair_style      hybrid/overlay mdpd/rhosum mdpd {kT:g} 1.0 {st.seed}")
    L.append(f"include         {system_dir}pair.lmp")
    L.append("include         params.lmp")
    L += ["neighbor        0.3 bin",
          "neigh_modify    delay 0 every 1 check yes",
          f"group           solute type 1:{n_solute_types}",
          f"velocity        all create {kT:g} {st.seed + 7} mom yes rot yes dist gaussian"]
    if st.style == "mdpd":
        L.append("fix             integ all mvv/dpd")
    else:
        L.append("fix             integ all nve")
    L += ["thermo          1000",
          "thermo_style    custom step temp press pe ebond eangle",
          f"timestep        {st.dt / 4:g}",
          f"run             {st.n_start}",
          f"timestep        {st.dt:g}",
          f"run             {st.n_equil}"]
    if dump:
        L += [f"dump            traj solute custom {st.dump_every} traj.lammpstrj id mol type xu yu zu",
              "dump_modify     traj sort id"]
    L += [f"run             {st.n_prod}",
          "write_data      final.data"]
    return "\n".join(L) + "\n"
