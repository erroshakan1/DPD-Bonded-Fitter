"""Unit tests for the parts that were wrong in MD2CG v3 (units, K convention, IBI direction,
periodic boundaries, charges) plus the I/O round trips."""
import numpy as np
import pytest

from cgdpd_fit.core.cgbuild import CGSystemSettings, build_system
from cgdpd_fit.core.distributions import Dist, ScoreSettings, angles_deg, bond_lengths, score
from cgdpd_fit.core.model import Bead, CGModel, auto_chain_model
from cgdpd_fit.core.params import BondedParams, ibi_update, initial_guess
from cgdpd_fit.core.topology import WholeMaker, from_lammps_data
from cgdpd_fit.core.units import DPDUnits, rc_from_mapping
from cgdpd_fit.io.dump import iter_dump, scan_dump
from cgdpd_fit.io.lammps_data import Box, read_lammps_data


def chain_data(tmp_path, n_mol=4, n=6, box=20.0, shift=0.0):
    """Linear molecules of CH2-like atoms (mass 14.027), charge +1 on the first atom."""
    lines = ["test", "", f"{n_mol * n} atoms", f"{n_mol * (n - 1)} bonds", "1 atom types", "1 bond types", "",
             f"0 {box} xlo xhi", f"0 {box} ylo yhi", f"0 {box} zlo zhi", "", "Masses", "", "1 14.027", "",
             "Atoms # full", ""]
    a = 0
    for m in range(n_mol):
        for k in range(n):
            a += 1
            x = np.mod(np.array([shift + 1.5 * k, 2.0 + 3 * m, 2.0]), box)
            lines.append(f"{a} {m + 1} 1 {1.0 if k == 0 else 0.0} {x[0]} {x[1]} {x[2]}")
    lines += ["", "Bonds", ""]
    b = 0
    for m in range(n_mol):
        for k in range(n - 1):
            b += 1
            lines.append(f"{b} 1 {m * n + k + 1} {m * n + k + 2}")
    p = tmp_path / "aa.data"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_rc_from_mapping():
    assert rc_from_mapping(3, 3.0) == pytest.approx(6.463, abs=1e-3)


def test_boltzmann_inversion_uses_lammps_convention_and_dpd_units():
    rng = np.random.default_rng(0)
    rc = 7.0
    sigma_a = 0.35
    d = Dist("bond", "b", rng.normal(4.5, sigma_a, 200_000))
    p = initial_guess({d.key: d}, DPDUnits(rc=rc))
    K, r0 = p.bonds["b"]
    # E = K (r - r0)^2 with r in r_c  ->  K = kT / (2 sigma^2), sigma in r_c
    assert K == pytest.approx(1.0 / (2 * (sigma_a / rc) ** 2), rel=0.02)
    assert r0 == pytest.approx(4.5 / rc, rel=1e-3)
    a = Dist("angle", "a", rng.normal(120, 10, 200_000))
    K, th = initial_guess({a.key: a}, DPDUnits(rc=rc)).angles["a"]
    assert K == pytest.approx(1.0 / (2 * np.radians(10) ** 2), rel=0.02)
    assert th == pytest.approx(120, abs=0.1)


def test_ibi_stiffens_a_too_broad_distribution():
    rng = np.random.default_rng(1)
    ref = Dist("bond", "b", rng.normal(4.0, 0.2, 50_000))
    cg = Dist("bond", "b", rng.normal(4.2, 0.4, 50_000))       # CG too broad and too long
    p = BondedParams({"b": (100.0, 0.6)})
    q = ibi_update(p, {ref.key: ref}, {cg.key: cg}, DPDUnits(rc=6.0), alpha=1.0)
    assert q.bonds["b"][0] > 100.0 * 3.5          # variance ratio 4
    assert q.bonds["b"][1] < 0.6


def test_score_zero_for_identical_and_positive_otherwise():
    rng = np.random.default_rng(2)
    a = Dist("bond", "b", rng.normal(4, 0.3, 10_000))
    assert score({a.key: a}, {a.key: a})[0] == pytest.approx(0.0)
    b = Dist("bond", "b", a.samples + 0.3)
    total, terms = score({a.key: a}, {b.key: b}, ScoreSettings())
    assert total == pytest.approx(1.0, rel=0.05)      # shift of one sigma


def test_geometry_helpers():
    x = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0]])
    assert bond_lengths(x, [(0, 1), (1, 2)]) == pytest.approx([1, 1])
    assert angles_deg(x, [(0, 1, 2)]) == pytest.approx([90.0])


def test_whole_maker_across_boundary():
    box = Box(np.zeros(3), np.full(3, 10.0))
    x = np.array([[[9.5, 1, 1], [0.5, 1, 1], [1.5, 1, 1]]])    # wrapped chain
    w = WholeMaker.from_bonds(3, [(0, 1), (1, 2)]).apply(x, box)
    assert np.diff(w[0, :, 0]) == pytest.approx([1.0, 1.0])


def test_species_mapping_charges_and_dump(tmp_path):
    p = chain_data(tmp_path, shift=17.0)          # molecules cross the x boundary
    topo = from_lammps_data(read_lammps_data(p))
    topo.build_species()
    assert len(topo.species) == 1 and topo.species[0].count == 4
    sp = topo.species[0]
    m = auto_chain_model(sp, 2)
    assert m.n_beads == 2 and m.bonds == [(0, 1)]
    # leave atom 5 unmapped: its charge (0) and atom 0 (+1) must still be accounted for
    m.beads[1].atoms = [a for a in m.beads[1].atoms if a != 5]
    q = m.bead_charges(sp, topo.charges[sp.instances[0]])
    assert q.sum() == pytest.approx(1.0)
    # dump with wrapped coordinates and no image flags
    d = read_lammps_data(p)
    dump = tmp_path / "t.lammpstrj"
    rows = [f"{i} 1 {x:.4f} {y:.4f} {z:.4f}" for i, (x, y, z) in zip(d.ids, d.pos)]
    frame = ["ITEM: TIMESTEP", "0", "ITEM: NUMBER OF ATOMS", str(len(rows)), "ITEM: BOX BOUNDS pp pp pp",
             "0 20", "0 20", "0 20", "ITEM: ATOMS id type x y z"] + rows[::-1]
    dump.write_text("\n".join(frame + frame) + "\n")
    assert scan_dump(dump).n_frames == 2
    from cgdpd_fit.core.mapping import FrameSelection, map_lammps_dump
    mt = map_lammps_dump(str(dump), topo, sp, m, FrameSelection())
    assert mt.x.shape == (2, 4, 2, 3)
    L = bond_lengths(mt.x, [(0, 1)])
    assert np.allclose(L, L[0]) and L[0] < 10.0            # molecules made whole
    fr = next(iter_dump(dump, ids=d.ids[:3]))
    assert list(fr.ids) == list(d.ids[:3])


@pytest.mark.parametrize("style", ["dpd", "dpd_q", "mdpd"])
def test_build_system_writes_consistent_files(tmp_path, style):
    m = CGModel(beads=[Bead("A", "A", [0]), Bead("B", "B", [1]), Bead("A", "A", [2])],
                bonds=[(0, 1), (1, 2)], angles=[(0, 1, 2)])
    m.regroup_bonds()
    m.regroup_angles()
    x = np.random.default_rng(0).normal(size=(2, 5, 3, 3)) + np.array([0, 6.0, 12.0])[:, None]
    st = CGSystemSettings(style=style, box_mode="custom", n_molecules=10, box_length=6.0)
    p = BondedParams({"A-B": (100.0, 0.8)}, {"A-B-A": (5.0, 150.0)})
    b = build_system(tmp_path / style, m, np.array([1.0, 0.0, 0.0]), x, np.full(3, 30.0),
                     DPDUnits(rc=6.0), st, p)
    data = read_lammps_data(b.data_file, {"dpd": "angle", "dpd_q": "full", "mdpd": "auto"}[style]) \
        if style != "mdpd" else None
    if data is not None:
        assert data.n_atoms == round(3.0 * 6.0 ** 3)
    if style == "dpd_q":
        assert b.n_ions == 10 and b.total_charge == pytest.approx(0.0)
        assert abs(data.charges.sum()) < 1e-6
    assert "bond_coeff 1 100 0.8  # A-B" in (b.workdir / "params.lmp").read_text()
    assert b.solute_ids.shape == (10, 3)


def test_histogram_edges_follow_the_requested_resolution():
    from cgdpd_fit.core.distributions import common_edges
    a = Dist("angle", "a", np.array([100.0, 120.0]))
    e = common_edges([a], "angle", 10.0)
    assert e[0] == 0.0 and e[-1] == 180.0 and np.allclose(np.diff(e), 10.0)
    b = Dist("bond", "b", np.array([3.93, 4.41]))
    e = common_edges([b], "bond", 0.1)
    assert np.allclose(np.diff(e), 0.1) and e[0] <= 3.93 and e[-1] >= 4.41


def test_flory_huggins_aij():
    from cgdpd_fit.core.flory import aij_from_solubility, bead_molar_volume, chi
    v = bead_molar_volume(6.46, 3.0)
    assert v == pytest.approx(54.0, rel=0.01)           # 90 A^3 per bead
    c = chi(47.8, 16.0, v, 298.15)
    assert c == pytest.approx(v * 31.8 ** 2 / (8.314462618 * 298.15))
    a, x = aij_from_solubility({"W": 47.8, "C": 16.0}, v, 298.15, 25.0, 3.27)
    assert a["W W"] == pytest.approx(25.0) and a["C C"] == pytest.approx(25.0)
    assert a["W C"] == pytest.approx(25.0 + 3.27 * x["W C"])


def test_equal_bead_names_share_bond_and_angle_groups():
    m = CGModel(beads=[Bead("H", "H", [0])] + [Bead("C", "C", [k]) for k in range(1, 5)],
                bonds=[(0, 1), (1, 2), (2, 3), (3, 4)])
    m.regroup_bonds()
    m.angles_from_bonds()
    assert m.bond_group_names() == ["C-H", "C-C"]
    assert m.angle_group_names() == ["C-C-H", "C-C-C"]
    assert m.bonds_in("C-C") == [(1, 2), (2, 3), (3, 4)]
    # renaming keeps hand-made group labels, updates automatic ones
    m.bond_groups[0] = "head"
    old = m.auto_labels()
    m.beads[0].name = "N"
    m.follow_names(old)
    assert m.bond_groups[0] == "head" and m.angle_groups[0] == "C-C-N"


def test_radius_of_gyration_is_reported_but_not_scored_by_default():
    from cgdpd_fit.core.distributions import compute_distributions, radius_of_gyration
    x = np.array([[[[0.0, 0, 0], [2.0, 0, 0]]]])               # 1 frame, 1 molecule, 2 beads
    assert radius_of_gyration(x) == pytest.approx([1.0])
    m = CGModel(beads=[Bead("A", "A", [0]), Bead("A", "A", [1])], bonds=[(0, 1)])
    m.regroup_bonds()
    ref = compute_distributions(np.repeat(x, 10, axis=0), m)
    assert "rg:molecule" in ref and "bond:A-A" in ref
    cg = {k: Dist(d.kind, d.group, d.samples + 0.5) for k, d in ref.items()}
    _, terms = score(ref, cg)
    assert "rg:molecule" not in terms
    _, terms = score(ref, cg, ScoreSettings(include_rg=True))
    assert "rg:molecule" in terms
    assert "rg:molecule" not in [k for k in initial_guess(ref, DPDUnits()).keys()]


def test_ibi_stops_automatically_when_the_score_levels_off():
    from cgdpd_fit.core.optimize import EvalResult, OptimizerSettings, Tracker, preset, run_ibi
    scores = iter([1.0, 0.5, 0.3, 0.299, 0.2985, 0.2984, 0.1, 0.1])
    calls = []

    def evaluate(plist, stage):
        calls.append(stage)
        return [EvalResult(len(calls), plist[0], next(scores), dists={})]

    st = preset("standard")
    tr = Tracker()
    run_ibi(BondedParams({"b": (10.0, 1.0)}), evaluate, lambda p, d: p, st, tr, log=lambda *_: None)
    assert len(calls) == 5            # 0.3 -> 0.299 -> 0.2985: two steps below 1 %, stop
    st.auto_stop = False
    calls.clear()
    scores = iter([1.0] * 20)
    run_ibi(BondedParams({"b": (10.0, 1.0)}), evaluate, lambda p, d: p, st, Tracker(), log=lambda *_: None)
    assert len(calls) == st.ibi_iterations + 1
    assert OptimizerSettings.from_dict(st.to_dict()).tolerance == st.tolerance
