"""The complete bonded fit: project settings, evaluation of parameter sets and the workflow."""
from __future__ import annotations

import csv
import json
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..io.dump import iter_dump, scan_dump
from ..io.lammps_data import read_lammps_data
from .cgbuild import BuiltSystem, CGSystemSettings, build_system, input_script, write_params
from .distributions import Dist, ScoreSettings, compute_distributions, score
from .mapping import FrameSelection, MappedTrajectory, map_frames, map_lammps_dump, species_rows
from .model import CGModel
from .optimize import EvalResult, OptimizerSettings, Tracker, clamp, run_ibi, run_pso
from .params import BondedParams, ibi_update, initial_guess
from .runner import Cancelled, LammpsPool, RunSettings
from .topology import AATopology, WholeMaker, from_lammps_data
from .units import DPDUnits

MAX_STORED_SAMPLES = 100_000


# ======================================================================= project
@dataclass
class AAInput:
    format: str = "lammps"           # "lammps" | "gromacs"
    data: str = ""                   # LAMMPS data file
    dump: str = ""                   # LAMMPS dump (lammpstrj)
    atom_style: str = "auto"
    split: str = "auto"
    tpr: str = ""                    # GROMACS
    traj: str = ""

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "AAInput":
        s = cls()
        s.__dict__.update({k: v for k, v in d.items() if k in s.__dict__})
        return s


@dataclass
class Project:
    aa: AAInput = field(default_factory=AAInput)
    species: int = 0                 # index into the species list (largest first)
    species_formula: str = ""        # sanity check when a project is re-opened
    model: CGModel = field(default_factory=CGModel)
    frames: FrameSelection = field(default_factory=FrameSelection)
    units: DPDUnits = field(default_factory=DPDUnits)
    cg: CGSystemSettings = field(default_factory=CGSystemSettings)
    angle_style: str = "harmonic"
    scoring: ScoreSettings = field(default_factory=ScoreSettings)
    optimizer: OptimizerSettings = field(default_factory=OptimizerSettings)
    run: RunSettings = field(default_factory=RunSettings)
    out_dir: str = "cgdpd_out"

    def to_dict(self) -> dict:
        return {"version": 1, "aa": self.aa.to_dict(), "species": self.species,
                "species_formula": self.species_formula, "model": self.model.to_dict(),
                "frames": self.frames.to_dict(), "units": self.units.to_dict(),
                "cg": self.cg.to_dict(), "angle_style": self.angle_style,
                "scoring": self.scoring.to_dict(), "optimizer": self.optimizer.to_dict(),
                "run": self.run.to_dict(), "out_dir": self.out_dir}

    @classmethod
    def from_dict(cls, d: dict, base: Path | None = None) -> "Project":
        p = cls(aa=AAInput.from_dict(d.get("aa", {})), species=int(d.get("species", 0)),
                species_formula=d.get("species_formula", ""),
                model=CGModel.from_dict(d.get("model", {})),
                frames=FrameSelection.from_dict(d.get("frames", {})),
                units=DPDUnits.from_dict(d.get("units", {})),
                cg=CGSystemSettings.from_dict(d.get("cg", {})),
                angle_style=d.get("angle_style", "harmonic"),
                scoring=ScoreSettings.from_dict(d.get("scoring", {})),
                optimizer=OptimizerSettings.from_dict(d.get("optimizer", {})),
                run=RunSettings.from_dict(d.get("run", {})), out_dir=d.get("out_dir", "cgdpd_out"))
        if base is not None:          # relative paths are relative to the project file
            for k in ("data", "dump", "tpr", "traj"):
                v = getattr(p.aa, k)
                if v and not Path(v).is_absolute():
                    setattr(p.aa, k, str((base / v).resolve()))
            if not Path(p.out_dir).is_absolute():
                p.out_dir = str((base / p.out_dir).resolve())
        return p

    def save(self, path: str | Path) -> None:
        """Write the project; paths inside the project folder are stored relative to it,
        so a project folder can be moved or shared."""
        path = Path(path)
        d = self.to_dict()
        base = path.resolve().parent

        def rel(v: str) -> str:
            if not v:
                return v
            try:
                return str(Path(v).resolve().relative_to(base))
            except ValueError:
                return v

        for k in ("data", "dump", "tpr", "traj"):
            d["aa"][k] = rel(d["aa"][k])
        d["out_dir"] = rel(d["out_dir"])
        path.write_text(json.dumps(d, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text()), base=path.parent)


# ======================================================================= loading
def load_topology(aa: AAInput) -> AATopology:
    if aa.format == "gromacs":
        from ..io.gromacs import load_topology as gro_topology
        topo = gro_topology(aa.tpr, aa.traj or None)
    else:
        topo = from_lammps_data(read_lammps_data(aa.data, aa.atom_style))
    topo.build_species(aa.split if aa.format == "lammps" else "auto")
    return topo


def count_frames(aa: AAInput) -> int:
    if aa.format == "gromacs":
        from ..io.gromacs import count_frames as gro_count
        return gro_count(aa.tpr, aa.traj)
    return scan_dump(aa.dump).n_frames


def map_aa(project: Project, topo: AATopology, progress=None, should_stop=None) -> MappedTrajectory:
    sp = topo.species[project.species]
    sel = project.frames
    try:
        n_all = count_frames(project.aa)
        stop = n_all if sel.stop is None else min(sel.stop, n_all)
        n_total = max(0, (stop - sel.start + sel.stride - 1) // max(1, sel.stride))
    except Exception:  # noqa: BLE001 - only used for the progress bar
        n_total = None
    if project.aa.format == "gromacs":
        from ..io.gromacs import iter_frames
        rows = species_rows(sp, sel.max_molecules)
        frames = iter_frames(project.aa.tpr, project.aa.traj, rows, sel.start, sel.stop, sel.stride)
        return map_frames(frames, topo, sp, project.model, rows, progress, should_stop, n_total)
    return map_lammps_dump(project.aa.dump, topo, sp, project.model, sel, progress, should_stop, n_total)


# ======================================================================= evaluation
class Evaluator:
    """Runs one CG simulation per parameter set and scores it against the AA reference."""

    def __init__(self, out_dir: Path, built: BuiltSystem, model: CGModel, project: Project,
                 ref: dict[str, Dist], pool: LammpsPool, log=print):
        self.out = Path(out_dir)
        self.runs = self.out / "runs"
        self.runs.mkdir(parents=True, exist_ok=True)
        self.built = built
        self.model = model
        self.project = project
        self.ref = ref
        self.pool = pool
        self.log = log
        self.whole = WholeMaker.from_bonds(model.n_beads, model.bonds)
        self.cache: dict[tuple, EvalResult] = {}
        self.counter = 0
        self.lock = threading.Lock()
        n_types = len(model.bead_types)
        self.script = input_script(project.cg, project.units, n_types, bool(model.bonds),
                                   bool(model.angles), project.angle_style, system_dir="../../system/")

    def _key(self, p: BondedParams) -> tuple:
        return tuple(float(f"{p.get(k):.6g}") for k in p.keys())

    def __call__(self, plist: list[BondedParams], stage: str = "") -> list[EvalResult]:
        jobs = []
        out: list[EvalResult | None] = [None] * len(plist)
        for i, p in enumerate(plist):
            key = self._key(p)
            if key in self.cache:
                c = self.cache[key]
                out[i] = EvalResult(c.index, p, c.score, c.terms, c.ok, c.message, stage, c.dists)
                continue
            with self.lock:
                self.counter += 1
                idx = self.counter
            jobs.append((i, idx, p, key))

        def run(job):
            i, idx, p, key = job
            return i, key, self._run(idx, p, stage)

        for i, key, res in self.pool.map(run, jobs):
            self.cache[key] = res
            out[i] = res
        return out  # type: ignore[return-value]

    def _run(self, idx: int, p: BondedParams, stage: str) -> EvalResult:
        wd = self.runs / f"e{idx:04d}"
        wd.mkdir(parents=True, exist_ok=True)
        write_params(wd / "params.lmp", p, self.model)
        (wd / "in.cg").write_text(self.script)
        t0 = time.time()
        rc, tail = self.pool.run_one(wd)
        traj = wd / "traj.lammpstrj"
        if rc != 0 or not traj.exists():
            return EvalResult(idx, p, float("inf"), {}, False, f"LAMMPS exit {rc}: {tail}", stage)
        try:
            cg = self.analyse(traj)
        except Exception as exc:  # noqa: BLE001
            return EvalResult(idx, p, float("inf"), {}, False, f"analysis failed: {exc}", stage)
        total, terms = score(self.ref, cg, self.project.scoring)
        if not self.project.run.keep_trajectories:
            for f in ("traj.lammpstrj", "final.data"):
                (wd / f).unlink(missing_ok=True)
        (wd / "result.json").write_text(json.dumps(
            {"index": idx, "stage": stage, "score": total, "terms": terms, "params": p.to_dict(),
             "seconds": round(time.time() - t0, 1)}, indent=1))
        thin = {k: d.thin(MAX_STORED_SAMPLES, idx) for k, d in cg.items()}
        return EvalResult(idx, p, total, terms, True, "", stage, thin)

    def analyse(self, traj: Path) -> dict[str, Dist]:
        ids = self.built.solute_ids
        xs = []
        for fr in iter_dump(traj, ids=ids.ravel()):
            x = fr.pos.reshape(ids.shape + (3,))
            xs.append(self.whole.apply(x, fr.box))
        if len(xs) < 2:
            raise ValueError("too few frames in the CG trajectory")
        x = self.project.units.to_angstrom(np.stack(xs[1:]))      # drop the first (pre-run) frame
        return compute_distributions(x, self.model)


# ======================================================================= workflow
@dataclass
class FitResult:
    best: EvalResult | None
    initial: BondedParams
    out_dir: Path
    n_simulations: int


class FitJob:
    """Runs the whole fit; callbacks let a GUI follow it.

    ``log(str)``, ``progress(stage, i, n)``, ``on_ref(ref)``, ``on_result(EvalResult)``,
    ``on_best(EvalResult)``.
    """

    def __init__(self, project: Project, topo: AATopology | None = None,
                 mapped: MappedTrajectory | None = None, log=print, progress=None,
                 on_ref=None, on_result=None, on_best=None):
        self.p = project
        self.topo = topo
        self.mapped = mapped
        self.log = log
        self.progress = progress or (lambda *a: None)
        self.on_ref = on_ref
        self.on_result = on_result
        self.on_best = on_best
        self.pool = LammpsPool(project.run)
        self._stop = False
        self.n_done = 0
        self.n_expected = 0

    def stop(self) -> None:
        self._stop = True
        self.pool.stop()

    def should_stop(self) -> bool:
        return self._stop

    def run(self) -> FitResult:
        p = self.p
        out = Path(p.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        p.save(out / "project.json")
        if self.topo is None:
            self.log("Reading all-atom topology ...")
            self.topo = load_topology(p.aa)
        sp = self.topo.species[p.species]
        msgs = p.model.validate(sp)
        if any(m.startswith(("no beads", "empty", "duplicate", "group labels")) for m in msgs):
            raise ValueError("mapping is not valid: " + "; ".join(msgs))
        for m in msgs:
            self.log("warning: " + m)
        if self.mapped is None:
            self.log(f"Mapping the all-atom trajectory of {sp.name} ({sp.count} molecules) ...")
            self.mapped = map_aa(p, self.topo, lambda i, n: self.progress("mapping", i, n or 0),
                                 self.should_stop)
        mt = self.mapped
        self.log(f"  {mt.n_frames} frames x {mt.n_molecules} molecules")
        ref = compute_distributions(mt.x, p.model)
        np.savez_compressed(out / "aa_reference.npz", **{k: d.samples for k, d in ref.items()})
        if self.on_ref:
            self.on_ref(ref)
        self.log("All-atom reference (mean +- sd):")
        for k, d in ref.items():
            unit = "A" if d.kind == "bond" else "deg"
            self.log(f"  {k:24s} {d.mean:8.3f} +- {d.std:6.3f} {unit}   ({len(d.samples)} samples)")

        p0 = clamp(initial_guess(ref, p.units, p.angle_style, p.optimizer.k_max_bond,
                                 p.optimizer.k_max_angle), p.optimizer)
        self.log("Boltzmann-inversion initial guess (DPD units):")
        for g, (K, r0) in p0.bonds.items():
            self.log(f"  bond  {g:20s} K = {K:9.3f}  r0 = {r0:.4f}")
        for g, (K, th) in p0.angles.items():
            self.log(f"  angle {g:20s} K = {K:9.3f}  theta0 = {th:.2f}")

        sysdir = out / "system"
        lengths = mt.lengths[p.cg.frame] if mt.lengths else self.topo.box.lengths
        charges = p.model.bead_charges(sp, self.topo.charges[sp.instances[0]])
        built = build_system(sysdir, p.model, charges, mt.x, lengths, p.units, p.cg, p0)
        self.log(f"CG system: {built.n_molecules} molecules, {built.n_solvent} solvent beads, "
                 f"{built.n_ions} ions, box {np.round(built.box_length, 2).tolist()} r_c")
        for w in built.warnings:
            self.log("warning: " + w)

        tracker = Tracker(on_result=self._on_result, on_best=self._on_best(out, ref, built))
        ev = Evaluator(out, built, p.model, p, ref, self.pool, self.log)
        evaluate = lambda plist, stage: ev(plist, stage)  # noqa: E731
        st = p.optimizer
        best = p0
        try:
            if st.method in ("ibi", "ibi+pso"):
                self.log(f"Iterative moment matching ({st.ibi_iterations} iterations) ...")
                self.n_expected += st.ibi_iterations + 1
                upd = lambda q, cg: ibi_update(q, ref, cg, p.units, st.ibi_alpha)  # noqa: E731
                best = run_ibi(p0, evaluate, upd, st, tracker, self.log, self.should_stop)
            if st.method in ("pso", "ibi+pso"):
                if st.method == "pso":
                    tracker.add(evaluate([p0], "initial"))
                best = run_pso(best, evaluate, st, tracker, p.run.n_jobs, self.log, self.should_stop,
                               on_cycle=lambda c, n, k: self._expect(k))
        except Cancelled:
            self.log("Stopped.")
        self._write_history(out, tracker)
        if tracker.best is not None:
            self.log(f"Best score {tracker.best.score:.4f} (simulation e{tracker.best.index:04d})")
            self.log(f"Results in {out / 'bestfit'}")
        return FitResult(tracker.best, p0, out, ev.counter)

    # ---------------------------------------------------------- callbacks
    def _expect(self, n: int) -> None:
        self.n_expected += n

    def _on_result(self, r: EvalResult) -> None:
        self.n_done += 1
        self.progress("fit", self.n_done, max(self.n_expected, self.n_done))
        if self.on_result:
            self.on_result(r)

    def _on_best(self, out: Path, ref, built: BuiltSystem):
        def cb(r: EvalResult) -> None:
            self.write_bestfit(out / "bestfit", r, ref, built)
            if self.on_best:
                self.on_best(r)
        return cb

    def write_bestfit(self, d: Path, r: EvalResult, ref, built: BuiltSystem) -> None:
        """Best parameters plus a ready-to-run CG system (same composition) in ``d``."""
        d.mkdir(parents=True, exist_ok=True)
        write_params(d / "params.lmp", r.params, self.p.model)
        for f in ("system.data", "pair.lmp"):
            shutil.copy(built.workdir / f, d / f)
        (d / "in.cg").write_text(input_script(self.p.cg, self.p.units, len(self.p.model.bead_types),
                                              bool(self.p.model.bonds), bool(self.p.model.angles),
                                              self.p.angle_style))
        (d / "result.json").write_text(json.dumps({
            "simulation": r.index, "stage": r.stage, "score": r.score, "terms": r.terms,
            "params": r.params.to_dict(), "units": self.p.units.to_dict(),
            "groups": {"bonds": self.p.model.bond_group_names(), "angles": self.p.model.angle_group_names()},
            "reference": {k: {"mean": v.mean, "std": v.std} for k, v in ref.items()},
            "cg": {k: {"mean": v.mean, "std": v.std} for k, v in (r.dists or {}).items()},
        }, indent=1))
        try:
            from ..plots import plot_distributions
            plot_distributions(ref, r.dists or {}, d / "distributions.png",
                               title=f"best fit: score {r.score:.4f} (e{r.index:04d})")
        except Exception as exc:  # noqa: BLE001 - plotting is optional
            self.log(f"plot warning: {exc}")

    def _write_history(self, out: Path, tracker: Tracker) -> None:
        if not tracker.history:
            return
        keys = tracker.history[0].params.keys()
        with open(out / "history.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["simulation", "stage", "score", "ok"] +
                       [f"{k[0]}:{k[1]}:{'K' if k[2] == 0 else 'x0'}" for k in keys])
            for r in tracker.history:
                w.writerow([r.index, r.stage, f"{r.score:.6g}", int(r.ok)] +
                           [f"{r.params.get(k):.6g}" for k in keys])
        try:
            from ..plots import plot_history
            plot_history(tracker.history, out / "score_history.png")
        except Exception as exc:  # noqa: BLE001
            self.log(f"plot warning: {exc}")
