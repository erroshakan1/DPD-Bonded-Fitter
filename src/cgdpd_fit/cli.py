"""Command line interface.

    cgdpd-fit                          start the GUI
    cgdpd-fit gui [project.json]       start the GUI (optionally opening a project)
    cgdpd-fit run project.json         run a saved project without the GUI
    cgdpd-fit info system.data|md.tpr  list the molecule species of an all-atom system
    cgdpd-fit legacy ...               build a project from MD2CG v3 files (ndx + bonds/angles maps)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(prog="cgdpd-fit", description="bonded DPD/mDPD parameters from all-atom MD")
    sub = ap.add_subparsers(dest="cmd")
    g = sub.add_parser("gui", help="start the graphical interface")
    g.add_argument("project", nargs="?")
    r = sub.add_parser("run", help="run a saved project headless")
    r.add_argument("project")
    r.add_argument("--out", help="override the output folder")
    r.add_argument("--jobs", type=int, help="simultaneous LAMMPS runs")
    i = sub.add_parser("info", help="list the molecule species (LAMMPS data file or GROMACS tpr)")
    i.add_argument("data")
    i.add_argument("--style", default="auto")
    lg = sub.add_parser("legacy", help="create a project from MD2CG v3 input files")
    lg.add_argument("--data", help="LAMMPS data file (all-atom)")
    lg.add_argument("--dump", help="LAMMPS dump (all-atom)")
    lg.add_argument("--tpr", help="GROMACS tpr")
    lg.add_argument("--traj", help="GROMACS xtc/trr")
    lg.add_argument("--ndx", required=True, help="cg_map.ndx")
    lg.add_argument("--bonds", help="bonds_map.txt")
    lg.add_argument("--angles", help="angles_map.txt")
    lg.add_argument("--rc", type=float, default=6.46, help="DPD length unit in Angstrom")
    lg.add_argument("-o", "--output", default="project.json")
    args = ap.parse_args(argv)

    if args.cmd in (None, "gui"):
        from .gui.app import main as gui_main
        return gui_main([sys.argv[0]] + ([args.project] if getattr(args, "project", None) else []))
    if args.cmd == "run":
        from .core.fit import FitJob, Project
        p = Project.load(args.project)
        if args.out:
            p.out_dir = args.out
        if args.jobs:
            p.run.n_jobs = args.jobs

        def progress(stage, i, n):
            if stage == "mapping" and n and (i % max(1, n // 10) == 0 or i == n):
                print(f"  mapped {i}/{n} frames", flush=True)

        res = FitJob(p, log=lambda s: print(s, flush=True), progress=progress).run()
        return 0 if res.best is not None else 1
    if args.cmd == "info":
        from .core.fit import AAInput, load_topology
        if args.data.endswith(".tpr"):
            aa = AAInput(format="gromacs", tpr=args.data)
        else:
            aa = AAInput(format="lammps", data=args.data, atom_style=args.style)
        t = load_topology(aa)
        box = "" if aa.format == "gromacs" else f", box {t.box.lengths.round(3).tolist()} A"
        print(f"{t.n_atoms} atoms{box}")
        for sp in t.species:
            q = float(t.charges[sp.instances[0]].sum())
            print(f"  {sp.name:40s} {sp.count:6d} molecules  {sp.n_atoms:5d} atoms  charge {q:+.3f}")
        return 0
    if args.cmd == "legacy":
        from .core.fit import AAInput, Project, load_topology
        from .core.units import DPDUnits
        from .io.legacy import read_legacy
        if args.tpr:
            aa = AAInput(format="gromacs", tpr=str(Path(args.tpr).resolve()),
                         traj=str(Path(args.traj).resolve()) if args.traj else "")
        else:
            aa = AAInput(format="lammps", data=str(Path(args.data).resolve()),
                         dump=str(Path(args.dump).resolve()))
        topo = load_topology(aa)
        si, model = read_legacy(topo, args.ndx, args.bonds, args.angles)
        sp = topo.species[si]
        p = Project(aa=aa, species=si, species_formula=sp.formula, model=model, units=DPDUnits(rc=args.rc))
        p.save(args.output)
        print(f"{args.output}: species {sp.name} ({sp.count} molecules), {model.n_beads} beads, "
              f"{len(model.bond_group_names())} bond groups, {len(model.angle_group_names())} angle groups")
        for m in model.validate(sp):
            print("  warning:", m)
        return 0
    return 1


def entry() -> int:
    try:
        return main()
    except (ValueError, OSError) as exc:
        print(f"cgdpd-fit: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(entry())
