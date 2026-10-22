<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logo-dark.svg">
    <img src="docs/logo.svg" alt="Raven Labs" width="220">
  </picture>
</p>

# DPD-Bonded-Fitter — bonded DPD / mDPD parameters from all-atom MD

[![tests](https://github.com/erroshakan1/DPD-Bonded-Fitter/actions/workflows/tests.yml/badge.svg)](https://github.com/erroshakan1/DPD-Bonded-Fitter/actions/workflows/tests.yml)

Python package and command: `cgdpd-fit`.

Takes an all-atom simulation from **LAMMPS** (data file + `lammpstrj`) or **GROMACS**
(`tpr` + `xtc`/`trr`), maps it onto a coarse-grained model you draw in the GUI and fits the
bonded parameters (harmonic bonds; harmonic, cosine/squared or tabulated angles) of DPD, charged DPD
or many-body DPD (mDPD) models with LAMMPS. Swarm-CG–style optimisation, written for DPD.

## Workflow

1. **All-atom input.** Molecules are split by molecule ID (or bonds) and identical
   molecules are grouped into species. Trajectories may hold `xu yu zu`, `x y z` with or
   without image flags, or scaled coordinates; molecules are made whole along their bonds.
2. **Mapping.** Click atoms of the species into beads (hydrogens follow their heavy atom),
   or start from *Auto (chain)*. For complex molecules switch the all-atom view from the data
   file to the *most extended* conformation of the trajectory (largest radius of gyration) or
   to any frame; atom names of GROMACS topologies are shown as labels. CG bonds come from the all-atom connectivity (or draw them
   bead to bead), angles from the bonds. Terms with the same **group** label share one LAMMPS
   type and are fitted together. By default the group follows the bead **names**, and names may
   repeat: name all tail beads of a lipid `C` and every tail bond becomes one `C-C` type and
   every tail angle one `C-C-C` type, fitted to the pooled all-atom distribution. The bead
   *type* only selects the pair coefficients and equals the name unless you change it.
3. **DPD units.** `r_c` in Å (`r_c = (ρ N_m 30 Å³)^(1/3)`, 6.46 Å for N_m = 3, ρ = 3).
4. **CG system.** `dpd` (`atom_style angle`), `dpd_q` (`atom_style full`,
   `dpd/coul/slater/long` + Ewald, bead charge = sum of its atomic charges, counter-ion beads),
   or `mdpd` (`atom_style hybrid angle mdpd`, `mdpd/rhosum` + `mdpd`, `fix mvv/dpd`, needs the
   DPD-MESO package). The box either keeps the all-atom arrangement (scaled, optionally
   replicated) or holds random copies; solvent beads fill it to the bead density.
   Non-bonded coefficients are fixed during the bonded fit.
5. **Fit.**
   * Boltzmann inversion of the mapped distributions gives the start values.
   * Iterative moment matching (IBI-like) updates them:
     `K ← K·(σ²_CG/σ²_AA)^α`, `x0 ← x0 + α(⟨x⟩_AA − ⟨x⟩_CG)`.
   * Swarm-CG–style PSO cycles then refine them. Force constants are searched on a log
     scale, and every PSO step runs the whole swarm in parallel.
   * Angle style *table*: for asymmetric or two-peaked angle distributions (e.g. the
     tacticity of one-bead-per-monomer polystyrene), which no single well reproduces. The
     angle potential is tabulated (`angles.table`, `angle_style table`): it starts from the
     Boltzmann inversion `U = −kT ln(P(θ)/sin θ)` and is refined by iterative Boltzmann
     inversion, `U ← U + α kT ln(P_CG/P_AA)`. The swarm then fits the bonds only.
   * Score: earth mover's distance of every group divided by the all-atom width (or the
     Swarm-CG–style absolute mix), combined as an RMS.
   * Effort presets *Quick*, *Standard* and *Thorough* set everything. As in Swarm-CG, each
     stage stops by itself once the best score improves by less than a tolerance (1 %) for a
     few steps; the step numbers are only upper limits. *Advanced* exposes the PSO cycles
     as a table (what to optimise, force-constant range ×, length ±%, angle ±°).
   * The radius of gyration of the molecule (beads) is shown next to the bond and angle
     distributions and written to the results; it is reported, and scored only on request.

   * The *Time series* tab plots the radius of gyration and the solvent accessible surface
     (Shrake–Rupley on the beads, each molecule alone, bead radius from `r_c³/ρ`, 1.4 Å
     probe) against time, mean over the molecules: the mapped all-atom run on the bottom
     axis (ps or ns; for LAMMPS dumps set the MD *Timestep*) and the best CG run on the top
     axis (DPD time units τ).

Output (`<out>/`): `bestfit/params.lmp`, a ready-to-run CG system (`system.data`,
`pair.lmp`, `in.cg`), `distributions.png`, `timeseries.png` with `cg_timeseries.csv` (and `aa_timeseries.csv`), `history.csv`, `score_history.png`, and one
folder per simulation under `runs/`.

## Conventions

LAMMPS energies have no factor 1/2: `E = K (r − r0)²`, so the Gaussian estimate is
`K = kT / (2σ²)` with σ in DPD length units (bonds), radians (harmonic angles) or cosines
(cosine/squared). Tabulated angles are in kT. Lengths in trajectories are Å (LAMMPS `real`/`metal`, or MDAnalysis).

## What you need

* **Python ≥ 3.10.** All Python packages (numpy, scipy, matplotlib, PySide6, MDAnalysis)
  are installed automatically.
* **LAMMPS**, used for the CG simulations whatever program made the all-atom run. It
  needs the `MOLECULE` and `DPD-BASIC` packages, plus `KSPACE` for charged DPD and
  `DPD-MESO` for mDPD. It is looked up on `PATH` as `lmp`, `lmp_serial`, `lmp_mpi`, … ;
  choose another executable in the GUI or set `LAMMPS_EXE`. A fit checks the packages
  (`lmp -h`) before it starts; to add one, rebuild with e.g.
  `cmake -D PKG_DPD-MESO=on . && make install` in your LAMMPS build folder.
* **Your all-atom simulation**, one of:
  * **LAMMPS**:
    * a data file (any atom style with molecule IDs, e.g. `full`, with `Masses` and
      `Bonds`; the output of `write_data` works);
    * a text dump of the same system with at least `id` and coordinates, e.g.
      `dump 1 all custom 1000 aa.lammpstrj id mol type x y z ix iy iz` (or `xu yu zu`).
    * Lengths must be in Å (`units real` or `metal`).
  * **GROMACS**:
    * the run input `.tpr` (topology, masses, charges, bonds);
    * the trajectory `.xtc` or `.trr`. Molecules may be broken across the box; they are
      made whole along their bonds.

You do not write any input file by hand. The mapping, the CG model and all settings are
made in the GUI and saved as a project (`.json`). The CG data and input files for LAMMPS
are generated for you. Mapping files of MD2CG v3 (`cg_map.ndx`, `bonds_map.txt`,
`angles_map.txt`) can still be imported.

## Installation and start

```bash
git clone https://github.com/erroshakan1/DPD-Bonded-Fitter.git
cd DPD-Bonded-Fitter
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install .
cgdpd-fit                          # opens the GUI
```

On Linux, `cgdpd-fit desktop` adds *DPD Bonded Fitter* with its icon to the application menu
(and makes the icon appear in the taskbar on Wayland); `cgdpd-fit desktop --remove` undoes it.

The next time, only activate the environment and type `cgdpd-fit`. Without cloning, one
command installs it into its own environment and puts `cgdpd-fit` on the `PATH`:

```bash
pipx install git+https://github.com/erroshakan1/DPD-Bonded-Fitter.git
```

## Tutorial: n-dodecane

The repository ships the inputs, not the trajectory; you create it in about four minutes.

```bash
cd examples/dodecane_ua
python make_system.py            # -> dodecane.data (100 TraPPE-UA chains, dilute box)
lmp -in in.aa                    # NPT 300 K, 1 atm -> aa.lammpstrj, dodecane_final.data
```

**GUI.** Start it with `cgdpd-fit`, then:

1. Load the system:
   * *Data file*: `dodecane_final.data`
   * *Trajectory*: `aa.lammpstrj`
   * *Load*. One species is found: `S1: C12`, 100 molecules.
2. Make the mapping:
   * *Auto (chain)* with 3 beads splits each chain into 4 + 4 + 4 carbons.
   * Rename the beads `E`, `M`, `E` (double-click a bead or F2). Both bonds then fall into
     group `E-M` and the angle into `E-M-E`.
3. DPD units: set *r_c* = 7.2 Å (300 beads at density 3 in the 33.4 Å box).
4. CG system: *match the all-atom system*, untick *Add solvent beads* (bulk liquid).
5. *Compute AA distributions* shows the mapped bond and angle distributions.
6. *Run fit*, then follow the *Fit progress* tab.

**Command line.** Or run the prepared project (same settings, short runs) without the GUI:

```bash
cgdpd-fit run dodecane_project.json      # results in fit_out/
```

The results land in `fit_out/bestfit/`:

* `params.lmp`: the fitted `bond_coeff` / `angle_coeff` lines;
* `system.data`, `pair.lmp`, `in.cg`: a CG simulation ready for `lmp -in in.cg`;
* `distributions.png`: AA vs CG distributions.

**GROMACS data.** The steps are the same. In section 1 choose *GROMACS (tpr + xtc)*,
select the `.tpr` and the `.xtc`, then *Load*. Water, ions and the solute appear as
separate species.

For production fits, use longer CG runs (*Production* steps) and leave the PSO *steps*
on *auto*.

## Command line

```bash
cgdpd-fit                   # GUI
cgdpd-fit gui project.json  # GUI with a project
cgdpd-fit run project.json  # headless run of a saved project (--jobs N, --out DIR)
cgdpd-fit info system.data  # list the molecule species (LAMMPS data file or GROMACS .tpr)
cgdpd-fit desktop           # application-menu entry and taskbar icon (Linux)
cgdpd-fit legacy --tpr md.tpr --traj md.xtc --ndx cg_map.ndx \
          --bonds bonds_map.txt --angles angles_map.txt -o project.json   # MD2CG v3 files
```

## Repository layout

```
src/cgdpd_fit/io/      LAMMPS data/dump readers, GROMACS tpr/xtc (MDAnalysis), MD2CG v3 import
src/cgdpd_fit/core/    topology & species, CG model, mapping, distributions, Boltzmann
                       inversion, CG system builder, LAMMPS runner, IBI/PSO, fit workflow
src/cgdpd_fit/gui/     PySide6 interface
examples/dodecane_ua/  tutorial system
tests/                 pytest
```

## License

MIT
