"""Main window: all-atom input, mapping, CG system and fit settings, live results."""
from __future__ import annotations

import os
import traceback
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, QSettings, Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QColor, QIcon, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox, QFileDialog,
    QFormLayout, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSpinBox,
    QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QToolButton, QVBoxLayout, QWidget,
)

from .. import __version__
from ..core.cgbuild import STYLES, CGSystemSettings
from ..core.distributions import ScoreSettings, compute_distributions
from ..core.fit import AAInput, FitJob, Project, count_frames, load_topology, map_aa
from ..core.mapping import FrameSelection, reference_beads
from ..core.model import CENTER_LABELS, CENTERS, Bead, CGModel, auto_chain_model
from ..core.optimize import OptimizerSettings
from ..core.params import ANGLE_STYLES, initial_guess
from ..core.runner import Cancelled, RunSettings, find_lammps
from ..core.units import DPDUnits, rc_from_mapping
from ..plots import draw_distributions, draw_history
from .mol_view import MoleculeView
from .pair_dialog import PairDialog
from .widgets import CollapsibleGroup, NoWheelFilter, block_wheel

BEAD_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6",
               "#bfef45", "#469990", "#9a6324", "#800000", "#808000", "#000075", "#fabed4",
               "#ffd8b1", "#aaffc3", "#dcbeff", "#a9a9a9"]
ELEMENT_COLORS = {"H": "#eeeeee", "C": "#606060", "N": "#3050f8", "O": "#ff0d0d", "P": "#ff8000",
                  "S": "#e6c200", "F": "#90e050", "Cl": "#1ff01f", "Br": "#a62929", "Na": "#ab5cf2"}

HELP = """<h3>Workflow</h3>
<ol>
<li><b>All-atom input</b>: a LAMMPS data file (topology, masses, charges) and the trajectory
(<code>.lammpstrj</code>; <code>xu yu zu</code>, <code>x y z</code> with or without image flags,
or scaled coordinates). Molecules are split by molecule ID and identical molecules are
grouped into species. GROMACS <code>.tpr</code> + <code>.xtc</code> works through MDAnalysis.</li>
<li><b>Map</b> the selected species: pick a bead (row, CG panel or keys 1-9) and click
heavy atoms in the all-atom panel (Shift+drag box-selects). Hydrogens follow their heavy
atom. <i>Bonds from atoms</i> connects beads that share an all-atom bond, <i>Angles from bonds</i>
adds all bead triples. <i>Draw bonds</i> lets you connect/disconnect two beads by clicking them.
Bonds/angles with the same <b>group</b> label share one LAMMPS type and are fitted together.</li>
<li><b>DPD units</b>: Angstrom per DPD length r_c (e.g. 6.46 A for N_m = 3, rho = 3).</li>
<li><b>CG system</b>: DPD (angle), charged DPD (full) or mDPD. <i>match</i> keeps the
all-atom arrangement (scaled, optionally replicated), <i>custom</i> drops random copies into
a box. Solvent beads fill the box to the bead density. Non-bonded coefficients stay fixed.</li>
<li><b>Fit</b>: Boltzmann inversion gives the start values; iterative moment matching (IBI)
and Swarm-CG style PSO cycles refine them. Each PSO step runs the whole swarm in parallel.
Results: <code>bestfit/</code> (params.lmp, ready-to-run system, plots), <code>history.csv</code>.</li>
</ol>
<h3>Mouse</h3>Left drag: rotate &nbsp; Ctrl+left: roll &nbsp; Right/middle drag: pan &nbsp;
Wheel: zoom &nbsp; Double-click: reset view"""


class Worker(QObject):
    log = Signal(str)
    progress = Signal(str, int, int)
    event = Signal(str, object)
    done = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.stop_requested = False
        self.job: FitJob | None = None

    def request_stop(self):
        self.stop_requested = True
        if self.job is not None:
            self.job.stop()

    @Slot()
    def run(self):
        try:
            self.done.emit(self.fn(self))
        except Cancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 - report everything to the GUI
            if self.stop_requested:
                self.cancelled.emit()
            else:
                self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


def _hline(*widgets) -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    for x in widgets:
        if isinstance(x, int):
            lay.addStretch(x)
        else:
            lay.addWidget(x)
    return w


def _swatch(color: str) -> QIcon:
    pm = QPixmap(14, 14)
    pm.fill(QColor(color))
    return QIcon(pm)


def _dspin(lo, hi, val, dec=3, step=None, suffix="") -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setValue(val)
    if step:
        s.setSingleStep(step)
    if suffix:
        s.setSuffix(suffix)
    return s


def _ispin(lo, hi, val, step=None, prefix="", suffix="") -> QSpinBox:
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    if step:
        s.setSingleStep(step)
    if prefix:
        s.setPrefix(prefix)
    if suffix:
        s.setSuffix(suffix)
    return s


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"cgdpd-fit {__version__} - bonded DPD parameters from all-atom MD")
        self.resize(1550, 950)
        self.settings = QSettings("cgdpd-fit", "cgdpd-fit")
        self.topo = None
        self.models: dict[int, CGModel] = {}
        self.cur: int | None = None
        self.active_bead = 0
        self.mapped = None
        self.ref = None
        self.best = None
        self.history = []
        self.cg_extra = CGSystemSettings()        # pair tables etc. not shown as widgets
        self.thread: QThread | None = None
        self.worker: Worker | None = None
        self._connect_first: int | None = None
        self._aa_atoms = np.zeros((0, 3))
        self._build_ui()
        self._build_menu()
        self._update_enabled()

    # ================================================================== UI
    def _section(self, title: str, collapsed: bool = False) -> CollapsibleGroup:
        key = f"collapsed/{title}"
        saved = self.settings.value(key)
        if saved is not None:
            collapsed = saved in (True, "true", "True", 1, "1")
        g = CollapsibleGroup(title, collapsed=collapsed)
        g.arrow.clicked.connect(lambda on, k=key: self.settings.setValue(k, not on))
        return g

    def _file_row(self, edit: QLineEdit, slot) -> QWidget:
        b = QToolButton()
        b.setText("...")
        b.clicked.connect(slot)
        return _hline(edit, b)

    def _build_ui(self):
        # ------------------------------------------------ central: tabs
        self.tabs = QTabWidget()
        # mapping tab
        self.aa_view = MoleculeView()
        self.aa_view.placeholder = "Load the all-atom system and select a species"
        self.cg_view = MoleculeView()
        self.cg_view.placeholder = "Define beads for the selected species"
        self.chk_h = QCheckBox("Hydrogens")
        self.chk_h.setChecked(True)
        self.chk_aa_labels = QCheckBox("Labels")
        self.cmb_color = QComboBox()
        self.cmb_color.addItems(["Color: mapping", "Color: element"])
        self.chk_cg_labels = QCheckBox("Labels")
        self.chk_cg_labels.setChecked(True)
        self.chk_skeleton = QCheckBox("Atoms")
        self.chk_skeleton.setToolTip("Show the heavy-atom skeleton inside the beads")
        self.chk_skeleton.setChecked(True)
        self.btn_draw = QToolButton()
        self.btn_draw.setText("Draw bonds")
        self.btn_draw.setCheckable(True)
        self.btn_draw.setToolTip("Click two beads to add (or remove) a CG bond between them")
        self.spn_instance = _ispin(1, 1, 1, prefix="molecule ")
        aa_panel = QWidget()
        la = QVBoxLayout(aa_panel)
        la.addWidget(_hline(QLabel("<b>All-atom</b>"), 1, self.chk_h, self.chk_aa_labels, self.cmb_color,
                            self.spn_instance))
        la.addWidget(self.aa_view, 1)
        cg_panel = QWidget()
        lc = QVBoxLayout(cg_panel)
        lc.addWidget(_hline(QLabel("<b>Coarse-grained</b>"), 1, self.btn_draw, self.chk_skeleton,
                            self.chk_cg_labels))
        lc.addWidget(self.cg_view, 3)
        self.topo_tabs = QTabWidget()
        self.tbl_bonds = self._topo_table(["Beads", "Group", "AA mean (Å)"])
        self.tbl_angles = self._topo_table(["Beads", "Group", "AA mean (°)"])
        self.topo_tabs.addTab(self.tbl_bonds, "Bonds")
        self.topo_tabs.addTab(self.tbl_angles, "Angles")
        btn_del = QPushButton("Remove selected")
        btn_del.clicked.connect(self.remove_selected_terms)
        btn_regroup = QPushButton("Groups by bead type")
        btn_regroup.setToolTip("Reset group labels: equal bead types -> one bond/angle type")
        btn_regroup.clicked.connect(self.regroup)
        btn_unique = QPushButton("One group each")
        btn_unique.clicked.connect(self.unique_groups)
        lc.addWidget(self.topo_tabs, 2)
        lc.addWidget(_hline(btn_del, btn_regroup, btn_unique, 1))
        msplit = QSplitter(Qt.Horizontal)
        msplit.addWidget(aa_panel)
        msplit.addWidget(cg_panel)
        msplit.setSizes([1000, 1000])
        self.tabs.addTab(msplit, "Mapping")
        # distributions tab
        self.fig_dist = Figure(figsize=(8, 6))
        self.canvas_dist = FigureCanvasQTAgg(self.fig_dist)
        self.lbl_dist = QLabel("")
        self.lbl_dist.setWordWrap(True)
        self.lbl_dist.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        wd = QWidget()
        ld = QVBoxLayout(wd)
        ld.addWidget(self.lbl_dist)
        ld.addWidget(self.canvas_dist, 1)
        self.tabs.addTab(wd, "Distributions")
        # progress tab
        self.fig_hist = Figure(figsize=(8, 3))
        self.canvas_hist = FigureCanvasQTAgg(self.fig_hist)
        self.tbl_params = QTableWidget(0, 4)
        self.tbl_params.setHorizontalHeaderLabels(["Term", "K", "r0 / θ0", "Score term"])
        self.tbl_params.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_params.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_params.verticalHeader().setVisible(False)
        psplit = QSplitter(Qt.Vertical)
        psplit.addWidget(self.canvas_hist)
        psplit.addWidget(self.tbl_params)
        self.tabs.addTab(psplit, "Fit progress")
        self.setCentralWidget(self.tabs)

        # ------------------------------------------------ left dock
        left = QWidget()
        L = QVBoxLayout(left)

        g1 = self._section("1  All-atom system")
        f1 = QFormLayout(g1.content)
        self.cmb_format = QComboBox()
        self.cmb_format.addItems(["LAMMPS (data + lammpstrj)", "GROMACS (tpr + xtc, MDAnalysis)"])
        self.cmb_format.currentIndexChanged.connect(self._update_format)
        f1.addRow("Format", self.cmb_format)
        self.ed_data = QLineEdit()
        self.ed_dump = QLineEdit()
        self.lbl_data = QLabel("Data file")
        self.lbl_dump = QLabel("Trajectory")
        f1.addRow(self.lbl_data, self._file_row(self.ed_data, self.browse_data))
        f1.addRow(self.lbl_dump, self._file_row(self.ed_dump, self.browse_dump))
        self.cmb_style = QComboBox()
        self.cmb_style.addItems(["auto", "full", "angle", "molecular", "bond"])
        self.cmb_split = QComboBox()
        self.cmb_split.addItems(["auto", "molid", "bonds"])
        self.row_style = _hline(self.cmb_style, QLabel("Molecules by"), self.cmb_split)
        f1.addRow("Atom style", self.row_style)
        self.btn_load = QPushButton("Load")
        self.btn_load.clicked.connect(self.load_aa)
        f1.addRow(self.btn_load)
        self.tbl_species = QTableWidget(0, 4)
        self.tbl_species.setHorizontalHeaderLabels(["Species", "Atoms", "Count", "Charge"])
        self.tbl_species.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_species.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tbl_species.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_species.verticalHeader().setVisible(False)
        self.tbl_species.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for c in (1, 2, 3):
            self.tbl_species.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.tbl_species.setMinimumHeight(110)
        self.tbl_species.itemSelectionChanged.connect(self.on_species_selected)
        f1.addRow(self.tbl_species)
        self.spn_start = _ispin(0, 10**8, 0)
        self.spn_stop = _ispin(0, 10**8, 0)
        self.spn_stop.setSpecialValueText("end")
        self.spn_stride = _ispin(1, 10**6, 1)
        f1.addRow("Frames", _hline(QLabel("from"), self.spn_start, QLabel("to"), self.spn_stop,
                                   QLabel("every"), self.spn_stride))
        self.spn_maxmol = _ispin(0, 10**7, 0)
        self.spn_maxmol.setSpecialValueText("all")
        self.spn_maxmol.setToolTip("Use only a random subset of the molecules (0 = all)")
        f1.addRow("Molecules", self.spn_maxmol)
        self.lbl_aa_info = QLabel("")
        self.lbl_aa_info.setWordWrap(True)
        f1.addRow(self.lbl_aa_info)
        L.addWidget(g1)

        g2 = self._section("2  Mapping (atoms -> beads)")
        v2 = QVBoxLayout(g2.content)
        self.tbl_beads = QTableWidget(0, 3)
        self.tbl_beads.setHorizontalHeaderLabels(["Bead", "Type", "Atoms"])
        self.tbl_beads.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_beads.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tbl_beads.verticalHeader().setVisible(False)
        self.tbl_beads.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.tbl_beads.setMinimumHeight(170)
        self.tbl_beads.itemSelectionChanged.connect(self.on_bead_row_selected)
        self.tbl_beads.itemChanged.connect(self.on_bead_edited)
        v2.addWidget(self.tbl_beads)
        b_add = QPushButton("Add bead")
        b_add.clicked.connect(self.add_bead)
        b_del = QPushButton("Remove bead")
        b_del.clicked.connect(self.remove_bead)
        b_clear = QPushButton("Clear atoms")
        b_clear.clicked.connect(self.clear_bead)
        v2.addWidget(_hline(b_add, b_del, b_clear))
        self.spn_auto = _ispin(1, 200, 3, suffix=" beads")
        b_auto = QPushButton("Auto (chain)")
        b_auto.setToolTip("Split the longest heavy-atom chain into consecutive beads")
        b_auto.clicked.connect(self.auto_map)
        v2.addWidget(_hline(b_auto, self.spn_auto, 1))
        b_bonds = QPushButton("Bonds from atoms")
        b_bonds.clicked.connect(self.bonds_from_atoms)
        b_angles = QPushButton("Angles from bonds")
        b_angles.clicked.connect(self.angles_from_bonds)
        v2.addWidget(_hline(b_bonds, b_angles))
        self.cmb_center = QComboBox()
        self.cmb_center.addItems([CENTER_LABELS[c] for c in CENTERS])
        self.cmb_center.currentIndexChanged.connect(self.on_center_changed)
        v2.addWidget(_hline(QLabel("Bead centre"), self.cmb_center, 1))
        b_legacy = QPushButton("Import MD2CG v3 files (ndx + maps) ...")
        b_legacy.clicked.connect(self.import_legacy)
        v2.addWidget(b_legacy)
        self.lbl_valid = QLabel("")
        self.lbl_valid.setWordWrap(True)
        v2.addWidget(self.lbl_valid)
        L.addWidget(g2)

        g3 = self._section("3  DPD units")
        f3 = QFormLayout(g3.content)
        self.spn_rc = _dspin(0.5, 50.0, 6.46, 3, 0.01, " Å")
        self.spn_rc.setToolTip("Length of one DPD unit (r_c) in Angstrom")
        self.spn_nm = _dspin(0.1, 50, 3.0, 2)
        b_rc = QPushButton("Set")
        b_rc.setToolTip("r_c = (rho * N_m * 30 A^3)^(1/3)")
        b_rc.clicked.connect(lambda: self.spn_rc.setValue(rc_from_mapping(self.spn_nm.value(),
                                                                          self.spn_density.value())))
        f3.addRow("r_c", self.spn_rc)
        f3.addRow("from N_m", _hline(self.spn_nm, QLabel("waters/bead"), b_rc, 1))
        L.addWidget(g3)

        g4 = self._section("4  CG system", collapsed=True)
        f4 = QFormLayout(g4.content)
        self.cmb_cgstyle = QComboBox()
        for k, v in STYLES.items():
            self.cmb_cgstyle.addItem(v, k)
        self.cmb_cgstyle.currentIndexChanged.connect(self._update_style)
        f4.addRow("Style", self.cmb_cgstyle)
        self.cmb_angle_style = QComboBox()
        self.cmb_angle_style.addItems(list(ANGLE_STYLES))
        f4.addRow("Angle style", self.cmb_angle_style)
        self.cmb_box = QComboBox()
        self.cmb_box.addItems(["match the all-atom system", "custom box"])
        self.cmb_box.currentIndexChanged.connect(self._update_box)
        f4.addRow("Box", self.cmb_box)
        self.spn_rep = _ispin(1, 10, 1, suffix=" ×")
        self.spn_rep.setToolTip("Replicate the (scaled) all-atom box n x n x n times")
        self.lbl_rep = QLabel("Replicate")
        f4.addRow(self.lbl_rep, self.spn_rep)
        self.spn_nmol = _ispin(1, 10**6, 50)
        self.spn_boxl = _dspin(2.0, 1000.0, 10.0, 2, 1.0, " r_c")
        self.lbl_custom = QLabel("Molecules / box")
        self.row_custom = _hline(self.spn_nmol, self.spn_boxl)
        f4.addRow(self.lbl_custom, self.row_custom)
        self.spn_density = _dspin(0.1, 20.0, 3.0, 2)
        f4.addRow("Bead density", self.spn_density)
        self.chk_solvent = QCheckBox("Add solvent beads")
        self.chk_solvent.setChecked(True)
        self.ed_solvent = QLineEdit("W")
        self.ed_solvent.setMaximumWidth(60)
        f4.addRow(_hline(self.chk_solvent, QLabel("type"), self.ed_solvent, 1))
        self.chk_neutral = QCheckBox("Neutralise with counter-ion beads")
        self.chk_neutral.setChecked(True)
        f4.addRow(self.chk_neutral)
        self.spn_a = _dspin(-200, 500, 25.0, 2)
        self.spn_gamma = _dspin(0.01, 100, 4.5, 2)
        self.lbl_a = QLabel("a_ii default")
        f4.addRow(self.lbl_a, _hline(self.spn_a, QLabel("γ"), self.spn_gamma))
        self.spn_A = _dspin(-500, 0, -40.0, 2)
        self.spn_B = _dspin(0, 500, 25.0, 2)
        self.spn_rd = _dspin(0.1, 1.0, 0.75, 3)
        self.lbl_mdpd = QLabel("A / B / r_d")
        self.row_mdpd = _hline(self.spn_A, self.spn_B, self.spn_rd)
        f4.addRow(self.lbl_mdpd, self.row_mdpd)
        b_pairs = QPushButton("Pair coefficients ...")
        b_pairs.clicked.connect(self.edit_pairs)
        f4.addRow(b_pairs)
        self.ed_special = QLineEdit("1 1 1")
        self.ed_special.setToolTip("special_bonds lj weights (DPD convention: 1 1 1, bonded beads repel)")
        f4.addRow("special_bonds lj", self.ed_special)
        self.spn_dt = _dspin(0.0005, 0.1, 0.02, 4, 0.005)
        self.spn_equil = _ispin(0, 10**8, 5000, 1000)
        self.spn_prod = _ispin(100, 10**9, 50000, 5000)
        self.spn_dump = _ispin(1, 10**7, 100, 10)
        f4.addRow("Time step", self.spn_dt)
        f4.addRow("Equilibration", _hline(self.spn_equil, QLabel("steps")))
        f4.addRow("Production", _hline(self.spn_prod, QLabel("dump every"), self.spn_dump))
        self.spn_seed = _ispin(1, 2**30, 4928459)
        f4.addRow("Seed", self.spn_seed)
        L.addWidget(g4)

        g5 = self._section("5  Optimiser", collapsed=True)
        f5 = QFormLayout(g5.content)
        self.cmb_method = QComboBox()
        self.cmb_method.addItems(["ibi+pso", "ibi", "pso"])
        self.cmb_method.setToolTip("ibi: moment matching only; pso: Swarm-CG PSO cycles; ibi+pso: both")
        f5.addRow("Method", self.cmb_method)
        self.spn_ibi = _ispin(0, 100, 6)
        self.spn_alpha = _dspin(0.05, 1.5, 0.7, 2, 0.05)
        f5.addRow("IBI iterations", _hline(self.spn_ibi, QLabel("α"), self.spn_alpha))
        self.spn_swarm = _ispin(0, 500, 0)
        self.spn_swarm.setSpecialValueText("auto")
        self.spn_steps = _ispin(0, 500, 0)
        self.spn_steps.setSpecialValueText("auto")
        self.spn_patience = _ispin(1, 100, 4)
        f5.addRow("PSO swarm", _hline(self.spn_swarm, QLabel("steps"), self.spn_steps))
        f5.addRow("Patience", self.spn_patience)
        self.ed_cycles = QLineEdit("all 3 0.2 20; angles 2 0.1 10; all 1.4 0.06 5")
        self.ed_cycles.setToolTip("PSO cycles separated by ';' : what K-factor r0-fraction theta-range(deg)\n"
                                  "what = all | bonds | angles")
        f5.addRow("Cycles", self.ed_cycles)
        self.cmb_score = QComboBox()
        self.cmb_score.addItems(["normalized (EMD / σ_AA)", "absolute (Swarm-CG)"])
        self.spn_bscale = _dspin(0.1, 1000, 10.0, 1, suffix=" °/Å")
        f5.addRow("Score", _hline(self.cmb_score, self.spn_bscale))
        L.addWidget(g5)

        g6 = self._section("6  LAMMPS and output")
        f6 = QFormLayout(g6.content)
        self.ed_lmp = QLineEdit()
        self.ed_lmp.setPlaceholderText(find_lammps() or "lmp")
        f6.addRow("Executable", self._file_row(self.ed_lmp, self.browse_lmp))
        ncpu = os.cpu_count() or 4
        self.spn_jobs = _ispin(1, 256, max(1, ncpu // 2))
        self.spn_threads = _ispin(1, 64, 1)
        f6.addRow("Parallel runs", _hline(self.spn_jobs, QLabel("threads each"), self.spn_threads))
        self.ed_extra = QLineEdit()
        self.ed_extra.setPlaceholderText("extra arguments, e.g. -sf omp -pk omp 2")
        f6.addRow("Arguments", self.ed_extra)
        self.chk_keep = QCheckBox("Keep CG trajectories")
        f6.addRow(self.chk_keep)
        self.ed_out = QLineEdit(str(Path.cwd() / "cgdpd_out"))
        f6.addRow("Output", self._file_row(self.ed_out, self.browse_out))
        L.addWidget(g6)

        self.btn_ref = QPushButton("Compute AA distributions")
        self.btn_ref.clicked.connect(self.compute_reference)
        self.btn_run = QPushButton("Run fit")
        self.btn_run.setStyleSheet("font-weight: bold")
        self.btn_run.clicked.connect(self.run_fit)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop)
        self.btn_stop.setEnabled(False)
        L.addWidget(_hline(self.btn_ref, self.btn_run, self.btn_stop))
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        L.addWidget(self.progress)
        L.addStretch(1)

        self.wheel = NoWheelFilter(self)
        block_wheel(left, self.wheel)
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setWidget(left)
        sa.setMinimumWidth(200)
        dl = QDockWidget("Setup", self)
        dl.setObjectName("setup")
        dl.setWidget(sa)
        dl.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.LeftDockWidgetArea, dl)
        self.resizeDocks([dl], [440], Qt.Horizontal)

        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumBlockCount(20000)
        db = QDockWidget("Log", self)
        db.setObjectName("log")
        db.setWidget(self.txt_log)
        db.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.BottomDockWidgetArea, db)
        self.resizeDocks([db], [170], Qt.Vertical)

        # signals
        self.aa_view.atomClicked.connect(self.on_aa_clicked)
        self.aa_view.atomsBoxSelected.connect(self.on_aa_box)
        self.cg_view.atomClicked.connect(self.on_cg_clicked)
        for w in (self.chk_h, self.chk_aa_labels, self.chk_cg_labels, self.chk_skeleton):
            w.toggled.connect(lambda _on: self.refresh_views(False))
        self.cmb_color.currentIndexChanged.connect(lambda _i: self.refresh_views(False))
        self.spn_instance.valueChanged.connect(lambda _v: self.refresh_views(False))
        self.btn_draw.toggled.connect(self._draw_toggled)
        for k in range(9):
            QShortcut(QKeySequence(str(k + 1)), self, activated=lambda k=k: self.set_active_bead(k))
        for w in (self.spn_start, self.spn_stop, self.spn_stride, self.spn_maxmol):
            w.valueChanged.connect(self.invalidate_reference)
        self._update_format()
        self._update_style()
        self._update_box()

    def _topo_table(self, headers) -> QTableWidget:
        t = QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        t.verticalHeader().setVisible(False)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        t.itemChanged.connect(self.on_group_edited)
        return t

    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        for text, slot, key in (("Open project ...", self.load_project, QKeySequence.Open),
                                ("Save project ...", self.save_project, QKeySequence.Save)):
            a = QAction(text, self)
            a.setShortcut(key)
            a.triggered.connect(slot)
            m.addAction(a)
        m.addSeparator()
        a = QAction("Quit", self)
        a.setShortcut(QKeySequence.Quit)
        a.triggered.connect(self.close)
        m.addAction(a)
        h = self.menuBar().addMenu("&Help")
        a = QAction("How to use", self)
        a.triggered.connect(lambda: QMessageBox.information(self, "cgdpd-fit", HELP))
        h.addAction(a)

    # ============================================================= helpers
    @Slot(str)
    def log(self, msg: str):
        self.txt_log.appendPlainText(msg)

    def error(self, title: str, msg: str):
        self.log(f"ERROR: {title}: {msg.splitlines()[0] if msg else ''}")
        QMessageBox.critical(self, title, msg)

    def _dir(self) -> str:
        return self.settings.value("last_dir", str(Path.cwd()))

    def _remember(self, path: str):
        self.settings.setValue("last_dir", str(Path(path).parent))

    def _browse(self, edit: QLineEdit, title: str, filt: str):
        f, _ = QFileDialog.getOpenFileName(self, title, self._dir(), filt)
        if f:
            edit.setText(f)
            self._remember(f)

    def browse_data(self):
        if self.cmb_format.currentIndex() == 0:
            self._browse(self.ed_data, "LAMMPS data file", "LAMMPS data (*.data *.lmp *.dat *);;All (*)")
        else:
            self._browse(self.ed_data, "GROMACS run input", "tpr (*.tpr);;All (*)")

    def browse_dump(self):
        if self.cmb_format.currentIndex() == 0:
            self._browse(self.ed_dump, "LAMMPS trajectory", "dump (*.lammpstrj *.dump *.gz *);;All (*)")
        else:
            self._browse(self.ed_dump, "GROMACS trajectory", "trajectory (*.xtc *.trr);;All (*)")

    def browse_lmp(self):
        self._browse(self.ed_lmp, "LAMMPS executable", "All (*)")

    def browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Output folder", self.ed_out.text() or self._dir())
        if d:
            self.ed_out.setText(d)

    @property
    def model(self) -> CGModel | None:
        return self.models.get(self.cur) if self.cur is not None else None

    @property
    def species(self):
        return self.topo.species[self.cur] if self.topo is not None and self.cur is not None else None

    def _busy(self) -> bool:
        return self.thread is not None

    def _update_enabled(self):
        busy = self._busy()
        have = self.model is not None and self.model.n_beads > 0
        self.btn_ref.setEnabled(have and not busy)
        self.btn_run.setEnabled(have and not busy)
        self.btn_stop.setEnabled(busy)
        self.btn_load.setEnabled(not busy)

    def _update_format(self):
        lmp = self.cmb_format.currentIndex() == 0
        self.lbl_data.setText("Data file" if lmp else "tpr")
        self.lbl_dump.setText("Trajectory" if lmp else "xtc / trr")
        self.row_style.setEnabled(lmp)

    def _update_style(self):
        st = self.cmb_cgstyle.currentData()
        md = st == "mdpd"
        for w in (self.lbl_mdpd, self.row_mdpd):
            w.setVisible(md)
        self.lbl_a.setText("γ (a_ij unused)" if md else "a_ii default")
        self.spn_a.setEnabled(not md)
        self.chk_neutral.setVisible(st == "dpd_q")
        # sensible defaults when switching
        if md and abs(self.spn_dt.value() - 0.02) < 1e-9:
            self.spn_dt.setValue(0.01)
            self.spn_density.setValue(6.0)
        elif not md and abs(self.spn_dt.value() - 0.01) < 1e-9 and abs(self.spn_density.value() - 6.0) < 1e-9:
            self.spn_dt.setValue(0.02)
            self.spn_density.setValue(3.0)

    def _update_box(self):
        match = self.cmb_box.currentIndex() == 0
        for w in (self.lbl_rep, self.spn_rep):
            w.setVisible(match)
        for w in (self.lbl_custom, self.row_custom):
            w.setVisible(not match)

    # ============================================================= loading
    def aa_input(self) -> AAInput:
        if self.cmb_format.currentIndex() == 0:
            return AAInput("lammps", self.ed_data.text().strip(), self.ed_dump.text().strip(),
                           self.cmb_style.currentText(), self.cmb_split.currentText())
        return AAInput("gromacs", tpr=self.ed_data.text().strip(), traj=self.ed_dump.text().strip())

    def load_aa(self, keep_models: bool = False):
        aa = self.aa_input()
        if aa.format == "gromacs" and not aa.traj:
            self.error("GROMACS input", "Select the trajectory (.xtc / .trr) together with the .tpr:\n"
                       "the .tpr holds the topology, the coordinates come from the trajectory.")
            return
        try:
            self.topo = load_topology(aa)
        except Exception as exc:  # noqa: BLE001
            self.error("Cannot read the all-atom system", str(exc))
            return
        if not keep_models:
            self.models = {}
        self.cur = None
        self.invalidate_reference()
        info = f"{self.topo.n_atoms} atoms, {len(self.topo.species)} species"
        try:
            if (aa.dump if aa.format == "lammps" else aa.traj):
                n = count_frames(aa)
                info += f", {n} frames"
                self.spn_start.setMaximum(max(0, n - 1))
                self.spn_stop.setMaximum(n)
        except Exception as exc:  # noqa: BLE001
            info += f"\ntrajectory: {exc}"
        self.lbl_aa_info.setText(info)
        self.log(f"Loaded {aa.data or aa.tpr}: {info}")
        t = self.tbl_species
        t.blockSignals(True)
        t.setRowCount(len(self.topo.species))
        for i, sp in enumerate(self.topo.species):
            q = float(self.topo.charges[sp.instances[0]].sum())
            for j, v in enumerate((sp.name, sp.n_atoms, sp.count, f"{q:+.2f}")):
                it = QTableWidgetItem(str(v))
                it.setToolTip(sp.name)
                t.setItem(i, j, it)
        t.blockSignals(False)
        # preselect the largest non-water species
        pick = next((i for i, sp in enumerate(self.topo.species) if not sp.is_water and sp.n_atoms > 1), 0)
        if self.topo.species:
            t.selectRow(pick)

    def on_species_selected(self):
        rows = self.tbl_species.selectionModel().selectedRows() if self.tbl_species.selectionModel() else []
        if not rows or self.topo is None:
            return
        self.cur = rows[0].row()
        sp = self.species
        if self.cur not in self.models:
            self.models[self.cur] = CGModel(beads=[Bead("B1", "B1")])
        self.spn_instance.blockSignals(True)
        self.spn_instance.setMaximum(sp.count)
        self.spn_instance.setValue(1)
        self.spn_instance.blockSignals(False)
        self.cmb_center.blockSignals(True)
        self.cmb_center.setCurrentIndex(CENTERS.index(self.model.center))
        self.cmb_center.blockSignals(False)
        self.active_bead = 0
        self.invalidate_reference()
        self.refresh_all(reset=True)

    # ============================================================= mapping
    def _edit(self, fn, topology_changed: bool = False):
        if self.model is None:
            return
        fn(self.model)
        self.invalidate_reference()
        self.refresh_all(reset=False)

    def set_active_bead(self, k: int):
        if self.model is None or not (0 <= k < self.model.n_beads):
            return
        self.active_bead = k
        self.tbl_beads.blockSignals(True)
        self.tbl_beads.selectRow(k)
        self.tbl_beads.blockSignals(False)
        self.refresh_views(False)

    def add_bead(self):
        m = self.model
        if m is None:
            return
        names = {b.name for b in m.beads}
        k = m.n_beads + 1
        while f"B{k}" in names:
            k += 1
        m.beads.append(Bead(f"B{k}", f"B{k}"))
        self.active_bead = m.n_beads - 1
        self.refresh_all(False)

    def remove_bead(self):
        m = self.model
        if m is None or m.n_beads == 0:
            return
        k = self.active_bead
        del m.beads[k]
        keep_b = [(b, g) for b, g in zip(m.bonds, m.bond_groups) if k not in b]
        m.bonds = [tuple(i - (i > k) for i in b) for b, _ in keep_b]
        m.bond_groups = [g for _, g in keep_b]
        keep_a = [(a, g) for a, g in zip(m.angles, m.angle_groups) if k not in a]
        m.angles = [tuple(i - (i > k) for i in a) for a, _ in keep_a]
        m.angle_groups = [g for _, g in keep_a]
        self.active_bead = max(0, min(k, m.n_beads - 1))
        self.invalidate_reference()
        self.refresh_all(False)

    def clear_bead(self):
        if self.model and self.model.n_beads:
            self._edit(lambda m: setattr(m.beads[self.active_bead], "atoms", []))

    def auto_map(self):
        if self.species is None:
            return
        center = self.model.center if self.model else "com"
        m = auto_chain_model(self.species, self.spn_auto.value())
        m.center = center
        self.models[self.cur] = m
        self.active_bead = 0
        self.invalidate_reference()
        self.refresh_all(reset=False)

    def bonds_from_atoms(self):
        self._edit(lambda m: m.bonds_from_atoms(self.species))

    def angles_from_bonds(self):
        self._edit(lambda m: m.angles_from_bonds())

    def regroup(self):
        self._edit(lambda m: (m.regroup_bonds(), m.regroup_angles()))

    def unique_groups(self):
        def f(m):
            m.bond_groups = [f"{m.beads[i].name}-{m.beads[j].name}" for i, j in m.bonds]
            m.angle_groups = [f"{m.beads[i].name}-{m.beads[j].name}-{m.beads[k].name}" for i, j, k in m.angles]
        self._edit(f)

    def remove_selected_terms(self):
        m = self.model
        if m is None:
            return
        if self.topo_tabs.currentIndex() == 0:
            rows = sorted({i.row() for i in self.tbl_bonds.selectedIndexes()}, reverse=True)
            for r in rows:
                del m.bonds[r]
                del m.bond_groups[r]
        else:
            rows = sorted({i.row() for i in self.tbl_angles.selectedIndexes()}, reverse=True)
            for r in rows:
                del m.angles[r]
                del m.angle_groups[r]
        self.invalidate_reference()
        self.refresh_all(False)

    def on_center_changed(self, i: int):
        if self.model is not None:
            self._edit(lambda m: setattr(m, "center", CENTERS[i]))

    def on_bead_row_selected(self):
        rows = self.tbl_beads.selectionModel().selectedRows()
        if rows:
            self.active_bead = rows[0].row()
            self.refresh_views(False)

    def on_bead_edited(self, item: QTableWidgetItem):
        m = self.model
        if m is None or item.row() >= m.n_beads:
            return
        b = m.beads[item.row()]
        txt = item.text().strip().replace(" ", "_")
        if not txt:
            self.refresh_bead_table()
            return
        if item.column() == 0:
            b.name = txt
        elif item.column() == 1:
            old = b.type
            b.type = txt
            # groups that were derived from the bead types follow the new type name
            if old != txt:
                m.regroup_bonds()
                m.regroup_angles()
                self.invalidate_reference()
        self.refresh_all(False)

    def on_group_edited(self, item: QTableWidgetItem):
        m = self.model
        if m is None or item.column() != 1:
            return
        t = item.tableWidget()
        txt = item.text().strip().replace(" ", "_")
        if not txt:
            return
        if t is self.tbl_bonds and item.row() < len(m.bond_groups):
            m.bond_groups[item.row()] = txt
        elif t is self.tbl_angles and item.row() < len(m.angle_groups):
            m.angle_groups[item.row()] = txt
        self.invalidate_reference()

    def on_aa_clicked(self, i: int):
        if self.model is None or self.model.n_beads == 0:
            return
        self._edit(lambda m: m.toggle(self.active_bead, i, self.species))

    def on_aa_box(self, atoms: list):
        if self.model is None or self.model.n_beads == 0:
            return
        heavy = [a for a in atoms if self.species.elements[a] != "H"] or atoms
        self._edit(lambda m: m.assign(self.active_bead, heavy, self.species))

    def _draw_toggled(self, on: bool):
        self._connect_first = None
        self.cg_view.set_info("Draw bonds: click two beads" if on else "")

    def on_cg_clicked(self, k: int):
        m = self.model
        if m is None:
            return
        if not self.btn_draw.isChecked():
            self.set_active_bead(k)
            return
        if self._connect_first is None:
            self._connect_first = k
            self.set_active_bead(k)
            self.cg_view.set_info(f"Draw bonds: {m.beads[k].name} - ?")
            return
        a, b = sorted((self._connect_first, k))
        self._connect_first = None
        self.cg_view.set_info("Draw bonds: click two beads")
        if a == b:
            return
        if (a, b) in m.bonds:
            i = m.bonds.index((a, b))
            del m.bonds[i]
            del m.bond_groups[i]
        else:
            m.bonds.append((a, b))
            m.bond_groups.append(m.bond_label((a, b)))
        self.invalidate_reference()
        self.refresh_all(False)

    # ============================================================== views
    def refresh_all(self, reset: bool):
        self.refresh_bead_table()
        self.refresh_topo_tables()
        self.refresh_views(reset)
        m, sp = self.model, self.species
        if m is not None and sp is not None:
            msgs = m.validate(sp)
            nb, na = len(m.bond_group_names()), len(m.angle_group_names())
            txt = f"{m.n_beads} beads, {len(m.bonds)} bonds in {nb} groups, {len(m.angles)} angles in {na} groups"
            if msgs:
                txt += "<br><span style='color:#c0392b'>" + "<br>".join(msgs) + "</span>"
            self.lbl_valid.setText(txt)
        self._update_enabled()

    def refresh_bead_table(self):
        t = self.tbl_beads
        m = self.model
        t.blockSignals(True)
        t.setRowCount(m.n_beads if m else 0)
        if m:
            sp = self.species
            for k, b in enumerate(m.beads):
                it = QTableWidgetItem(b.name)
                it.setIcon(_swatch(BEAD_COLORS[k % len(BEAD_COLORS)]))
                t.setItem(k, 0, it)
                t.setItem(k, 1, QTableWidgetItem(b.type))
                heavy = [a for a in b.atoms if sp.elements[a] != "H"]
                it = QTableWidgetItem(f"{len(heavy)} heavy + {len(b.atoms) - len(heavy)} H")
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                t.setItem(k, 2, it)
            if m.n_beads:
                t.selectRow(min(self.active_bead, m.n_beads - 1))
        t.blockSignals(False)

    def refresh_topo_tables(self):
        m = self.model
        means = {}
        if self.ref:
            means = {k: d.mean for k, d in self.ref.items()}
        for t, terms, groups, kind in ((self.tbl_bonds, m.bonds if m else [], m.bond_groups if m else [], "bond"),
                                       (self.tbl_angles, m.angles if m else [], m.angle_groups if m else [], "angle")):
            t.blockSignals(True)
            t.setRowCount(len(terms))
            for r, (term, g) in enumerate(zip(terms, groups)):
                it = QTableWidgetItem("-".join(m.beads[i].name for i in term))
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                t.setItem(r, 0, it)
                t.setItem(r, 1, QTableWidgetItem(g))
                v = means.get(f"{kind}:{g}")
                it = QTableWidgetItem("" if v is None else f"{v:.2f}")
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                t.setItem(r, 2, it)
            t.blockSignals(False)

    def _aa_colors(self) -> list[QColor]:
        sp, m = self.species, self.model
        if self.cmb_color.currentIndex() == 1 or m is None:
            return [QColor(ELEMENT_COLORS.get(e, "#c080c0")) for e in sp.elements]
        own = m.owner(sp.n_atoms)
        out = []
        for a, k in enumerate(own):
            if k < 0:
                c = QColor("#b8b8b8")
            else:
                c = QColor(BEAD_COLORS[k % len(BEAD_COLORS)])
            if sp.elements[a] == "H":
                c = c.lighter(150)
            out.append(c)
        return out

    def refresh_views(self, reset: bool):
        sp, m = self.species, self.model
        if sp is None:
            self.aa_view.clear("Load the all-atom system and select a species")
            self.cg_view.clear("Define beads for the selected species")
            return
        inst = min(self.spn_instance.value() - 1, sp.count - 1)
        x, beads = reference_beads(self.topo, sp, m or CGModel(), inst)
        self._aa_atoms = x
        radii = np.array([0.28 if e == "H" else 0.5 for e in sp.elements])
        labels = [f"{a + 1}{e}" for a, e in enumerate(sp.elements)]
        self.aa_view.set_molecule(x, radii, self._aa_colors(), sp.bonds, labels, reset_view=reset)
        vis = np.array([self.chk_h.isChecked() or e != "H" for e in sp.elements])
        self.aa_view.set_visible(vis)
        self.aa_view.set_show_labels(self.chk_aa_labels.isChecked())
        rings = {}
        if m is not None and m.n_beads:
            ring = QColor(self.palette().highlight().color())
            for a in m.beads[min(self.active_bead, m.n_beads - 1)].atoms:
                rings[a] = ring
        self.aa_view.set_rings(rings)
        if m is None or m.n_beads == 0:
            self.cg_view.clear("Add beads")
            return
        filled = [k for k, b in enumerate(m.beads) if b.atoms]
        pos = np.array(beads)
        for k, b in enumerate(m.beads):
            if not b.atoms:     # place empty beads next to the molecule so they can be clicked
                pos[k] = x.mean(0) + np.array([0.0, -6.0 - 2.5 * k, 0.0])
        rc = self.spn_rc.value()
        colors = [QColor(BEAD_COLORS[k % len(BEAD_COLORS)]) for k in range(m.n_beads)]
        self.cg_view.set_molecule(pos, np.full(m.n_beads, 0.22 * rc), colors, m.bonds,
                                  [b.name for b in m.beads], reset_view=reset or not filled)
        if reset:       # look at the CG molecule from the same side as at the atoms
            self.cg_view.rot = self.aa_view.rot.copy()
            self.cg_view.center = self.aa_view.center.copy()
        self.cg_view.set_alpha(150 if self.chk_skeleton.isChecked() else 255)
        self.cg_view.set_show_labels(self.chk_cg_labels.isChecked())
        self.cg_view.set_rings({min(self.active_bead, m.n_beads - 1): QColor(self.palette().highlight().color())})
        if self.chk_skeleton.isChecked():
            heavy = [a for a, e in enumerate(sp.elements) if e != "H"]
            idx = {a: k for k, a in enumerate(heavy)}
            hb = [(idx[a], idx[b]) for a, b in sp.bonds if a in idx and b in idx]
            self.cg_view.set_overlay(x[heavy], [QColor("#707070")] * len(heavy), hb, radius=0.18)
        else:
            self.cg_view.set_overlay(np.zeros((0, 3)), [], [])

    # ======================================================== distributions
    def invalidate_reference(self, *_):
        self.mapped = None
        self.ref = None

    def current_project(self) -> Project:
        p = Project()
        p.aa = self.aa_input()
        p.species = self.cur if self.cur is not None else 0
        p.species_formula = self.species.formula if self.species is not None else ""
        p.model = CGModel.from_dict(self.model.to_dict()) if self.model else CGModel()
        p.frames = FrameSelection(self.spn_start.value(), self.spn_stop.value() or None,
                                  self.spn_stride.value(), self.spn_maxmol.value())
        p.units = DPDUnits(rc=self.spn_rc.value())
        c = CGSystemSettings.from_dict(self.cg_extra.to_dict())
        c.style = self.cmb_cgstyle.currentData()
        c.box_mode = "match" if self.cmb_box.currentIndex() == 0 else "custom"
        c.replicate = self.spn_rep.value()
        c.n_molecules = self.spn_nmol.value()
        c.box_length = self.spn_boxl.value()
        c.density = self.spn_density.value()
        c.solvent = self.chk_solvent.isChecked()
        c.solvent_name = self.ed_solvent.text().strip() or "W"
        c.neutralize = self.chk_neutral.isChecked()
        c.a_default = self.spn_a.value()
        c.gamma = self.spn_gamma.value()
        c.A_default = self.spn_A.value()
        c.B_default = self.spn_B.value()
        c.rd = self.spn_rd.value()
        try:
            c.special_lj = tuple(float(v) for v in self.ed_special.text().split()[:3])
            if len(c.special_lj) != 3:
                raise ValueError
        except ValueError:
            c.special_lj = (1.0, 1.0, 1.0)
        c.dt = self.spn_dt.value()
        c.n_equil = self.spn_equil.value()
        c.n_prod = self.spn_prod.value()
        c.dump_every = self.spn_dump.value()
        c.seed = self.spn_seed.value()
        p.cg = c
        p.angle_style = self.cmb_angle_style.currentText()
        p.scoring = ScoreSettings("normalized" if self.cmb_score.currentIndex() == 0 else "absolute",
                                  self.spn_bscale.value())
        o = OptimizerSettings()
        o.method = self.cmb_method.currentText()
        o.ibi_iterations = self.spn_ibi.value()
        o.ibi_alpha = self.spn_alpha.value()
        o.swarm_size = self.spn_swarm.value()
        o.max_steps = self.spn_steps.value()
        o.patience = self.spn_patience.value()
        o.cycles = self._parse_cycles(o.cycles)
        p.optimizer = o
        p.run = RunSettings(lammps_exe=self.ed_lmp.text().strip(), extra_args=self.ed_extra.text().strip(),
                            n_jobs=self.spn_jobs.value(), threads=self.spn_threads.value(),
                            keep_trajectories=self.chk_keep.isChecked())
        p.out_dir = self.ed_out.text().strip() or "cgdpd_out"
        return p

    def _parse_cycles(self, default):
        from ..core.optimize import CycleSpec
        out = []
        try:
            for part in self.ed_cycles.text().split(";"):
                t = part.split()
                if not t:
                    continue
                if t[0] not in ("all", "bonds", "angles"):
                    raise ValueError(t[0])
                out.append(CycleSpec(t[0], float(t[1]), float(t[2]), float(t[3])))
        except (ValueError, IndexError):
            self.log("warning: could not parse the PSO cycles, using the defaults")
            return default
        return out or default

    def _start(self, fn, on_done, label: str):
        # The worker lives in another thread: connect its signals to bound methods of this
        # window only (queued into the GUI thread); plain functions would run in the worker.
        self._job_done, self._job_label = on_done, label
        self.thread = QThread()
        self.worker = Worker(fn)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.log)
        self.worker.progress.connect(self.on_progress)
        self.worker.event.connect(self.on_event)
        self.worker.done.connect(self._on_job_done)
        self.worker.failed.connect(self._on_job_failed)
        self.worker.cancelled.connect(self._on_job_cancelled)
        self._update_enabled_busy(True)
        self.thread.start()

    @Slot(object)
    def _on_job_done(self, value):
        self._finished()
        try:
            self._job_done(value)
        except Exception as exc:  # noqa: BLE001
            self.error(self._job_label, f"{exc}\n\n{traceback.format_exc()}")

    @Slot(str)
    def _on_job_failed(self, msg: str):
        self._finished()
        self.error(f"{self._job_label} failed", msg)

    @Slot()
    def _on_job_cancelled(self):
        self._finished()
        self.log(f"{self._job_label} stopped")

    def _update_enabled_busy(self, busy: bool):
        self.btn_ref.setEnabled(not busy)
        self.btn_run.setEnabled(not busy)
        self.btn_load.setEnabled(not busy)
        self.btn_stop.setEnabled(busy)

    def _finished(self):
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait()
        self.thread = None
        self.worker = None
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self._update_enabled()

    def stop(self):
        if self.worker is not None:
            self.worker.request_stop()
            self.log("stopping ...")

    @Slot(str, int, int)
    def on_progress(self, stage: str, i: int, n: int):
        if n <= 0:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, n)
            self.progress.setValue(i)
        self.progress.setFormat(f"{stage} {i}/{n}" if n > 0 else stage)

    def compute_reference(self):
        if self.model is None or self.topo is None:
            return
        p = self.current_project()
        msgs = p.model.validate(self.species)
        if any(s.startswith(("no beads", "empty", "duplicate")) for s in msgs):
            self.error("Mapping", "\n".join(msgs))
            return
        topo = self.topo

        def task(w: Worker):
            w.log.emit(f"Mapping the trajectory of {self.species.name} ...")
            mt = map_aa(p, topo, lambda i, n: w.progress.emit("mapping", i, n or 0),
                        lambda: w.stop_requested)
            return mt

        def done(mt):
            self.mapped = mt
            self.ref = compute_distributions(mt.x, p.model)
            self.log(f"  {mt.n_frames} frames x {mt.n_molecules} molecules")
            p0 = initial_guess(self.ref, p.units, p.angle_style)
            self.show_reference(p0)
            self.refresh_topo_tables()
            self.tabs.setCurrentIndex(1)

        self._start(task, done, "Mapping")

    def show_reference(self, p0=None):
        if not self.ref:
            return
        cg = self.best.dists if self.best is not None and self.best.dists else {}
        cg = {k: v for k, v in cg.items() if k in self.ref}
        title = "all-atom (mapped)" + (f" vs best CG, score {self.best.score:.4f}" if cg else "")
        draw_distributions(self.fig_dist, self.ref, cg, title, self.best.terms if cg else None)
        self.canvas_dist.draw_idle()
        if p0 is not None:
            lines = [f"{k}: {d.mean:.3f} ± {d.std:.3f}" for k, d in self.ref.items()]
            self.lbl_dist.setText("  |  ".join(lines))
            self.fill_params(p0, {})

    def fill_params(self, params, terms):
        rows = [(f"bond {g}", K, x, terms.get(f"bond:{g}")) for g, (K, x) in params.bonds.items()]
        rows += [(f"angle {g}", K, x, terms.get(f"angle:{g}")) for g, (K, x) in params.angles.items()]
        t = self.tbl_params
        t.setRowCount(len(rows))
        for r, (name, K, x, s) in enumerate(rows):
            for c, v in enumerate((name, f"{K:.4g}", f"{x:.4g}", "" if s is None else f"{s:.3f}")):
                t.setItem(r, c, QTableWidgetItem(v))

    # =================================================================== fit
    def run_fit(self):
        if self.model is None or self.topo is None:
            return
        p = self.current_project()
        msgs = p.model.validate(self.species)
        if any(s.startswith(("no beads", "empty", "duplicate", "group labels")) for s in msgs):
            self.error("Mapping", "\n".join(msgs))
            return
        try:
            p.run.exe()
        except FileNotFoundError as exc:
            self.error("LAMMPS", str(exc))
            return
        out = Path(p.out_dir)
        if (out / "runs").exists() and any((out / "runs").iterdir()):
            r = QMessageBox.question(self, "Output folder", f"{out} already contains runs.\n"
                                     "Continue and add to it (old runs are kept)?")
            if r != QMessageBox.Yes:
                return
        self.best = None
        self.history = []
        topo, mapped = self.topo, self.mapped
        self.tabs.setCurrentIndex(2)

        def task(w: Worker):
            job = FitJob(p, topo, mapped, log=w.log.emit,
                         progress=lambda s, i, n: w.progress.emit(s, i, n),
                         on_ref=lambda ref: w.event.emit("ref", ref),
                         on_result=lambda r: w.event.emit("result", r),
                         on_best=lambda r: w.event.emit("best", r))
            w.job = job
            if w.stop_requested:
                job.stop()
            res = job.run()
            return (job.mapped, res)

        def done(val):
            mt, res = val
            self.mapped = mt
            if res.best is not None:
                self.log(f"Done: best score {res.best.score:.4f}; results in {res.out_dir / 'bestfit'}")
            else:
                self.log("Done: no successful simulation")

        self._start(task, done, "Fit")

    @Slot(str, object)
    def on_event(self, kind: str, obj):
        if kind == "ref":
            self.ref = obj
            self.refresh_topo_tables()
            self.show_reference()
        elif kind == "result":
            self.history.append(obj)
            if len(self.history) % 2 == 0 or len(self.history) < 10:
                draw_history(self.fig_hist, self.history)
                self.canvas_hist.draw_idle()
            if not obj.ok:
                self.log(f"  simulation e{obj.index:04d} failed: {obj.message}")
        elif kind == "best":
            self.best = obj
            self.fill_params(obj.params, obj.terms)
            self.show_reference()
            draw_history(self.fig_hist, self.history)
            self.canvas_hist.draw_idle()

    # ============================================================ pair table
    def edit_pairs(self):
        m = self.model
        types = list(m.bead_types) if m else []
        p = self.current_project()
        if p.cg.solvent and p.cg.solvent_name not in types:
            types.append(p.cg.solvent_name)
        if p.cg.style == "dpd_q" and p.cg.neutralize:
            types += ["CIp", "CIm"]
        if not types:
            return
        c = self.cg_extra
        if p.cg.style == "mdpd":
            tables = {"A (attraction)": (c.A_ij, self.spn_A.value()), "B (repulsion)": (c.B_ij, self.spn_B.value())}
        else:
            tables = {"a_ij": (c.a_ij, self.spn_a.value())}
        dlg = PairDialog(types, tables, self)
        if dlg.exec():
            vals = dlg.values()
            if p.cg.style == "mdpd":
                c.A_ij.update(vals["A (attraction)"])
                c.B_ij.update(vals["B (repulsion)"])
            else:
                c.a_ij.update(vals["a_ij"])

    # =========================================================== legacy import
    def import_legacy(self):
        if self.topo is None:
            self.error("Import", "Load the all-atom system first")
            return
        ndx, _ = QFileDialog.getOpenFileName(self, "cg_map.ndx", self._dir(), "ndx (*.ndx);;All (*)")
        if not ndx:
            return
        bonds, _ = QFileDialog.getOpenFileName(self, "bonds_map.txt (cancel: derive from atoms)",
                                               str(Path(ndx).parent), "txt (*.txt);;All (*)")
        angles, _ = QFileDialog.getOpenFileName(self, "angles_map.txt (cancel: derive from bonds)",
                                                str(Path(ndx).parent), "txt (*.txt);;All (*)")
        from ..io.legacy import read_legacy
        try:
            si, model = read_legacy(self.topo, ndx, bonds or None, angles or None)
        except Exception as exc:  # noqa: BLE001
            self.error("Import", str(exc))
            return
        self.models[si] = model
        self.tbl_species.selectRow(si)
        self.cur = si
        self.invalidate_reference()
        self.refresh_all(reset=True)
        self.log(f"Imported {ndx}: {model.n_beads} beads")

    # ============================================================== projects
    def save_project(self):
        f, _ = QFileDialog.getSaveFileName(self, "Save project", self._dir(), "Project (*.json)")
        if not f:
            return
        if not f.endswith(".json"):
            f += ".json"
        self.current_project().save(f)
        self._remember(f)
        self.log(f"Saved {f}")

    def load_project(self, path: str | None = None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Open project", self._dir(), "Project (*.json)")
            if not path:
                return
        try:
            p = Project.load(path)
        except Exception as exc:  # noqa: BLE001
            self.error("Open project", str(exc))
            return
        self._remember(path)
        self.apply_project(p)

    def apply_project(self, p: Project):
        self.cmb_format.setCurrentIndex(0 if p.aa.format == "lammps" else 1)
        self.ed_data.setText(p.aa.data if p.aa.format == "lammps" else p.aa.tpr)
        self.ed_dump.setText(p.aa.dump if p.aa.format == "lammps" else p.aa.traj)
        self.cmb_style.setCurrentText(p.aa.atom_style)
        self.cmb_split.setCurrentText(p.aa.split)
        self.spn_rc.setValue(p.units.rc)
        c = p.cg
        self.cg_extra = CGSystemSettings.from_dict(c.to_dict())
        self.cmb_cgstyle.setCurrentIndex(max(0, self.cmb_cgstyle.findData(c.style)))
        self.cmb_box.setCurrentIndex(0 if c.box_mode == "match" else 1)
        self.spn_rep.setValue(c.replicate)
        self.spn_nmol.setValue(c.n_molecules)
        self.spn_boxl.setValue(c.box_length)
        self.spn_density.setValue(c.density)
        self.chk_solvent.setChecked(c.solvent)
        self.ed_solvent.setText(c.solvent_name)
        self.chk_neutral.setChecked(c.neutralize)
        self.spn_a.setValue(c.a_default)
        self.spn_gamma.setValue(c.gamma)
        self.spn_A.setValue(c.A_default)
        self.spn_B.setValue(c.B_default)
        self.spn_rd.setValue(c.rd)
        self.ed_special.setText(" ".join(f"{v:g}" for v in c.special_lj))
        self.spn_dt.setValue(c.dt)
        self.spn_equil.setValue(c.n_equil)
        self.spn_prod.setValue(c.n_prod)
        self.spn_dump.setValue(c.dump_every)
        self.spn_seed.setValue(c.seed)
        self.cmb_angle_style.setCurrentText(p.angle_style)
        self.cmb_score.setCurrentIndex(0 if p.scoring.mode == "normalized" else 1)
        self.spn_bscale.setValue(p.scoring.bond_scale)
        o = p.optimizer
        self.cmb_method.setCurrentText(o.method)
        self.spn_ibi.setValue(o.ibi_iterations)
        self.spn_alpha.setValue(o.ibi_alpha)
        self.spn_swarm.setValue(o.swarm_size)
        self.spn_steps.setValue(o.max_steps)
        self.spn_patience.setValue(o.patience)
        self.ed_cycles.setText("; ".join(f"{cy.what} {cy.k_factor:g} {cy.r0_frac:g} {cy.theta_deg:g}"
                                         for cy in o.cycles))
        r = p.run
        self.ed_lmp.setText(r.lammps_exe)
        self.ed_extra.setText(r.extra_args)
        self.spn_jobs.setValue(r.n_jobs)
        self.spn_threads.setValue(r.threads)
        self.chk_keep.setChecked(r.keep_trajectories)
        self.ed_out.setText(p.out_dir)
        self.load_aa()
        if self.topo is None:
            return
        if p.species >= len(self.topo.species):
            self.error("Open project", "the species of the project does not exist in this system")
            return
        if p.species_formula and self.topo.species[p.species].formula != p.species_formula:
            self.log(f"warning: species {p.species + 1} is {self.topo.species[p.species].formula}, "
                     f"project expects {p.species_formula}")
        self.models = {p.species: p.model}
        self.tbl_species.selectRow(p.species)
        self.cur = p.species
        self.spn_start.setValue(p.frames.start)
        self.spn_stop.setValue(p.frames.stop or 0)
        self.spn_stride.setValue(p.frames.stride)
        self.spn_maxmol.setValue(p.frames.max_molecules)
        self.cmb_center.blockSignals(True)
        self.cmb_center.setCurrentIndex(CENTERS.index(p.model.center))
        self.cmb_center.blockSignals(False)
        self.invalidate_reference()
        self.refresh_all(reset=True)

    def closeEvent(self, ev):
        if self.worker is not None:
            r = QMessageBox.question(self, "Quit", "A job is running. Stop it and quit?")
            if r != QMessageBox.Yes:
                ev.ignore()
                return
            self.worker.request_stop()
            if self.thread is not None:
                self.thread.quit()
                self.thread.wait(5000)
        ev.accept()
