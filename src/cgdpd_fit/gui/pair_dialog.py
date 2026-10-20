"""Dialog for the (fixed) non-bonded coefficients of the CG system used in the bonded fit."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QGroupBox,
                               QHBoxLayout, QHeaderView, QLabel, QPushButton, QTableWidget,
                               QTableWidgetItem, QTabWidget, QVBoxLayout)

from ..core.flory import aij_from_solubility

FH_HELP = ("χ<sub>ij</sub> = V (δ<sub>i</sub> − δ<sub>j</sub>)² / RT,&nbsp;&nbsp; "
           "a<sub>ij</sub> = a<sub>ii</sub> + f χ<sub>ij</sub>. "
           "δ in MPa<sup>½</sup> (water 47.8, n-alkanes ≈ 15–16); V = r<sub>c</sub>³/ρ per bead; "
           "f = 3.27 at ρ = 3 (Groot &amp; Madden 1998). Types without δ are left unchanged.")


def _spin(lo, hi, val, dec, suffix="") -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setValue(val)
    s.setSuffix(suffix)
    return s


class PairDialog(QDialog):
    """Symmetric matrices of a_ij (DPD) or A_ij / B_ij (mDPD); empty cells use the default.

    For DPD the a_ij matrix can be filled from solubility parameters (Flory-Huggins chi).
    ``fh`` = dict(delta, temperature, factor, volume, a_ii) enables that section.
    """

    def __init__(self, types: list[str], tables: dict[str, tuple[dict, float]], parent=None,
                 fh: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Pair coefficients")
        self.types = types
        self.tables: dict[str, QTableWidget] = {}
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Non-bonded coefficients stay fixed during the bonded fit. "
                             "Empty cells use the default value."))
        tabs = QTabWidget()
        for name, (values, default) in tables.items():
            t = QTableWidget(len(types), len(types))
            t.setHorizontalHeaderLabels(types)
            t.setVerticalHeaderLabels(types)
            t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            t.setToolTip(f"default {default:g}")
            for i, a in enumerate(types):
                for j, b in enumerate(types):
                    v = values.get(f"{a} {b}", values.get(f"{b} {a}"))
                    it = QTableWidgetItem("" if v is None else f"{float(v):g}")
                    if j < i:                      # lower triangle mirrors the upper one
                        it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                    t.setItem(i, j, it)
            t.itemChanged.connect(lambda item, tt=t: self._mirror(tt, item))
            tabs.addTab(t, name)
            self.tables[name] = t
        lay.addWidget(tabs)

        self.fh = None
        if fh is not None and "a_ij" in self.tables:
            self.fh = fh
            box = QGroupBox("Fill a_ij from Flory-Huggins solubility parameters")
            bl = QVBoxLayout(box)
            help_ = QLabel(FH_HELP)
            help_.setWordWrap(True)
            bl.addWidget(help_)
            self.tbl_delta = QTableWidget(len(types), 2)
            self.tbl_delta.setHorizontalHeaderLabels(["Type", "δ (MPa½)"])
            self.tbl_delta.verticalHeader().setVisible(False)
            self.tbl_delta.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            for i, tname in enumerate(types):
                it = QTableWidgetItem(tname)
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                self.tbl_delta.setItem(i, 0, it)
                d = fh.get("delta", {}).get(tname)
                self.tbl_delta.setItem(i, 1, QTableWidgetItem("" if d is None else f"{float(d):g}"))
            self.tbl_delta.setMaximumHeight(60 + 30 * min(len(types), 6))
            bl.addWidget(self.tbl_delta)
            form = QFormLayout()
            self.spn_T = _spin(1, 2000, fh.get("temperature", 298.15), 2, " K")
            self.spn_V = _spin(1, 10000, fh.get("volume", 54.0), 2, " cm³/mol")
            self.spn_V.setToolTip("Molar volume of one bead, r_c^3 / rho (from the DPD units and density)")
            self.spn_aii = _spin(0, 500, fh.get("a_ii", 25.0), 2)
            self.spn_aii.setToolTip("Like-like repulsion: 25 for N_m = 1, ~78 for N_m = 3 (rho = 3)")
            self.spn_f = _spin(0.01, 100, fh.get("factor", 3.27), 3)
            row = QHBoxLayout()
            for lbl, w in (("T", self.spn_T), ("V", self.spn_V)):
                row.addWidget(QLabel(lbl))
                row.addWidget(w)
            form.addRow(row)
            row = QHBoxLayout()
            for lbl, w in (("a_ii", self.spn_aii), ("f", self.spn_f)):
                row.addWidget(QLabel(lbl))
                row.addWidget(w)
            form.addRow(row)
            bl.addLayout(form)
            btn = QPushButton("Fill a_ij table")
            btn.clicked.connect(self.fill_from_solubility)
            self.lbl_fh = QLabel("")
            self.lbl_fh.setWordWrap(True)
            bl.addWidget(btn)
            bl.addWidget(self.lbl_fh)
            lay.addWidget(box)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.resize(160 + 80 * len(types), 260 + 34 * len(types) + (300 if self.fh else 0))

    def _mirror(self, t: QTableWidget, item: QTableWidgetItem) -> None:
        i, j = item.row(), item.column()
        if j > i:
            other = t.item(j, i)
            if other is not None:
                t.blockSignals(True)
                other.setText(item.text())
                t.blockSignals(False)

    def deltas(self) -> dict[str, float]:
        out = {}
        if self.fh is None:
            return out
        for i, tname in enumerate(self.types):
            txt = self.tbl_delta.item(i, 1).text().strip()
            if txt:
                out[tname] = float(txt)
        return out

    def fill_from_solubility(self) -> None:
        try:
            delta = self.deltas()
        except ValueError:
            self.lbl_fh.setText("<span style='color:#c0392b'>δ values must be numbers</span>")
            return
        if len(delta) < 1:
            self.lbl_fh.setText("Enter δ for at least one type.")
            return
        a, x = aij_from_solubility(delta, self.spn_V.value(), self.spn_T.value(),
                                   self.spn_aii.value(), self.spn_f.value())
        t = self.tables["a_ij"]
        idx = {n: k for k, n in enumerate(self.types)}
        for key, v in a.items():
            t1, t2 = key.split()
            i, j = sorted((idx[t1], idx[t2]))
            item = t.item(i, j)
            item.setText(f"{v:.2f}")
            item.setToolTip(f"χ = {x[key]:.3f}")
            t.item(j, i).setToolTip(f"χ = {x[key]:.3f}")
        skipped = [n for n in self.types if n not in delta]
        msg = f"Filled {len(a)} pairs."
        if skipped:
            msg += " Without δ (unchanged): " + ", ".join(skipped)
        self.lbl_fh.setText(msg)

    def fh_settings(self) -> dict:
        if self.fh is None:
            return {}
        return {"delta": self.deltas(), "temperature": self.spn_T.value(), "factor": self.spn_f.value()}

    def values(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for name, t in self.tables.items():
            d = {}
            for i, a in enumerate(self.types):
                for j in range(i, len(self.types)):
                    txt = t.item(i, j).text().strip()
                    if txt:
                        d[f"{a} {self.types[j]}"] = float(txt)
            out[name] = d
        return out
