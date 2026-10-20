"""Dialog for the (fixed) non-bonded coefficients of the CG system used in the bonded fit."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QHeaderView, QLabel, QTableWidget,
                               QTableWidgetItem, QTabWidget, QVBoxLayout)


class PairDialog(QDialog):
    """Symmetric matrices of a_ij (DPD) or A_ij / B_ij (mDPD); empty cells use the default."""

    def __init__(self, types: list[str], tables: dict[str, tuple[dict, float]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pair coefficients")
        self.types = types
        self.tables: dict[str, QTableWidget] = {}
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Non-bonded coefficients stay fixed during the bonded fit. "
                             "Empty cells use the default value shown in the header tooltip."))
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
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.resize(120 + 80 * len(types), 160 + 34 * len(types))

    def _mirror(self, t: QTableWidget, item: QTableWidgetItem) -> None:
        i, j = item.row(), item.column()
        if j > i:
            other = t.item(j, i)
            if other is not None:
                t.blockSignals(True)
                other.setText(item.text())
                t.blockSignals(False)

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
