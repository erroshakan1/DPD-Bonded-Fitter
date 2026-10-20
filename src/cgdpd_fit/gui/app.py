"""GUI entry point."""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    from PySide6.QtCore import QLocale
    from PySide6.QtWidgets import QApplication

    from .main_window import MainWindow

    argv = sys.argv if argv is None else argv
    QLocale.setDefault(QLocale.c())   # always use '.' as decimal separator
    app = QApplication(argv)
    app.setApplicationName("cgdpd-fit")
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    if len(argv) > 1 and argv[1].endswith(".json"):
        w.load_project(argv[1])
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
