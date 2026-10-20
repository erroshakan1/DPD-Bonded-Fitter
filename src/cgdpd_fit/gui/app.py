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
    from pathlib import Path
    from PySide6.QtGui import QIcon
    # Wayland desktops take the taskbar icon from the matching cgdpd-fit.desktop entry
    # (installed by "cgdpd-fit desktop"); X11, Windows and macOS use the window icon.
    app.setDesktopFileName("cgdpd-fit")
    app.setWindowIcon(QIcon(str(Path(__file__).with_name("icon.png"))))
    w = MainWindow()
    w.show()
    if len(argv) > 1 and argv[1].endswith(".json"):
        w.load_project(argv[1])
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
