"""Application colours: a few accents on top of the Fusion palette (light or dark)."""
from __future__ import annotations

from PySide6.QtGui import QColor, QPalette

ACCENT = "#3b6ea5"          # steel blue, the all-atom colour of the plots
ACCENT_HOVER = "#4a80ba"
ACCENT_PRESSED = "#2f5a88"
DANGER = "#c44e52"          # the CG red of the plots


def stylesheet(dark: bool) -> str:
    header_bg = "rgba(59, 110, 165, 0.28)" if dark else "rgba(59, 110, 165, 0.12)"
    content_bg = "rgba(255, 255, 255, 0.03)" if dark else "rgba(255, 255, 255, 0.55)"
    border = "rgba(255, 255, 255, 0.14)" if dark else "rgba(0, 0, 0, 0.13)"
    return f"""
QToolButton#sectionHeader {{
    font-weight: bold;
    text-align: left;
    padding: 4px 6px;
    border: none;
    border-left: 3px solid {ACCENT};
    border-radius: 3px;
    background: {header_bg};
}}
QToolButton#sectionHeader:hover {{ background: rgba(59, 110, 165, 0.22); }}
QFrame#sectionContent {{
    border: 1px solid {border};
    border-radius: 4px;
    background: {content_bg};
}}
QDockWidget::title {{
    background: {ACCENT};
    color: white;
    padding: 4px 8px;
    font-weight: bold;
}}
QDockWidget {{ color: white; }}
QPushButton#runButton {{
    background: {ACCENT};
    color: white;
    font-weight: bold;
    border: 1px solid {ACCENT_PRESSED};
    border-radius: 4px;
    padding: 5px 14px;
}}
QPushButton#runButton:hover {{ background: {ACCENT_HOVER}; }}
QPushButton#runButton:pressed {{ background: {ACCENT_PRESSED}; }}
QPushButton#runButton:disabled {{ background: rgba(59, 110, 165, 0.35); border-color: transparent; }}
QPushButton#stopButton:enabled {{
    color: {DANGER};
    font-weight: bold;
}}
QProgressBar {{
    border: 1px solid {border};
    border-radius: 4px;
    text-align: center;
    min-height: 16px;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
QTabBar::tab:selected {{ color: {ACCENT if not dark else "#8db4de"}; font-weight: bold; }}
QLabel#hint {{ color: palette(placeholder-text); }}
"""


def apply(app) -> None:
    pal = app.palette()
    dark = pal.color(QPalette.Window).lightness() < 128
    pal.setColor(QPalette.Highlight, QColor(ACCENT))
    pal.setColor(QPalette.HighlightedText, QColor("white"))
    pal.setColor(QPalette.Link, QColor(ACCENT if not dark else "#8db4de"))
    app.setPalette(pal)
    app.setStyleSheet(stylesheet(dark))
