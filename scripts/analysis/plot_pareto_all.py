# -*- coding: utf-8 -*-
"""Compatibility wrapper for the current Paper 3 Pareto and bar figures."""

from __future__ import annotations

from paper3_cross_township_bars import main as render_cross_township_bars
from paper3_pareto_figures import main as render_pareto_figures


def main() -> None:
    render_pareto_figures()
    render_cross_township_bars()


if __name__ == "__main__":
    main()
