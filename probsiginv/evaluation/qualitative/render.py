from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def draw_plot_spec(ax: Any, plot_spec: dict[str, Any]) -> None:
    for band in plot_spec.get("bands", []):
        ax.fill_between(
            band["x"],
            band["lower"],
            band["upper"],
            color=band.get("color"),
            alpha=band.get("alpha", 0.15),
            linewidth=0.0,
        )

    for line in plot_spec.get("lines", []):
        ax.plot(
            line["x"],
            line["y"],
            color=line.get("color"),
            alpha=line.get("alpha", 1.0),
            linewidth=line.get("linewidth", 1.4),
            linestyle=line.get("linestyle", "-"),
            label=line.get("label"),
        )

    if plot_spec.get("title"):
        ax.set_title(plot_spec["title"])
    if plot_spec.get("xlabel"):
        ax.set_xlabel(plot_spec["xlabel"])
    if plot_spec.get("ylabel"):
        ax.set_ylabel(plot_spec["ylabel"])
    if plot_spec.get("xlim") is not None:
        ax.set_xlim(*plot_spec["xlim"])
    if plot_spec.get("ylim") is not None:
        ax.set_ylim(*plot_spec["ylim"])
    if plot_spec.get("box_aspect") is not None:
        ax.set_box_aspect(float(plot_spec["box_aspect"]))
    if plot_spec.get("grid", True):
        ax.grid(alpha=0.2)
    if any(line.get("label") for line in plot_spec.get("lines", [])):
        ax.legend(frameon=False, fontsize=8)


def render_plot_spec_png(
    *,
    plot_spec: dict[str, Any],
    output_path: str | Path,
    figsize: tuple[float, float] = (6.2, 5.2),
) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    subplots = plot_spec.get("subplots")
    if subplots:
        n_subplots = len(subplots)
        resolved_figsize = (figsize[0], max(figsize[1], 2.6 * n_subplots))
        fig, axes = plt.subplots(n_subplots, 1, figsize=resolved_figsize, constrained_layout=True)
        if n_subplots == 1:
            axes = [axes]
        for ax, subplot_spec in zip(axes, subplots):
            draw_plot_spec(ax, subplot_spec)
    else:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        draw_plot_spec(ax, plot_spec)
    fig.savefig(destination, dpi=200)
    plt.close(fig)


def render_panel_png(
    *,
    overlay_plot_spec: dict[str, Any],
    qq_plot_spec: dict[str, Any],
    acf_plot_spec: dict[str, Any],
    output_path: str | Path,
) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8.0, 4.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[3.0, 1.0], height_ratios=[1.0, 1.0])
    axes = (
        fig.add_subplot(grid[:, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 1]),
    )

    draw_plot_spec(axes[0], {**overlay_plot_spec, "box_aspect": 2.0 / 3.0})
    draw_plot_spec(axes[1], {**qq_plot_spec, "box_aspect": 1.0})
    draw_plot_spec(axes[2], {**acf_plot_spec, "box_aspect": 1.0})

    fig.savefig(destination, dpi=200)
    plt.close(fig)


__all__ = ["draw_plot_spec", "render_panel_png", "render_plot_spec_png"]
