from __future__ import annotations

import zlib
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from probsiginv.evaluation.qualitative.statistics import ACFCurve, QQCurve


def _style(plot_style: Mapping[str, Any], key: str) -> dict[str, Any]:
    return dict(plot_style[key])


def _baseline_order(plot_style: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(value).casefold() for value in plot_style["baseline_order"])


def _baseline_family(candidate_key: str, plot_style: Mapping[str, Any]) -> str:
    normalized = str(candidate_key).casefold()
    if normalized.startswith("reg_"):
        return "reg"
    for family in _baseline_order(plot_style):
        if normalized.startswith(family):
            return family
    return normalized


def baseline_sort_key(candidate_key: str, plot_style: Mapping[str, Any]) -> tuple[int, str]:
    family = _baseline_family(candidate_key, plot_style)
    order = {name: index for index, name in enumerate(_baseline_order(plot_style))}
    return order.get(family, len(order)), str(candidate_key).casefold()


def sorted_baseline_keys(candidate_keys: Iterable[str], plot_style: Mapping[str, Any]) -> list[str]:
    return sorted(candidate_keys, key=lambda key: baseline_sort_key(key, plot_style))


def _baseline_color(candidate_key: str, plot_style: Mapping[str, Any]) -> str:
    family = _baseline_family(candidate_key, plot_style)
    normalized_key = str(candidate_key).casefold()
    baseline_colors = {
        str(key).casefold(): str(value)
        for key, value in dict(plot_style["baseline_colors"]).items()
    }
    if normalized_key in baseline_colors:
        return baseline_colors[normalized_key]
    if family in baseline_colors:
        return baseline_colors[family]
    fallback_colors = tuple(str(color) for color in plot_style["baseline_fallback_colors"])
    fallback_index = max(
        0,
        baseline_sort_key(candidate_key, plot_style)[0] - len(_baseline_order(plot_style)),
    )
    return fallback_colors[fallback_index % len(fallback_colors)]


def line_style_for_candidate(
    candidate_key: str,
    candidate_kind: str | None,
    plot_style: Mapping[str, Any],
) -> dict[str, Any]:
    if candidate_kind == "baseline":
        style = _style(plot_style, "baseline")
        style["color"] = _baseline_color(candidate_key, plot_style)
        return style
    return _style(plot_style, "candidate")


def _display_conditioning(candidate_key: str) -> str:
    return str(candidate_key).replace("TLL3", "TLL").replace("_", "-")


def _display_baseline(candidate_key: str) -> str:
    normalized = str(candidate_key).casefold()
    if normalized.startswith("insertion"):
        return "Insertion"
    if normalized.startswith("fourier"):
        return "Fourier"
    if normalized.startswith("legendre"):
        return "Legendre"
    return _display_conditioning(str(candidate_key))


def display_label_for_candidate(candidate_key: str, candidate_kind: str | None) -> str:
    key = str(candidate_key)
    if candidate_kind == "baseline":
        if key.casefold().startswith(("reg_", "reg-")):
            return f"Reg-{_display_conditioning(key[4:])}"
        return _display_baseline(key)
    if candidate_kind == "generative":
        return f"Ours-{_display_conditioning(key)}"
    return _display_conditioning(key)


def _line(
    *,
    x: np.ndarray | Sequence[float],
    y: np.ndarray | Sequence[float],
    label: str | None,
    style: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "x": np.asarray(x, dtype=float).tolist(),
        "y": np.asarray(y, dtype=float).tolist(),
        "label": label,
        "color": style.get("color"),
        "linewidth": style.get("linewidth", 1.4),
        "alpha": style.get("alpha", 1.0),
        "linestyle": style.get("linestyle", "-"),
    }


def _companion_style(style: Mapping[str, Any]) -> dict[str, Any]:
    companion = dict(style)
    companion["linestyle"] = "--"
    companion["alpha"] = min(float(companion.get("alpha", 1.0)), 0.72)
    companion["linewidth"] = float(companion.get("linewidth", 1.4))
    return companion


def _append_acf_lines(
    lines: list[dict[str, Any]],
    *,
    curve: ACFCurve,
    label: str,
    style: Mapping[str, Any],
) -> None:
    rho = np.asarray(curve.rho, dtype=float)
    if rho.ndim == 2:
        for dim in range(rho.shape[0]):
            lines.append(_line(x=curve.tau, y=rho[dim], label=f"{label} dim{dim}", style=style))
        return

    lines.append(_line(x=curve.tau, y=rho, label=label, style=style))
    companion_style = _companion_style(style)
    for companion in curve.companions:
        lines.append(
            _line(
                x=companion.tau,
                y=companion.rho,
                label=f"{label} volatility",
                style=companion_style,
            )
        )


def _expanded_ylim(*path_groups: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([np.asarray(group, dtype=float).reshape(-1) for group in path_groups])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    y_min = float(np.min(finite))
    y_max = float(np.max(finite))
    span = y_max - y_min
    pad = max(abs(y_min) * 0.25, 1.0) if span <= 0.0 else 0.25 * span
    return y_min - pad, y_max + pad


def select_path_indices(n_paths: int, n_show: int) -> list[int]:
    if n_show <= 0:
        raise ValueError("n_show must be positive")
    if n_paths <= n_show:
        return list(range(n_paths))

    raw = np.rint(np.linspace(0, n_paths - 1, num=n_show)).astype(int)
    selected: list[int] = []
    seen: set[int] = set()
    for index in raw.tolist():
        if index not in seen:
            seen.add(index)
            selected.append(int(index))
    cursor = 0
    while len(selected) < n_show:
        if cursor not in seen:
            seen.add(cursor)
            selected.append(cursor)
        cursor += 1
    return selected


def sample_candidate_indices(
    n_total: int,
    n_select: int,
    *,
    path_idx: int,
    candidate_key: str,
    random_seed: int,
) -> list[int]:
    if n_select <= 0:
        raise ValueError("n_gen must be positive")
    if n_total <= n_select:
        return list(range(n_total))

    seed_key = f"{int(random_seed)}:{candidate_key}:{path_idx}"
    seed = zlib.crc32(seed_key.encode("utf-8")) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    return sorted(int(value) for value in rng.choice(n_total, size=n_select, replace=False).tolist())


def resolve_panel_targets(
    candidates: list[Any],
    *,
    fallback_candidates: list[Any] | None = None,
) -> tuple[list[Any], list[Any]]:
    by_key: dict[str, Any] = {}
    for candidate in list(candidates) + list(fallback_candidates or []):
        if candidate.kind == "baseline":
            by_key.setdefault(str(candidate.key).casefold(), candidate)

    probabilistic = [candidate for candidate in candidates if candidate.kind == "generative"]
    if not probabilistic:
        raise ValueError("qualitative plots require at least one probabilistic candidate")
    return list(by_key.values()), probabilistic


def build_candidate_baseline_routes(
    baselines: list[Any],
    candidates: list[Any],
    *,
    plot_style: Mapping[str, Any],
) -> dict[str, list[str]]:
    by_normalized_key = {str(baseline.key).casefold(): baseline for baseline in baselines}
    common = sorted(
        [
            baseline
            for baseline in baselines
            if baseline.method != "regression" and not str(baseline.key).casefold().startswith("reg_")
        ],
        key=lambda candidate: baseline_sort_key(candidate.key, plot_style),
    )

    routes: dict[str, list[str]] = {}
    for candidate in candidates:
        route = [baseline.key for baseline in common]
        regression = by_normalized_key.get(f"reg_{candidate.key}".casefold())
        if regression is not None:
            route.append(regression.key)
        routes[candidate.key] = route
    return routes


def build_overlay_plot_spec(
    *,
    time_grid: np.ndarray,
    real_path: np.ndarray,
    deterministic_paths: Mapping[str, np.ndarray],
    candidate_paths: np.ndarray,
    candidate_key: str,
    path_idx: int,
    plot_style: Mapping[str, Any],
) -> dict[str, Any]:
    time_values = np.asarray(time_grid, dtype=float)
    real_values = np.asarray(real_path, dtype=float)
    candidate_values = np.asarray(candidate_paths, dtype=float)
    deterministic_values = {
        key: np.asarray(deterministic_paths[key], dtype=float)
        for key in sorted_baseline_keys(deterministic_paths, plot_style)
    }

    if real_values.ndim == 2:
        subplots = [
            build_overlay_plot_spec(
                time_grid=time_values,
                real_path=real_values[:, dim],
                deterministic_paths={
                    key: path[:, dim]
                    for key, path in deterministic_values.items()
                },
                candidate_paths=candidate_values[:, :, dim],
                candidate_key=candidate_key,
                path_idx=path_idx,
                plot_style=plot_style,
            )
            for dim in range(real_values.shape[1])
        ]
        for dim, subplot in enumerate(subplots):
            subplot["title"] = f"path {path_idx} | {candidate_key} | dim{dim}"
            subplot["ylabel"] = f"dim{dim} value"
        return {
            "title": f"path {path_idx} | {candidate_key}",
            "subplots": subplots,
        }

    lines: list[dict[str, Any]] = []
    for sample_idx, sample_path in enumerate(candidate_values):
        lines.append(
            _line(
                x=time_values,
                y=sample_path,
                label=display_label_for_candidate(candidate_key, "generative") if sample_idx == 0 else None,
                style=_style(plot_style, "candidate_sample"),
            )
        )
    for key, path in deterministic_values.items():
        lines.append(
            _line(
                x=time_values,
                y=path,
                label=display_label_for_candidate(key, "baseline"),
                style=line_style_for_candidate(key, "baseline", plot_style),
            )
        )
    lines.append(_line(x=time_values, y=real_values, label="Real", style=_style(plot_style, "real")))

    return {
        "title": f"path {path_idx} | {candidate_key}",
        "xlabel": "time",
        "ylabel": "value",
        "bands": [],
        "lines": lines,
        "ylim": _expanded_ylim(real_values, candidate_values, *deterministic_values.values()),
        "grid": True,
    }


def build_qq_plot_spec(
    *,
    candidate_key: str,
    candidate_kind: str | None,
    qq_curve: QQCurve,
    baseline_curves: Mapping[str, QQCurve] | None,
    plot_style: Mapping[str, Any],
) -> dict[str, Any]:
    real_curve = np.asarray(qq_curve.real, dtype=float)
    candidate_curve = np.asarray(qq_curve.candidate, dtype=float)
    baseline_values = {
        key: np.asarray(curve.candidate, dtype=float)
        for key, curve in (baseline_curves or {}).items()
    }
    finite = np.concatenate(
        [real_curve.reshape(-1), candidate_curve.reshape(-1), *[values.reshape(-1) for values in baseline_values.values()]]
    )
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        diagonal = np.array([0.0, 1.0], dtype=float)
    else:
        diagonal = np.array([float(np.min(finite)), float(np.max(finite))], dtype=float)
        if diagonal[0] == diagonal[1]:
            pad = max(abs(float(diagonal[0])) * 0.25, 1.0)
            diagonal = np.array([diagonal[0] - pad, diagonal[1] + pad], dtype=float)

    lines = [_line(x=diagonal, y=diagonal, label="Real", style=_style(plot_style, "real"))]
    candidate_label = display_label_for_candidate(candidate_key, candidate_kind)
    candidate_style = line_style_for_candidate(candidate_key, candidate_kind, plot_style)
    if real_curve.ndim == 2:
        for dim in range(real_curve.shape[0]):
            lines.append(
                _line(
                    x=real_curve[dim],
                    y=candidate_curve[dim],
                    label=f"{candidate_label} dim{dim}",
                    style=candidate_style,
                )
            )
    else:
        lines.append(
            _line(
                x=real_curve,
                y=candidate_curve,
                label=candidate_label,
                style=candidate_style,
            )
        )
    for key in sorted_baseline_keys(baseline_values, plot_style):
        baseline_label = display_label_for_candidate(key, "baseline")
        baseline_style = line_style_for_candidate(key, "baseline", plot_style)
        if real_curve.ndim == 2:
            for dim in range(real_curve.shape[0]):
                lines.append(
                    _line(
                        x=real_curve[dim],
                        y=baseline_values[key][dim],
                        label=f"{baseline_label} dim{dim}",
                        style=baseline_style,
                    )
                )
        else:
            lines.append(
                _line(
                    x=real_curve,
                    y=baseline_values[key],
                    label=baseline_label,
                    style=baseline_style,
                )
            )

    return {
        "title": "QQ",
        "xlabel": "real increments",
        "ylabel": "candidate increments",
        "bands": [],
        "lines": lines,
        "xlim": tuple(diagonal.tolist()),
        "ylim": tuple(diagonal.tolist()),
        "box_aspect": 1.0,
        "grid": True,
    }


def build_acf_plot_spec(
    *,
    candidate_key: str,
    candidate_kind: str | None,
    real_curve: ACFCurve,
    candidate_curve: ACFCurve,
    baseline_curves: Mapping[str, ACFCurve] | None,
    plot_style: Mapping[str, Any],
) -> dict[str, Any]:
    lines: list[dict[str, Any]] = []
    _append_acf_lines(
        lines,
        curve=real_curve,
        label="Real",
        style=_style(plot_style, "real"),
    )
    _append_acf_lines(
        lines,
        curve=candidate_curve,
        label=display_label_for_candidate(candidate_key, candidate_kind),
        style=line_style_for_candidate(candidate_key, candidate_kind, plot_style),
    )
    for key in sorted_baseline_keys((baseline_curves or {}).keys(), plot_style):
        curve = baseline_curves[key]
        _append_acf_lines(
            lines,
            curve=curve,
            label=display_label_for_candidate(key, "baseline"),
            style=line_style_for_candidate(key, "baseline", plot_style),
        )

    companion_targets = {
        companion.target
        for curve in [real_curve, candidate_curve, *(baseline_curves or {}).values()]
        for companion in curve.companions
    }
    target_label = candidate_curve.target
    if companion_targets:
        target_label = " + ".join([candidate_curve.target, *sorted(companion_targets)])

    return {
        "title": f"ACF ({target_label}, {candidate_curve.center}-centered)",
        "xlabel": "tau",
        "ylabel": "acf",
        "bands": [],
        "lines": lines,
        "box_aspect": 1.0,
        "grid": True,
    }


__all__ = [
    "baseline_sort_key",
    "build_acf_plot_spec",
    "build_candidate_baseline_routes",
    "build_overlay_plot_spec",
    "build_qq_plot_spec",
    "display_label_for_candidate",
    "line_style_for_candidate",
    "resolve_panel_targets",
    "sample_candidate_indices",
    "select_path_indices",
    "sorted_baseline_keys",
]
