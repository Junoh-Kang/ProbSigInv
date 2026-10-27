from __future__ import annotations

from pathlib import Path


def _model_dir_name(model_npz_path: str | Path) -> str:
    path = Path(model_npz_path)
    parent_name = path.parent.name
    if "_d" in parent_name and ("_logsig" in parent_name or "_sig" in parent_name):
        return parent_name

    for ancestor in path.parents[1:3]:
        name = ancestor.name
        if "_d" in name and ("_logsig" in name or "_sig" in name):
            return name

    return parent_name


def detect_aug_type(folder_name: str) -> str:
    """Return augment_path-compatible aug type from folder name."""
    if "_TLL3_" in folder_name or "_lead-lag3_" in folder_name:
        return "TLL3"
    if "_TLL_" in folder_name or "_lead-lag_" in folder_name:
        return "TLL"
    if "_TA_" in folder_name or "_linear_" in folder_name:
        return "TA"
    if "_LS_" in folder_name:
        return "LS"
    return "TA"


_AUG_SHORT = {"TLL3": "tll3", "TLL": "tll", "TA": "ta", "LS": "ls"}
_AUG_DISPLAY = {
    "ls": "LS",
    "ta": "TA",
    "lin": "TA",
    "tll": "TLL",
    "ll": "TLL",
    "tll3": "TLL3",
    "ll3": "TLL3",
}


def model_label(model_npz_path: str | Path) -> str:
    path = Path(model_npz_path)
    name = _model_dir_name(path)
    aug_type = detect_aug_type(name)
    aug = _AUG_SHORT.get(aug_type, aug_type.lower())

    depth = "?"
    for part in name.split("_"):
        if part.startswith("d") and part[1:].isdigit():
            depth = part
            break

    transform = ""
    if "_raw_" in name or name.endswith("_raw"):
        transform = "raw"
    elif "level2_symlog" in name:
        transform = "lv2sym"
    elif "level_symlog" in name:
        transform = "lvsym"
    elif "level2" in name:
        transform = "lv2"
    elif "symlog" in name:
        transform = "sym"

    extra_tags: list[str] = []
    if "_full_" in name:
        extra_tags.append("full")
    if "_pruned" in name:
        extra_tags.append("pr")

    parts = [f"{aug}-{depth}"]
    if transform:
        parts.append(transform)
    parts.extend(extra_tags)
    return "-".join(parts)


def display_label(label: str) -> str:
    parts = label.split("-")
    aug = _AUG_DISPLAY.get(parts[0], parts[0])
    if len(parts) == 1:
        return aug
    return f"{aug}-{'-'.join(parts[1:])}"


def display_column(col_key: str) -> str:
    parts = col_key.split("-")
    aug = _AUG_DISPLAY.get(parts[0], parts[0])
    if len(parts) == 1:
        return aug
    return f"{aug}-{'-'.join(parts[1:])}"
