#!/usr/bin/env python3
"""Plot selected columns from GMT inference CSV logs."""

import argparse
import csv
import math
from pathlib import Path


def default_csv_path() -> Path:
    return Path("/home/robros/ros_ws/igris_c_gmt_public/log/inference_observation_action.csv")


def default_obs_yaml_path() -> Path:
    return Path("/home/robros/ros_ws/igris_c_gmt_public/config/obs.yaml")


OBS_TERM_SIZES = {
    "joint_pos": 23,
    "joint_pos_rel": 23,
    "joint_vel": 23,
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "last_actions": 23,
    "motion_joint_pos": 23,
    "motion_joint_vel": 23,
    "motion_body_pos": 42,
    "motion_body_orientation": 56,
    "motion_body_lin_vel": 42,
    "motion_body_ang_vel": 42,
    "motion_anchor_lin_vel": 3,
    "motion_anchor_ang_vel": 3,
    "motion_anchor_lin_vel_b": 3,
    "motion_anchor_ang_vel_b": 3,
    "motion_anchor_orientation": 6,
    "motion_anchor_height": 1,
    "motion_root_state": 6,
}

ACTION_JOINT_NAMES = [
    "LSP",
    "RSP",
    "WY",
    "LSR",
    "RSR",
    "WR",
    "LSY",
    "RSY",
    "WP",
    "LEP",
    "REP",
    "LHP",
    "RHP",
    "LHR",
    "RHR",
    "LHY",
    "RHY",
    "LKP",
    "RKP",
    "LAP",
    "RAP",
    "LAR",
    "RAR",
]

ACTION_COLUMN_BY_JOINT = {
    joint_name.lower(): f"raw_action_{index}" for index, joint_name in enumerate(ACTION_JOINT_NAMES)
}


def parse_index_list(value: str) -> list[int]:
    if not value:
        return []
    result: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        result.append(int(item))
    return result


def pick_columns(header: list[str], args: argparse.Namespace) -> list[str]:
    if args.columns:
        columns = [column.strip() for column in args.columns.split(",") if column.strip()]
    else:
        columns = []
        if args.raw_actions:
            raw_action_columns = select_raw_action_columns(header, args.joints)
            columns.extend(sorted(raw_action_columns, key=raw_action_index))
        columns.extend(f"obs_obs_{index}" for index in parse_index_list(args.obs))

    missing = [column for column in columns if column not in header]
    if missing:
        raise SystemExit(f"Missing CSV columns: {', '.join(missing)}")
    if not columns:
        raise SystemExit("No columns selected. Use --raw-actions, --obs, or --columns.")
    return columns


def raw_action_index(column: str) -> int:
    try:
        return int(column.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def select_raw_action_columns(header: list[str], joints: str) -> list[str]:
    if not joints:
        return [column for column in header if column.startswith("raw_action_")]

    selected = []
    unknown = []
    for joint_name in [item.strip() for item in joints.split(",") if item.strip()]:
        column = ACTION_COLUMN_BY_JOINT.get(joint_name.lower())
        if column is None:
            unknown.append(joint_name)
        else:
            selected.append(column)
    if unknown:
        valid = ", ".join(ACTION_JOINT_NAMES)
        raise SystemExit(f"Unknown action joint(s): {', '.join(unknown)}. Valid joints: {valid}")
    return selected


def action_label(column: str) -> str:
    index = raw_action_index(column)
    if 0 <= index < len(ACTION_JOINT_NAMES):
        return f"{ACTION_JOINT_NAMES[index]} ({column})"
    return column


def load_obs_terms(obs_yaml_path: Path, group_name: str) -> list[dict[str, int | str]]:
    try:
        import yaml
    except ImportError as error:
        raise SystemExit("PyYAML is required for --obs-terms: python3 -m pip install PyYAML") from error

    with obs_yaml_path.open() as file:
        root = yaml.safe_load(file)

    obs_root = root.get("igris_c_gmt_public_observation", {}) if root else {}
    for group in obs_root.get("groups", []):
        if group.get("name") != group_name:
            continue

        terms: list[dict[str, int | str]] = []
        offset = 0
        for term in group.get("terms", []):
            function = term.get("function")
            history = int(term.get("history", 1))
            if function not in OBS_TERM_SIZES:
                raise SystemExit(f"Unknown observation term size: {function}")
            size = OBS_TERM_SIZES[function]
            latest_offset = offset + (history - 1) * size
            terms.append(
                {
                    "group": group_name,
                    "function": function,
                    "history": history,
                    "size": size,
                    "latest_offset": latest_offset,
                }
            )
            offset += history * size
        return terms

    raise SystemExit(f"Observation group not found in {obs_yaml_path}: {group_name}")


def term_columns(term: dict[str, int | str]) -> list[str]:
    group = str(term["group"])
    latest_offset = int(term["latest_offset"])
    size = int(term["size"])
    return [f"obs_{group}_{index}" for index in range(latest_offset, latest_offset + size)]


def should_use_action_subplots(columns: list[str], args: argparse.Namespace) -> bool:
    return (
        args.action_subplots
        and not args.columns
        and not args.obs
        and len(columns) > 1
        and all(column.startswith("raw_action_") for column in columns)
    )


def read_selected(csv_path: Path, columns: list[str], max_points: int) -> tuple[list[float], dict[str, list[float]]]:
    with csv_path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise SystemExit(f"CSV has no header: {csv_path}")

        rows = list(reader)

    if max_points > 0 and len(rows) > max_points:
        stride = max(1, len(rows) // max_points)
        rows = rows[::stride]

    x_values: list[float] = []
    y_values = {column: [] for column in columns}
    first_timestamp_ns: int | None = None

    for row_index, row in enumerate(rows):
        timestamp_text = row.get("timestamp_ns", "")
        if timestamp_text:
            timestamp_ns = int(timestamp_text)
            if first_timestamp_ns is None:
                first_timestamp_ns = timestamp_ns
            x_values.append((timestamp_ns - first_timestamp_ns) / 1e9)
        else:
            x_values.append(float(row_index))

        for column in columns:
            value = row.get(column, "")
            y_values[column].append(float(value) if value else float("nan"))

    return x_values, y_values


def plot_overlay(plt, csv_path: Path, header: list[str], x_values: list[float], y_values: dict[str, list[float]]) -> None:
    x_label = "time_s" if "timestamp_ns" in header else "row"
    _, axis = plt.subplots(figsize=(12, 7))
    for column, values in y_values.items():
        axis.plot(x_values, values, label=column, linewidth=1.0)

    axis.set_title(csv_path.name)
    axis.set_xlabel(x_label)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", fontsize="small", ncols=2)
    plt.tight_layout()


def plot_action_subplots(
    plt,
    csv_path: Path,
    header: list[str],
    x_values: list[float],
    y_values: dict[str, list[float]],
    columns: list[str],
) -> None:
    x_label = "time_s" if "timestamp_ns" in header else "row"
    columns_per_row = 4
    rows = math.ceil(len(columns) / columns_per_row)
    figure, axes = plt.subplots(rows, columns_per_row, figsize=(16, 2.2 * rows), sharex=True)
    flat_axes = list(axes.flat)
    for axis, column in zip(flat_axes, columns):
        axis.plot(x_values, y_values[column], linewidth=0.9)
        axis.set_title(action_label(column), fontsize="small")
        axis.grid(True, alpha=0.3)
    for axis in flat_axes[len(columns) :]:
        axis.set_visible(False)
    for axis in flat_axes[-columns_per_row:]:
        if axis.get_visible():
            axis.set_xlabel(x_label)
    figure.suptitle(csv_path.name)
    plt.tight_layout()


def save_or_show(plt, output: Path | None) -> None:
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_obs_terms(plt, csv_path: Path, header: list[str], args: argparse.Namespace) -> int:
    obs_yaml_path = args.obs_yaml.expanduser().resolve()
    terms = load_obs_terms(obs_yaml_path, args.group)
    missing_terms: list[str] = []

    for term in terms:
        columns = term_columns(term)
        existing_columns = [column for column in columns if column in header]
        if len(existing_columns) != len(columns):
            missing_terms.append(str(term["function"]))
            continue

        x_values, y_values = read_selected(csv_path, existing_columns, args.max_points)
        x_label = "time_s" if "timestamp_ns" in header else "row"
        columns_per_row = 4
        rows = math.ceil(len(existing_columns) / columns_per_row)
        figure, axes = plt.subplots(
            rows,
            columns_per_row,
            figsize=(16, max(2.2 * rows, 4.0)),
            sharex=True,
        )
        flat_axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

        for axis, column in zip(flat_axes, existing_columns):
            local_index = int(column.rsplit("_", 1)[1]) - int(term["latest_offset"])
            axis.plot(x_values, y_values[column], linewidth=0.8)
            axis.set_title(f"{term['function']}[{local_index}]", fontsize="small")
            axis.grid(True, alpha=0.3)

        for axis in flat_axes[len(existing_columns) :]:
            axis.set_visible(False)
        for axis in flat_axes[-columns_per_row:]:
            if axis.get_visible():
                axis.set_xlabel(x_label)

        figure.suptitle(
            f"{args.group}.{term['function']} latest history "
            f"(size={term['size']}, history={term['history']})"
        )
        plt.tight_layout()

        output = None
        if args.output:
            output_base = args.output.expanduser()
            if output_base.suffix:
                output = output_base.with_name(
                    f"{output_base.stem}_{args.group}_{term['function']}{output_base.suffix}"
                )
            else:
                output = output_base / f"{args.group}_{term['function']}.png"
        save_or_show(plt, output)

    if missing_terms:
        raise SystemExit(f"Missing CSV columns for terms: {', '.join(missing_terms)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", type=Path, default=default_csv_path())
    parser.add_argument("--columns", help="Comma-separated exact CSV column names.")
    parser.add_argument("--obs", default="", help="Comma-separated observation indices, e.g. 0,1,2.")
    parser.add_argument("--obs-terms", action="store_true", help="Plot each obs.yaml term to its own image; term dimensions are separate subplots using only the latest history segment.")
    parser.add_argument("--obs-yaml", type=Path, default=default_obs_yaml_path())
    parser.add_argument("--group", default="obs", help="Observation group name from obs.yaml.")
    parser.add_argument("--joints", default="", help="Comma-separated raw action joint names, e.g. LSP,RSP,WY.")
    parser.add_argument(
        "--raw-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot all raw_action_* columns.",
    )
    parser.add_argument("--max-points", type=int, default=5000)
    parser.add_argument("--output", type=Path, help="Write plot image instead of opening a window.")
    parser.add_argument(
        "--action-subplots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot raw_action_* columns as separate subplots in one image.",
    )
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit("matplotlib is required: python3 -m pip install matplotlib") from error

    csv_path = args.csv_path.expanduser().resolve()
    with csv_path.open(newline="") as file:
        reader = csv.reader(file)
        header = next(reader, None)
    if header is None:
        raise SystemExit(f"CSV has no header: {csv_path}")

    if args.obs_terms:
        return plot_obs_terms(plt, csv_path, header, args)

    columns = pick_columns(header, args)
    x_values, y_values = read_selected(csv_path, columns, args.max_points)

    if should_use_action_subplots(columns, args):
        plot_action_subplots(plt, csv_path, header, x_values, y_values, columns)
    else:
        plot_overlay(plt, csv_path, header, x_values, y_values)

    save_or_show(plt, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
