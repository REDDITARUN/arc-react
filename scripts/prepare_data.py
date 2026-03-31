#!/usr/bin/env python3
"""
Phase 1 ARC-AGI-1 data preparation.

This script does only the minimum needed for the first modeling phase:
1. clone or reuse ARC-AGI
2. load training and evaluation tasks
3. save canonical unpadded json files
4. pad every grid to 30x30 and create masks
5. save padded json files
6. print a few small analysis tables

Tiny note on examples:
- each ARC task has train examples and test/query examples
- train examples have input/output pairs
- test/query examples are the inputs the model should solve
- we keep both the raw shapes and the padded 30x30 version
"""

import argparse
import json
import os
import shutil
import subprocess

from collections import Counter
from pathlib import Path


# ============================================================
# Config
# ============================================================

CONFIG = {
    "arc_repo_url": "https://github.com/fchollet/ARC-AGI",
    "arc_repo_dir_name": "ARC-AGI",
    "output_dir_name": "data",
    "canonical_train_dir_name": "canonical_train",
    "canonical_evaluation_dir_name": "canonical_evaluation",
    "padded_dir_name": "padded_processed",
    "reports_dir_name": "reports",
    "manifests_dir_name": "manifests",
    "pad_size": 30,
    "cleanup_clone": False,
    "sample_preview_tasks": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ARC-AGI-1 phase one data.")
    parser.add_argument(
        "--workspace",
        default=".",
        help="Workspace folder that contains the repo and output data.",
    )
    parser.add_argument(
        "--cleanup-clone",
        action="store_true",
        help="Delete the ARC-AGI clone after outputs are safely written.",
    )
    return parser.parse_args()


def make_runtime_config(args: argparse.Namespace) -> dict:
    config = dict(CONFIG)
    config["cleanup_clone"] = args.cleanup_clone
    return config


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: object) -> None:
    ensure_directory(path.parent)
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)


def save_text(path: Path, text: str) -> None:
    ensure_directory(path.parent)
    with open(path, "w") as handle:
        handle.write(text)


def run_command(command: list, cwd: Path = None) -> None:
    print("Running command:")
    print("  " + " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def build_table(headers: list, rows: list, title: str = "") -> str:
    widths = []
    header_index = 0
    while header_index < len(headers):
        widths.append(len(str(headers[header_index])))
        header_index += 1

    row_index = 0
    while row_index < len(rows):
        row = rows[row_index]
        col_index = 0
        while col_index < len(row):
            cell_text = str(row[col_index])
            if len(cell_text) > widths[col_index]:
                widths[col_index] = len(cell_text)
            col_index += 1
        row_index += 1

    border_parts = []
    width_index = 0
    while width_index < len(widths):
        border_parts.append("-" * (widths[width_index] + 2))
        width_index += 1
    border = "+" + "+".join(border_parts) + "+"

    lines = []
    if title:
        lines.append(title)
    lines.append(border)

    header_cells = []
    header_index = 0
    while header_index < len(headers):
        text = str(headers[header_index]).ljust(widths[header_index])
        header_cells.append(" " + text + " ")
        header_index += 1
    lines.append("|" + "|".join(header_cells) + "|")
    lines.append(border)

    row_index = 0
    while row_index < len(rows):
        row = rows[row_index]
        row_cells = []
        col_index = 0
        while col_index < len(row):
            text = str(row[col_index]).ljust(widths[col_index])
            row_cells.append(" " + text + " ")
            col_index += 1
        lines.append("|" + "|".join(row_cells) + "|")
        row_index += 1

    lines.append(border)
    return "\n".join(lines)


def print_section(title: str) -> None:
    print("")
    print(title)
    print("=" * len(title))


def print_config(config: dict) -> None:
    keys = []
    for key in config:
        keys.append(key)
    keys.sort()

    rows = []
    index = 0
    while index < len(keys):
        key = keys[index]
        rows.append([key, config[key]])
        index += 1

    print(build_table(["key", "value"], rows, "Pipeline Config"))


def build_paths(workspace: Path, config: dict) -> dict:
    paths = {}
    paths["workspace"] = workspace
    paths["arc_repo"] = workspace / config["arc_repo_dir_name"]
    paths["output_root"] = workspace / config["output_dir_name"]
    paths["canonical_train_root"] = paths["output_root"] / config["canonical_train_dir_name"]
    paths["canonical_evaluation_root"] = paths["output_root"] / config["canonical_evaluation_dir_name"]
    paths["padded_root"] = paths["output_root"] / config["padded_dir_name"]
    paths["reports_root"] = paths["output_root"] / config["reports_dir_name"]
    paths["manifests_root"] = paths["output_root"] / config["manifests_dir_name"]
    paths["canonical_train_tasks"] = paths["canonical_train_root"] / "tasks"
    paths["canonical_evaluation_tasks"] = paths["canonical_evaluation_root"] / "tasks"
    paths["padded_train_tasks"] = paths["padded_root"] / "train"
    paths["padded_evaluation_tasks"] = paths["padded_root"] / "evaluation"
    return paths


def ensure_output_layout(paths: dict) -> None:
    keys = [
        "output_root",
        "canonical_train_root",
        "canonical_evaluation_root",
        "padded_root",
        "reports_root",
        "manifests_root",
        "canonical_train_tasks",
        "canonical_evaluation_tasks",
        "padded_train_tasks",
        "padded_evaluation_tasks",
    ]

    index = 0
    while index < len(keys):
        ensure_directory(paths[keys[index]])
        index += 1


def clone_or_reuse_repo(workspace: Path, repo_url: str, repo_dir_name: str) -> Path:
    repo_path = workspace / repo_dir_name
    if repo_path.exists():
        print(f"Reusing existing repo: {repo_path}")
        return repo_path

    print(f"Cloning repo: {repo_url}")
    run_command(["git", "clone", repo_url, repo_dir_name], cwd=workspace)
    return repo_path


def load_json(path: Path) -> object:
    with open(path, "r") as handle:
        return json.load(handle)


def load_split_tasks(arc_repo: Path, split_name: str) -> dict:
    split_dir = arc_repo / "data" / split_name
    task_files = []
    tasks = {}

    for entry in os.listdir(split_dir):
        if entry.endswith(".json"):
            task_files.append(entry)

    task_files.sort()

    file_index = 0
    while file_index < len(task_files):
        file_name = task_files[file_index]
        task_id = file_name.replace(".json", "")
        tasks[task_id] = load_json(split_dir / file_name)
        file_index += 1

    return tasks


def task_record(task_id: str, task_data: dict, source: str) -> dict:
    record = {}
    record["task_id"] = task_id
    record["source"] = source
    record["train"] = task_data["train"]
    record["test"] = task_data["test"]
    return record


def save_canonical_split(tasks: dict, output_dir: Path, source: str) -> None:
    task_ids = []
    for task_id in tasks:
        task_ids.append(task_id)
    task_ids.sort()

    index = 0
    while index < len(task_ids):
        task_id = task_ids[index]
        record = task_record(task_id, tasks[task_id], source)
        save_json(output_dir / f"{task_id}.json", record)
        index += 1


def grid_shape(grid: list) -> list:
    return [len(grid), len(grid[0])]


def grid_area(grid: list) -> int:
    return len(grid) * len(grid[0])


def size_relation(pair: dict) -> str:
    if grid_shape(pair["input"]) == grid_shape(pair["output"]):
        return "same"
    return "changed"


def count_colors(grid: list) -> int:
    colors = set()
    row_index = 0
    while row_index < len(grid):
        row = grid[row_index]
        col_index = 0
        while col_index < len(row):
            colors.add(row[col_index])
            col_index += 1
        row_index += 1
    return len(colors)


def pad_grid(grid: list, pad_size: int) -> tuple:
    padded = []
    mask = []

    row_index = 0
    while row_index < pad_size:
        padded_row = []
        mask_row = []
        col_index = 0
        while col_index < pad_size:
            if row_index < len(grid) and col_index < len(grid[0]):
                padded_row.append(grid[row_index][col_index])
                mask_row.append(1)
            else:
                padded_row.append(0)
                mask_row.append(0)
            col_index += 1
        padded.append(padded_row)
        mask.append(mask_row)
        row_index += 1

    return padded, mask


def zero_grid(size: int) -> list:
    grid = []
    row_index = 0
    while row_index < size:
        row = []
        col_index = 0
        while col_index < size:
            row.append(0)
            col_index += 1
        grid.append(row)
        row_index += 1
    return grid


def padded_pair(pair: dict, pair_index: int, pad_size: int) -> dict:
    input_padded, input_mask = pad_grid(pair["input"], pad_size)

    if "output" in pair:
        output_padded, output_mask = pad_grid(pair["output"], pad_size)
        output_shape = grid_shape(pair["output"])
    else:
        output_padded = zero_grid(pad_size)
        output_mask = zero_grid(pad_size)
        output_shape = None

    record = {}
    record["pair_index"] = pair_index
    record["input_shape"] = grid_shape(pair["input"])
    record["output_shape"] = output_shape
    record["input"] = input_padded
    record["output"] = output_padded
    record["input_mask"] = input_mask
    record["output_mask"] = output_mask
    return record


def padded_task_record(task_id: str, task_data: dict, source: str, pad_size: int) -> dict:
    record = {}
    record["task_id"] = task_id
    record["source"] = source
    record["train"] = []
    record["test"] = []

    split_names = ["train", "test"]
    split_index = 0
    while split_index < len(split_names):
        split_name = split_names[split_index]
        pairs = task_data[split_name]
        pair_index = 0
        while pair_index < len(pairs):
            record[split_name].append(padded_pair(pairs[pair_index], pair_index, pad_size))
            pair_index += 1
        split_index += 1

    return record


def save_padded_split(tasks: dict, output_dir: Path, source: str, pad_size: int) -> None:
    task_ids = []
    for task_id in tasks:
        task_ids.append(task_id)
    task_ids.sort()

    index = 0
    while index < len(task_ids):
        task_id = task_ids[index]
        record = padded_task_record(task_id, tasks[task_id], source, pad_size)
        save_json(output_dir / f"{task_id}.json", record)
        index += 1


def limited_counter_rows(counter: Counter, limit: int = 12) -> list:
    keys = []
    for key in counter:
        keys.append(key)
    keys.sort(key=lambda value: str(value))

    rows = []
    index = 0
    while index < len(keys) and index < limit:
        key = keys[index]
        rows.append([key, counter[key]])
        index += 1
    return rows


def analyze_training_tasks(tasks: dict) -> tuple:
    examples_per_task = Counter()
    input_size_dist = Counter()
    output_size_dist = Counter()
    size_relation_dist = Counter()
    input_color_dist = Counter()

    task_ids = []
    for task_id in tasks:
        task_ids.append(task_id)
    task_ids.sort()

    task_index = 0
    while task_index < len(task_ids):
        task = tasks[task_ids[task_index]]
        examples_per_task[len(task["train"])] += 1

        pair_index = 0
        while pair_index < len(task["train"]):
            pair = task["train"][pair_index]
            input_shape = grid_shape(pair["input"])
            output_shape = grid_shape(pair["output"])
            input_size_dist[f"{input_shape[0]}x{input_shape[1]}"] += 1
            output_size_dist[f"{output_shape[0]}x{output_shape[1]}"] += 1
            size_relation_dist[size_relation(pair)] += 1
            input_color_dist[count_colors(pair["input"])] += 1
            pair_index += 1

        task_index += 1

    summary_rows = []
    summary_rows.append(["num training tasks", len(tasks)])
    summary_rows.append(["distinct train example counts", len(examples_per_task)])
    summary_rows.append(["distinct input sizes", len(input_size_dist)])
    summary_rows.append(["distinct output sizes", len(output_size_dist)])
    summary_rows.append(["distinct input color counts", len(input_color_dist)])

    sections = []
    sections.append(build_table(["metric", "value"], summary_rows, "Original Analysis Summary"))
    sections.append(build_table(["value", "count"], limited_counter_rows(examples_per_task), "Train Examples Per Task"))
    sections.append(build_table(["value", "count"], limited_counter_rows(size_relation_dist), "Input/Output Same Vs Changed"))
    sections.append(build_table(["value", "count"], limited_counter_rows(input_size_dist), "Top Input Grid Sizes"))
    sections.append(build_table(["value", "count"], limited_counter_rows(output_size_dist), "Top Output Grid Sizes"))
    sections.append(build_table(["value", "count"], limited_counter_rows(input_color_dist), "Input Color Count Distribution"))

    text = "\n\n".join(sections)

    summary = {
        "examples_per_task": dict(examples_per_task),
        "input_size_dist": dict(input_size_dist),
        "output_size_dist": dict(output_size_dist),
        "size_relation_dist": dict(size_relation_dist),
        "input_color_dist": dict(input_color_dist),
    }
    return summary, text


def analyze_padding(train_tasks: dict, evaluation_tasks: dict, pad_size: int) -> tuple:
    total_real_cells = 0
    total_padded_cells = 0

    all_splits = [
        ("train", train_tasks),
        ("evaluation", evaluation_tasks),
    ]

    split_group_index = 0
    while split_group_index < len(all_splits):
        split_tasks = all_splits[split_group_index][1]

        task_ids = []
        for task_id in split_tasks:
            task_ids.append(task_id)
        task_ids.sort()

        task_index = 0
        while task_index < len(task_ids):
            task = split_tasks[task_ids[task_index]]

            split_names = ["train", "test"]
            split_index = 0
            while split_index < len(split_names):
                split_name = split_names[split_index]
                pairs = task[split_name]
                pair_index = 0
                while pair_index < len(pairs):
                    pair = pairs[pair_index]
                    input_area = grid_area(pair["input"])
                    total_real_cells += input_area
                    total_padded_cells += (pad_size * pad_size) - input_area

                    if "output" in pair:
                        output_area = grid_area(pair["output"])
                        total_real_cells += output_area
                        total_padded_cells += (pad_size * pad_size) - output_area

                    pair_index += 1
                split_index += 1

            task_index += 1

        split_group_index += 1

    total_cells = total_real_cells + total_padded_cells
    padding_ratio = 0.0
    if total_cells > 0:
        padding_ratio = total_padded_cells / total_cells

    sections = []
    sections.append(
        build_table(
            ["metric", "value"],
            [
                ["total train tasks", len(train_tasks)],
                ["total evaluation tasks", len(evaluation_tasks)],
                ["real cells", total_real_cells],
                ["padded cells", total_padded_cells],
                ["padding ratio", f"{padding_ratio:.4f}"],
            ],
            "Padded Analysis Summary",
        )
    )

    text = "\n\n".join(sections)

    summary = {
        "total_train_tasks": len(train_tasks),
        "total_evaluation_tasks": len(evaluation_tasks),
        "real_cells": total_real_cells,
        "padded_cells": total_padded_cells,
        "padding_ratio": padding_ratio,
    }
    return summary, text


def preview_examples(train_tasks: dict, evaluation_tasks: dict, pad_size: int, sample_count: int) -> str:
    train_ids = []
    for task_id in train_tasks:
        train_ids.append(task_id)
    train_ids.sort()

    eval_ids = []
    for task_id in evaluation_tasks:
        eval_ids.append(task_id)
    eval_ids.sort()

    rows = []

    index = 0
    while index < sample_count and index < len(train_ids):
        task = train_tasks[train_ids[index]]
        raw_pair = task["train"][0]
        padded = padded_pair(raw_pair, 0, pad_size)
        rows.append([
            "train_raw",
            train_ids[index],
            f"{len(raw_pair['input'])}x{len(raw_pair['input'][0])}",
            f"{len(raw_pair['output'])}x{len(raw_pair['output'][0])}",
            "-",
        ])
        rows.append([
            "train_padded",
            train_ids[index],
            f"{padded['input_shape'][0]}x{padded['input_shape'][1]} -> {pad_size}x{pad_size}",
            f"{padded['output_shape'][0]}x{padded['output_shape'][1]} -> {pad_size}x{pad_size}",
            f"in_mask={sum(sum(row) for row in padded['input_mask'])}, out_mask={sum(sum(row) for row in padded['output_mask'])}",
        ])
        index += 1

    index = 0
    while index < sample_count and index < len(eval_ids):
        task = evaluation_tasks[eval_ids[index]]
        raw_pair = task["test"][0]
        padded = padded_pair(raw_pair, 0, pad_size)
        output_shape = "unknown"
        if "output" in raw_pair:
            output_shape = f"{len(raw_pair['output'])}x{len(raw_pair['output'][0])}"
        padded_output = "unknown -> 30x30_zero_mask"
        if padded["output_shape"] is not None:
            padded_output = f"{padded['output_shape'][0]}x{padded['output_shape'][1]} -> {pad_size}x{pad_size}"
        rows.append([
            "eval_raw",
            eval_ids[index],
            f"{len(raw_pair['input'])}x{len(raw_pair['input'][0])}",
            output_shape,
            "-",
        ])
        rows.append([
            "eval_padded",
            eval_ids[index],
            f"{padded['input_shape'][0]}x{padded['input_shape'][1]} -> {pad_size}x{pad_size}",
            padded_output,
            f"in_mask={sum(sum(row) for row in padded['input_mask'])}, out_mask={sum(sum(row) for row in padded['output_mask'])}",
        ])
        index += 1

    return build_table(
        ["kind", "task_id", "input", "output", "note"],
        rows,
        "Before And After Padded Examples",
    )


def git_commit_hash(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def build_manifest(config: dict, paths: dict) -> dict:
    manifest = {}
    manifest["config"] = config
    manifest["arc_repo_path"] = str(paths["arc_repo"])
    manifest["arc_repo_commit"] = git_commit_hash(paths["arc_repo"])
    return manifest


def safe_to_cleanup(paths: dict) -> bool:
    required_files = [
        paths["manifests_root"] / "run_manifest.json",
        paths["manifests_root"] / "original_analysis_summary.json",
        paths["manifests_root"] / "padded_analysis_summary.json",
        paths["reports_root"] / "original_analysis.txt",
        paths["reports_root"] / "padded_analysis.txt",
    ]

    index = 0
    while index < len(required_files):
        file_path = required_files[index]
        if not file_path.exists():
            return False
        if file_path.stat().st_size == 0:
            return False
        index += 1

    return True


def cleanup_clone(paths: dict) -> None:
    if not safe_to_cleanup(paths):
        raise RuntimeError("Cleanup requested but required outputs are missing.")

    if paths["arc_repo"].exists():
        print(f"Deleting repo clone: {paths['arc_repo']}")
        shutil.rmtree(paths["arc_repo"])


def main() -> None:
    args = parse_args()
    config = make_runtime_config(args)
    workspace = Path(args.workspace).resolve()
    paths = build_paths(workspace, config)

    print_config(config)
    ensure_output_layout(paths)

    paths["arc_repo"] = clone_or_reuse_repo(workspace, config["arc_repo_url"], config["arc_repo_dir_name"])

    manifest = build_manifest(config, paths)
    save_json(paths["manifests_root"] / "run_manifest.json", manifest)

    print_section("Loading ARC-AGI")
    train_tasks = load_split_tasks(paths["arc_repo"], "training")
    evaluation_tasks = load_split_tasks(paths["arc_repo"], "evaluation")

    print_section("Saving Canonical Data")
    save_canonical_split(train_tasks, paths["canonical_train_tasks"], "original_train")
    save_canonical_split(evaluation_tasks, paths["canonical_evaluation_tasks"], "original_evaluation")

    print_section("Saving Padded Data")
    save_padded_split(train_tasks, paths["padded_train_tasks"], "padded_train", config["pad_size"])
    save_padded_split(evaluation_tasks, paths["padded_evaluation_tasks"], "padded_evaluation", config["pad_size"])

    print_section("Original Analysis")
    original_summary, original_text = analyze_training_tasks(train_tasks)
    save_json(paths["manifests_root"] / "original_analysis_summary.json", original_summary)
    save_text(paths["reports_root"] / "original_analysis.txt", original_text)
    print(original_text)

    print_section("Padded Analysis")
    padded_summary, padded_text = analyze_padding(train_tasks, evaluation_tasks, config["pad_size"])
    save_json(paths["manifests_root"] / "padded_analysis_summary.json", padded_summary)
    save_text(paths["reports_root"] / "padded_analysis.txt", padded_text)
    print(padded_text)

    print_section("Sample Preview")
    preview_text = preview_examples(
        train_tasks,
        evaluation_tasks,
        config["pad_size"],
        config["sample_preview_tasks"],
    )
    save_text(paths["reports_root"] / "sample_preview.txt", preview_text)
    print(preview_text)

    if config["cleanup_clone"]:
        cleanup_clone(paths)

    print_section("Done")
    print(f"Outputs written to: {paths['output_root']}")


if __name__ == "__main__":
    main()
