from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


DEFAULT_RESULT_DIR = (
    Path(__file__).resolve().parent
    / "results"
    / "demo_multitask_joint_nk_20260630002415"
)

OUTPUT_FORMAT = "png"

METRICS = {
    "reward_mean": "Reward Mean",
    "done_rate": "Done Rate",
    "success_rate": "Success Rate",
    "episode_return_mean": "Episode Return Mean",
    "episode_length_mean": "Episode Length Mean",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot training metrics by update from a result directory."
    )
    parser.add_argument(
        "result_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="Path to a result directory containing train_history.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save figures. Defaults to <result_dir>/plots",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after saving them.",
    )
    return parser.parse_args()


def load_history(history_path: Path) -> list[dict]:
    with history_path.open("r", encoding="utf-8") as f:
        history = json.load(f)

    if not isinstance(history, list) or not history:
        raise ValueError(f"{history_path} does not contain a non-empty history list.")

    return sorted(history, key=lambda item: item["update"])


def plot_metric(
    updates: list[int],
    values: list[float],
    metric_name: str,
    label: str,
    output_dir: Path,
) -> Path:
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(updates, values, linewidth=2, color="#1f77b4")
    axis.set_title(f"{label} by Update")
    axis.set_xlabel("Update")
    axis.set_ylabel(label)
    axis.grid(True, linestyle="--", alpha=0.35)
    figure.tight_layout()

    output_path = output_dir / f"{metric_name}_by_update.{OUTPUT_FORMAT}"
    figure.savefig(output_path, dpi=200)
    return output_path


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    history_path = result_dir / "train_history.json"
    output_dir = (args.output_dir or result_dir / "plots").resolve()

    if not history_path.exists():
        raise FileNotFoundError(f"Could not find {history_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    history = load_history(history_path)
    updates = [entry["update"] for entry in history]

    saved_paths: list[Path] = []
    for metric_name, label in METRICS.items():
        values = [entry[metric_name] for entry in history]
        saved_paths.append(
            plot_metric(
                updates=updates,
                values=values,
                metric_name=metric_name,
                label=label,
                output_dir=output_dir,
            )
        )

    for figure_id in plt.get_fignums():
        plt.figure(figure_id)
        if args.show:
            plt.show()
        plt.close()

    print("Saved plots:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
