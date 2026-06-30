from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape


DEFAULT_RESULT_DIR = (
    Path(__file__).resolve().parent
    / "results"
    / "demo_multitask_joint_nk_20260630002415"
)

METRICS = {
    "reward_mean": "Reward Mean",
    "done_rate": "Done Rate",
    "episode_return_mean": "Episode Return Mean",
    "episode_length_mean": "Episode Length Mean",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot training metrics by update into SVG files."
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
        help="Directory to save SVG figures. Defaults to <result_dir>/plots",
    )
    return parser.parse_args()


def load_history(history_path: Path) -> list[dict]:
    with history_path.open("r", encoding="utf-8") as f:
        history = json.load(f)

    if not isinstance(history, list) or not history:
        raise ValueError(f"{history_path} does not contain a non-empty history list.")

    return sorted(history, key=lambda item: item["update"])


def format_tick(value: float) -> str:
    if abs(value) >= 100 or value.is_integer():
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def build_ticks(min_value: float, max_value: float, count: int = 5) -> list[float]:
    if min_value == max_value:
        return [min_value]
    return [
        min_value + (max_value - min_value) * step / (count - 1)
        for step in range(count)
    ]


def render_svg(
    updates: list[int],
    values: list[float],
    title: str,
    output_path: Path,
) -> None:
    width = 960
    height = 540
    margin_left = 96
    margin_right = 32
    margin_top = 64
    margin_bottom = 72
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    min_x = min(updates)
    max_x = max(updates)
    min_y = min(values)
    max_y = max(values)

    if min_x == max_x:
        min_x -= 1
        max_x += 1

    if min_y == max_y:
        padding = abs(min_y) * 0.05 or 1.0
        min_y -= padding
        max_y += padding
    else:
        padding = (max_y - min_y) * 0.05
        min_y -= padding
        max_y += padding

    def x_to_svg(value: float) -> float:
        return margin_left + (value - min_x) / (max_x - min_x) * plot_width

    def y_to_svg(value: float) -> float:
        return margin_top + (max_y - value) / (max_y - min_y) * plot_height

    x_ticks = build_ticks(float(min(updates)), float(max(updates)))
    y_ticks = build_ticks(min_y, max_y)

    polyline_points = " ".join(
        f"{x_to_svg(float(x)):.2f},{y_to_svg(y):.2f}"
        for x, y in zip(updates, values)
    )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        f'<text x="{width / 2:.1f}" y="36" text-anchor="middle" font-size="24" font-family="Arial, sans-serif" fill="#111827">{escape(title)} by Update</text>',
    ]

    for tick in y_ticks:
        y = y_to_svg(tick)
        parts.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{margin_left - 12}" y="{y + 5:.2f}" text-anchor="end" font-size="12" font-family="Arial, sans-serif" fill="#4b5563">{escape(format_tick(tick))}</text>'
        )

    for tick in x_ticks:
        x = x_to_svg(tick)
        parts.append(
            f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{height - margin_bottom}" stroke="#f3f4f6" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{x:.2f}" y="{height - margin_bottom + 24}" text-anchor="middle" font-size="12" font-family="Arial, sans-serif" fill="#4b5563">{escape(format_tick(tick))}</text>'
        )

    parts.extend(
        [
            f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#111827" stroke-width="1.5" />',
            f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#111827" stroke-width="1.5" />',
            f'<polyline fill="none" stroke="#2563eb" stroke-width="3" points="{polyline_points}" />',
            f'<text x="{width / 2:.1f}" y="{height - 20}" text-anchor="middle" font-size="16" font-family="Arial, sans-serif" fill="#111827">Update</text>',
            f'<text x="24" y="{height / 2:.1f}" text-anchor="middle" font-size="16" font-family="Arial, sans-serif" fill="#111827" transform="rotate(-90 24 {height / 2:.1f})">{escape(title)}</text>',
        ]
    )

    for x, y in zip(updates, values):
        parts.append(
            f'<circle cx="{x_to_svg(float(x)):.2f}" cy="{y_to_svg(y):.2f}" r="3.5" fill="#1d4ed8" />'
        )

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    history_path = result_dir / "train_history.json"
    output_dir = (args.output_dir or result_dir / "plots").resolve()

    if not history_path.exists():
        raise FileNotFoundError(f"Could not find {history_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    history = load_history(history_path)
    updates = [int(entry["update"]) for entry in history]

    saved_paths: list[Path] = []
    for metric_name, label in METRICS.items():
        values = [float(entry[metric_name]) for entry in history]
        output_path = output_dir / f"{metric_name}_by_update.svg"
        render_svg(updates=updates, values=values, title=label, output_path=output_path)
        saved_paths.append(output_path)

    print("Saved plots:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
