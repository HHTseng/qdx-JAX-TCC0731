"""Shared helpers for profiling training and validation flows."""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax


def block_until_ready_tree(value):
    """Synchronize every JAX leaf in a pytree."""

    def _block(item):
        if hasattr(item, "block_until_ready"):
            item.block_until_ready()
        return item

    return jax.tree_util.tree_map(_block, value)


def timed_section(recorder, label):
    """Return a timing context manager or a no-op if profiling is disabled."""

    if recorder is None:
        return nullcontext()
    return recorder.section(label)


@dataclass
class TimingRecorder:
    """Collect wall-clock timing data grouped by labeled sections."""

    entries: list[dict[str, Any]] = field(default_factory=list)

    def record(self, label, seconds):
        self.entries.append({"label": label, "seconds": float(seconds)})

    @contextmanager
    def section(self, label):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.record(label, time.perf_counter() - started)

    def summary(self):
        totals = OrderedDict()
        counts = OrderedDict()
        for entry in self.entries:
            label = entry["label"]
            if label not in totals:
                totals[label] = 0.0
                counts[label] = 0
            totals[label] += float(entry["seconds"])
            counts[label] += 1
        return [
            {"label": label, "seconds": totals[label], "count": counts[label]}
            for label in totals
        ]

    def phase_summary(self):
        phases = OrderedDict()
        for item in self.summary():
            phase, _, detail = item["label"].partition("/")
            phase_entry = phases.setdefault(
                phase,
                {
                    "phase": phase,
                    "seconds": 0.0,
                    "count": 0,
                    "items": [],
                },
            )
            phase_entry["seconds"] += item["seconds"]
            phase_entry["count"] += item["count"]
            phase_entry["items"].append(
                {
                    "label": detail or phase,
                    "seconds": item["seconds"],
                    "count": item["count"],
                }
            )
        return list(phases.values())

    def as_dict(self):
        phases = self.phase_summary()
        return {
            "total_seconds": sum(phase["seconds"] for phase in phases),
            "entries": self.entries,
            "summary": self.summary(),
            "phases": phases,
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(self.as_dict(), file, indent=2)

    def format_report(self, title="Timing breakdown"):
        phases = self.phase_summary()
        lines = [title]
        if not phases:
            lines.append("  (no timing data)")
            return "\n".join(lines)

        total_seconds = sum(phase["seconds"] for phase in phases)
        lines.append(f"  total: {total_seconds:.3f}s")
        for phase in phases:
            lines.append(f"{phase['phase']}: {phase['seconds']:.3f}s")
            for item in sorted(
                phase["items"], key=lambda entry: entry["seconds"], reverse=True
            ):
                count_suffix = f" x{item['count']}" if item["count"] > 1 else ""
                lines.append(
                    f"  {item['label']}: {item['seconds']:.3f}s{count_suffix}"
                )
        return "\n".join(lines)
