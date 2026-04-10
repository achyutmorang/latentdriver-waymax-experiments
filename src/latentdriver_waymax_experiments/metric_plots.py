from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .artifacts import results_root
from .config import load_config
from .wayboard.data import RunRecord, discover_runs

DEFAULT_METRICS = (
    "ar_75_95",
    "mar_75_95",
    "collision_rate",
    "offroad_rate",
    "progress_rate",
    "reward_mean",
)

METRIC_ALIASES = {
    "reward_mean": ("average", "reward/reward_mean"),
    "metric/AR[75:95]": ("average", "metric/AR[75:95]"),
    "metric/offroad_rate": ("average", "metric/offroad_rate"),
    "metric/collision_rate": ("average", "metric/collision_rate"),
    "metric/progress_rate": ("average", "metric/progress_rate"),
    "reward/reward_mean": ("average", "reward/reward_mean"),
}


@dataclass(frozen=True)
class MetricRow:
    model: str
    tier: str
    seed: int | None
    run_id: str
    run_dir: Path
    values: dict[str, float | None]


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def configured_eval_models() -> list[str]:
    cfg = load_config()
    return [name for name, spec in cfg["checkpoints"].items() if spec.get("method")]


def metric_value(record: RunRecord, metric: str) -> float | None:
    summary_value = _to_float(record.summary.get(metric))
    if summary_value is not None:
        return summary_value
    alias = METRIC_ALIASES.get(metric)
    if alias is None:
        return None
    section, key = alias
    section_payload = record.summary.get(section, {})
    return _to_float(section_payload.get(key) if isinstance(section_payload, dict) else None)


def latest_metric_rows(
    *,
    records: Iterable[RunRecord],
    tier: str,
    seed: int | None,
    models: Sequence[str],
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> list[MetricRow]:
    by_model: dict[str, MetricRow] = {}
    for record in records:
        if record.tier != tier:
            continue
        if seed is not None and record.seed != seed:
            continue
        if not record.model or record.model not in models:
            continue
        if record.model in by_model:
            continue
        by_model[record.model] = MetricRow(
            model=record.model,
            tier=record.tier or tier,
            seed=record.seed,
            run_id=record.run_id,
            run_dir=record.run_dir,
            values={metric: metric_value(record, metric) for metric in metrics},
        )
    return [by_model[model] for model in models if model in by_model]


def write_metric_csv(path: Path, rows: Sequence[MetricRow], metrics: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "tier", "seed", "run_id", "run_dir", *metrics])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model": row.model,
                    "tier": row.tier,
                    "seed": row.seed,
                    "run_id": row.run_id,
                    "run_dir": str(row.run_dir),
                    **{metric: row.values.get(metric) for metric in metrics},
                }
            )


def write_metric_json(path: Path, rows: Sequence[MetricRow], metrics: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": list(metrics),
        "rows": [
            {
                "model": row.model,
                "tier": row.tier,
                "seed": row.seed,
                "run_id": row.run_id,
                "run_dir": str(row.run_dir),
                "values": row.values,
            }
            for row in rows
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover - exercised only when optional deps are missing.
        raise RuntimeError("matplotlib and numpy are required to generate comparison plots") from exc
    return plt, np


def write_metric_plot(path: Path, rows: Sequence[MetricRow], metrics: Sequence[str]) -> None:
    if not rows:
        raise ValueError("No metric rows available to plot")
    if not metrics:
        raise ValueError("At least one metric is required to plot")
    plt, np = _require_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)

    models = [row.model for row in rows]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(max(8, 1.7 * len(models)), 2.2 * len(metrics)), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]
    x = np.arange(len(models))
    palette = ["#2f6f73", "#d88c31", "#4f6fad", "#b95652", "#6f5a8f", "#4c8f52"]
    colors = [palette[index % len(palette)] for index in range(len(models))]

    for axis, metric in zip(axes, metrics):
        values = [row.values.get(metric) for row in rows]
        numeric = [float(value) if value is not None else np.nan for value in values]
        axis.bar(x, numeric, color=colors)
        axis.set_title(metric)
        axis.set_xticks(x, models, rotation=25, ha="right")
        axis.axhline(0, color="#404040", linewidth=0.8)
        axis.grid(axis="y", alpha=0.25)
        for idx, value in enumerate(values):
            label = "n/a" if value is None else f"{value:.3g}"
            y = 0 if value is None else float(value)
            offset = 3 if y >= 0 else -10
            axis.annotate(
                label,
                (idx, y),
                textcoords="offset points",
                xytext=(0, offset),
                ha="center",
                va="bottom" if y >= 0 else "top",
                fontsize=8,
            )
        axis.margins(y=0.18)
    fig.suptitle("Waymax Model Metric Comparison", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def default_output_dir(root: Path, *, tier: str, seed: int | None) -> Path:
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    seed_part = "allseeds" if seed is None else f"seed{seed}"
    return root / "plots" / f"{tag}_{tier}_{seed_part}"


def generate_metric_comparison(
    *,
    root: Path | None,
    tier: str,
    seed: int | None,
    models: Sequence[str] | None = None,
    metrics: Sequence[str] = DEFAULT_METRICS,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    resolved_root = (root or results_root()).expanduser().resolve()
    selected_models = list(models or configured_eval_models())
    records = discover_runs(resolved_root)
    rows = latest_metric_rows(records=records, tier=tier, seed=seed, models=selected_models, metrics=metrics)
    missing_models = [model for model in selected_models if model not in {row.model for row in rows}]
    resolved_output_dir = (output_dir or default_output_dir(resolved_root, tier=tier, seed=seed)).expanduser().resolve()
    outputs = {
        "csv": str(resolved_output_dir / "model_metrics.csv"),
        "json": str(resolved_output_dir / "model_metrics.json"),
        "plot": str(resolved_output_dir / "model_metrics.png"),
    }
    payload = {
        "results_root": str(resolved_root),
        "tier": tier,
        "seed": seed,
        "models": selected_models,
        "metrics": list(metrics),
        "rows": [
            {
                "model": row.model,
                "run_id": row.run_id,
                "run_dir": str(row.run_dir),
                "values": row.values,
            }
            for row in rows
        ],
        "missing_models": missing_models,
        "outputs": outputs,
        "ready": bool(rows),
    }
    if dry_run:
        return payload
    if not rows:
        raise FileNotFoundError(f"No completed runs found for tier={tier!r}, seed={seed!r}, models={selected_models!r}")
    write_metric_csv(Path(outputs["csv"]), rows, metrics)
    write_metric_json(Path(outputs["json"]), rows, metrics)
    write_metric_plot(Path(outputs["plot"]), rows, metrics)
    return payload
