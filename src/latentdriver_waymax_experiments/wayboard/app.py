from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.document import Document
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource,
    DataTable,
    Div,
    MultiChoice,
    PreText,
    Select,
    TableColumn,
    Tabs,
    TabPanel,
)
from bokeh.plotting import figure
from bokeh.server.server import Server
from tornado.ioloop import IOLoop
from tornado.web import StaticFileHandler

from ..artifacts import results_root
from .data import RunRecord, SuiteRecord, discover_runs, discover_suites, models, tiers

ARTIFACT_ROUTE = "/artifacts"
SUMMARY_METRICS = [
    "ar_75_95",
    "mar_75_95",
    "collision_rate",
    "offroad_rate",
    "progress_rate",
]


def _tail_text(path: Path | None, max_chars: int = 4000) -> str:
    if not path or not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _overview_source(records: Iterable[RunRecord]) -> Dict[str, List[Any]]:
    rows = list(records)
    return {
        "run_id": [row.run_id for row in rows],
        "model": [row.model or "" for row in rows],
        "tier": [row.tier or "" for row in rows],
        "seed": [row.seed if row.seed is not None else "" for row in rows],
        "episodes": [row.summary.get("number_of_episodes", "") for row in rows],
        "ar_75_95": [row.summary.get("ar_75_95", "") for row in rows],
        "mar_75_95": [row.summary.get("mar_75_95", "") for row in rows],
        "collision_rate": [row.summary.get("collision_rate", "") for row in rows],
        "offroad_rate": [row.summary.get("offroad_rate", "") for row in rows],
        "progress_rate": [row.summary.get("progress_rate", "") for row in rows],
        "media_count": [len(row.media_artifacts) for row in rows],
    }


def _suite_source(suites: Iterable[SuiteRecord]) -> Dict[str, List[Any]]:
    rows = list(suites)
    return {
        "run_id": [row.run_id for row in rows],
        "tier": [row.tier or "" for row in rows],
        "seed": [row.seed if row.seed is not None else "" for row in rows],
        "models": [", ".join(row.models) for row in rows],
        "runs": [len(row.runs) for row in rows],
    }


class WaymaxBoard:
    def __init__(self, results_dir: Path | None = None, port: int = 5007):
        self._results_dir = (results_dir or results_root()).expanduser().resolve()
        self._port = port

    @property
    def results_dir(self) -> Path:
        return self._results_dir

    def run(self) -> None:
        app = Application(FunctionHandler(self._build_document))
        server = Server(
            {"/": app},
            io_loop=IOLoop.current(),
            port=self._port,
            allow_websocket_origin=["*"],
            extra_patterns=[(rf"{ARTIFACT_ROUTE}/(.*)", StaticFileHandler, {"path": str(self._results_dir)})],
        )
        server.start()
        IOLoop.current().add_callback(server.show, "/")
        IOLoop.current().start()

    def _build_document(self, doc: Document) -> None:
        doc.title = "WaymaxBoard"
        records = discover_runs(self._results_dir)
        suites = discover_suites(self._results_dir)

        model_filter = MultiChoice(title="Models", value=models(records), options=models(records))
        tier_filter = MultiChoice(title="Tiers", value=tiers(records), options=tiers(records))
        metric_select = Select(title="Metric", value="ar_75_95", options=SUMMARY_METRICS)
        run_select = Select(title="Run", value=records[0].run_id if records else "", options=[row.run_id for row in records])
        media_select = Select(title="Media", value="", options=[])

        filtered_records: List[RunRecord] = list(records)
        filtered_source = ColumnDataSource(_overview_source(filtered_records))
        suite_source = ColumnDataSource(_suite_source(suites))
        metric_source = ColumnDataSource(data={"x": [], "top": [], "color": []})
        histogram_source = ColumnDataSource(data={"left": [], "right": [], "top": []})

        overview_table = DataTable(
            source=filtered_source,
            columns=[
                TableColumn(field="run_id", title="Run ID"),
                TableColumn(field="model", title="Model"),
                TableColumn(field="tier", title="Tier"),
                TableColumn(field="seed", title="Seed"),
                TableColumn(field="episodes", title="Episodes"),
                TableColumn(field="ar_75_95", title="AR"),
                TableColumn(field="mar_75_95", title="mAR"),
                TableColumn(field="collision_rate", title="Collision"),
                TableColumn(field="offroad_rate", title="Off-road"),
                TableColumn(field="progress_rate", title="Progress"),
                TableColumn(field="media_count", title="Media"),
            ],
            sizing_mode="stretch_width",
            height=360,
            sortable=True,
            index_position=None,
        )
        suite_table = DataTable(
            source=suite_source,
            columns=[
                TableColumn(field="run_id", title="Suite ID"),
                TableColumn(field="tier", title="Tier"),
                TableColumn(field="seed", title="Seed"),
                TableColumn(field="models", title="Models"),
                TableColumn(field="runs", title="Run Count"),
            ],
            sizing_mode="stretch_width",
            height=180,
            sortable=True,
            index_position=None,
        )

        metric_plot = figure(
            title="Run-level metric values",
            x_range=[],
            height=340,
            sizing_mode="stretch_width",
            toolbar_location=None,
        )
        metric_plot.vbar(x="x", top="top", width=0.9, color="color", source=metric_source)
        metric_plot.xgrid.grid_line_color = None
        metric_plot.xaxis.major_label_orientation = 1.0

        histogram_plot = figure(
            title="Metric distribution",
            height=340,
            sizing_mode="stretch_width",
            toolbar_location=None,
        )
        histogram_plot.quad(
            top="top",
            bottom=0,
            left="left",
            right="right",
            alpha=0.6,
            source=histogram_source,
            color="#4c78a8",
        )

        run_summary = Div(text="<p>No run selected.</p>", sizing_mode="stretch_width")
        media_embed = Div(text="<p>No media available.</p>", sizing_mode="stretch_width")
        manifest_view = PreText(text="", sizing_mode="stretch_width", height=200)
        stdout_view = PreText(text="", sizing_mode="stretch_width", height=180)
        stderr_view = PreText(text="", sizing_mode="stretch_width", height=180)

        def selected_records() -> List[RunRecord]:
            active_models = set(model_filter.value)
            active_tiers = set(tier_filter.value)
            return [
                row
                for row in records
                if (not active_models or (row.model or "") in active_models)
                and (not active_tiers or (row.tier or "") in active_tiers)
            ]

        def update_metric_views(active: List[RunRecord]) -> None:
            chosen_metric = metric_select.value
            xs: List[str] = []
            tops: List[float] = []
            colors: List[str] = []
            palette = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]
            by_model_color: Dict[str, str] = {}
            for record in active:
                value = _safe_float(record.summary.get(chosen_metric))
                if value is None:
                    continue
                xs.append(record.run_id)
                tops.append(value)
                model_name = record.model or "unknown"
                if model_name not in by_model_color:
                    by_model_color[model_name] = palette[len(by_model_color) % len(palette)]
                colors.append(by_model_color[model_name])
            metric_source.data = {"x": xs, "top": tops, "color": colors}
            metric_plot.x_range.factors = xs
            metric_plot.title.text = f"Run-level {chosen_metric}"
            if tops:
                hist, edges = np.histogram(np.array(tops, dtype=float), bins=min(10, max(3, len(tops))))
                histogram_source.data = {
                    "left": edges[:-1].tolist(),
                    "right": edges[1:].tolist(),
                    "top": hist.tolist(),
                }
            else:
                histogram_source.data = {"left": [], "right": [], "top": []}
            histogram_plot.title.text = f"{chosen_metric} distribution"

        def update_selected_run(run_id: str) -> None:
            lookup = {row.run_id: row for row in records}
            record = lookup.get(run_id)
            if not record:
                run_summary.text = "<p>No run selected.</p>"
                media_select.options = []
                media_select.value = ""
                media_embed.text = "<p>No media available.</p>"
                manifest_view.text = ""
                stdout_view.text = ""
                stderr_view.text = ""
                return
            summary_lines = [
                f"<b>{record.run_id}</b>",
                f"Model: {record.model or 'unknown'}",
                f"Tier: {record.tier or 'unknown'}",
                f"Seed: {record.seed if record.seed is not None else 'n/a'}",
                f"Episodes: {record.summary.get('number_of_episodes', 'n/a')}",
                f"AR: {record.summary.get('ar_75_95', 'n/a')}",
                f"mAR: {record.summary.get('mar_75_95', 'n/a')}",
                f"Collision: {record.summary.get('collision_rate', 'n/a')}",
                f"Off-road: {record.summary.get('offroad_rate', 'n/a')}",
                f"Progress: {record.summary.get('progress_rate', 'n/a')}",
                f"Media files: {len(record.media_artifacts)}",
            ]
            run_summary.text = "<br>".join(summary_lines)
            media_options = [artifact.relative_path for artifact in record.media_artifacts]
            media_select.options = media_options
            media_select.value = media_options[0] if media_options else ""
            manifest_view.text = json.dumps(record.manifest, indent=2, sort_keys=True)
            stdout_view.text = _tail_text(record.stdout_path) or "No stdout log."
            stderr_view.text = _tail_text(record.stderr_path) or "No stderr log."
            update_selected_media(record, media_select.value)

        def update_selected_media(record: RunRecord | None, relative_path: str) -> None:
            if not record or not relative_path:
                media_embed.text = "<p>No media available.</p>"
                return
            artifact = next((item for item in record.media_artifacts if item.relative_path == relative_path), None)
            if artifact is None:
                media_embed.text = "<p>No media available.</p>"
                return
            url = artifact.artifact_url(ARTIFACT_ROUTE)
            if artifact.media_type == "video":
                media_embed.text = f'<video controls style="width: 100%; max-height: 720px;" src="{url}"></video>'
            elif artifact.media_type == "pdf":
                media_embed.text = (
                    f'<p><a href="{url}" target="_blank">Open PDF artifact</a></p>'
                    f'<iframe src="{url}" style="width: 100%; height: 720px; border: 1px solid #ddd;"></iframe>'
                )
            else:
                media_embed.text = f'<img src="{url}" style="max-width: 100%; max-height: 720px;" />'

        def refresh() -> None:
            active = selected_records()
            filtered_source.data = _overview_source(active)
            run_ids = [row.run_id for row in active]
            run_select.options = run_ids
            if run_ids and run_select.value not in run_ids:
                run_select.value = run_ids[0]
            if not run_ids:
                run_select.value = ""
            update_metric_views(active)
            update_selected_run(run_select.value)

        def on_filters_change(attr: str, old: Any, new: Any) -> None:
            refresh()

        def on_run_change(attr: str, old: Any, new: Any) -> None:
            update_selected_run(new)

        def on_media_change(attr: str, old: Any, new: Any) -> None:
            record = next((row for row in records if row.run_id == run_select.value), None)
            update_selected_media(record, new)

        model_filter.on_change("value", on_filters_change)
        tier_filter.on_change("value", on_filters_change)
        metric_select.on_change("value", on_filters_change)
        run_select.on_change("value", on_run_change)
        media_select.on_change("value", on_media_change)

        refresh()

        header = Div(
            text=f"<h2>WaymaxBoard</h2><p>Results root: <code>{self._results_dir}</code></p>",
            sizing_mode="stretch_width",
        )
        filters = row(model_filter, tier_filter, metric_select, sizing_mode="stretch_width")
        overview_tab = TabPanel(
            title="Overview",
            child=column(
                Div(text="<h3>Completed runs</h3>"),
                overview_table,
                Div(text="<h3>Suite summaries</h3>"),
                suite_table,
                sizing_mode="stretch_width",
            ),
        )
        metrics_tab = TabPanel(
            title="Metrics",
            child=column(metric_plot, histogram_plot, sizing_mode="stretch_width"),
        )
        artifacts_tab = TabPanel(
            title="Artifacts",
            child=column(
                row(run_select, media_select, sizing_mode="stretch_width"),
                run_summary,
                media_embed,
                Div(text="<h3>Run manifest</h3>"),
                manifest_view,
                Div(text="<h3>stdout tail</h3>"),
                stdout_view,
                Div(text="<h3>stderr tail</h3>"),
                stderr_view,
                sizing_mode="stretch_width",
            ),
        )
        doc.add_root(column(header, filters, Tabs(tabs=[overview_tab, metrics_tab, artifacts_tab]), sizing_mode="stretch_width"))
