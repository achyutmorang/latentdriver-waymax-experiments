from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .artifacts import write_json

_JSON_SUFFIXES = {".json", ".jsonl"}
_RECORD_CONTAINER_KEYS = ("records", "items", "data", "annotations", "examples", "scenarios", "rows")
_SCENARIO_ID_KEYS = ("sid", "scenario_id", "scenario/id")
_REASONING_AGENT_KEYS = ("rel_id", "related_agent_ids")
_EGO_AGENT_KEYS = ("ego", "ego_agent_id")
_CAUSAL_AGENT_KEYS = ("causal_agent_ids", "causal_ids")


def _normalize_id(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value)
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _collect_scalar_ids(value: Any) -> set[str]:
    ids: set[str] = set()
    normalized = _normalize_id(value)
    if normalized is not None:
        ids.add(normalized)
        return ids
    if isinstance(value, Mapping):
        for item in value.values():
            ids.update(_collect_scalar_ids(item))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            ids.update(_collect_scalar_ids(item))
    return ids


def _find_key_values(payload: Any, keys: Sequence[str]) -> list[Any]:
    found: list[Any] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if key in keys:
                found.append(value)
            found.extend(_find_key_values(value, keys))
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            found.extend(_find_key_values(item, keys))
    return found


def _iter_dataset_files(path: Path) -> Iterator[Path]:
    if path.is_file():
        if path.suffix.lower() not in _JSON_SUFFIXES:
            raise ValueError(f"Unsupported metadata file suffix for {path}")
        yield path
        return
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_dir():
        raise ValueError(f"Expected file or directory: {path}")
    files = sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in _JSON_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No .json or .jsonl metadata files found under {path}")
    yield from files


def _iter_records_from_payload(payload: Any) -> Iterator[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, Mapping):
                yield dict(item)
        return
    if isinstance(payload, Mapping):
        for key in _RECORD_CONTAINER_KEYS:
            if key not in payload:
                continue
            value = payload[key]
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, Mapping):
                        yield dict(item)
                return
            if isinstance(value, Mapping):
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, Mapping):
                        record = dict(nested_value)
                        record.setdefault("_top_level_key", nested_key)
                        yield record
                return
        if payload and all(isinstance(value, Mapping) for value in payload.values()):
            for nested_key, nested_value in payload.items():
                record = dict(nested_value)
                record.setdefault("_top_level_key", nested_key)
                yield record
            return
        yield dict(payload)


def _iter_records(path: Path) -> Iterator[dict[str, Any]]:
    for file_path in _iter_dataset_files(path):
        if file_path.suffix.lower() == ".jsonl":
            for line_number, raw_line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                if not isinstance(record, Mapping):
                    continue
                item = dict(record)
                item["_source_file"] = str(file_path)
                item["_source_line"] = line_number
                yield item
            continue
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        for record in _iter_records_from_payload(payload):
            record["_source_file"] = str(file_path)
            yield record


def _extract_scenario_id(record: Mapping[str, Any]) -> str | None:
    for key in _SCENARIO_ID_KEYS:
        if key in record:
            normalized = _normalize_id(record[key])
            if normalized is not None:
                return normalized
    for value in _find_key_values(record, _SCENARIO_ID_KEYS):
        normalized = _normalize_id(value)
        if normalized is not None:
            return normalized
    return _normalize_id(record.get("_top_level_key"))


def _extract_reasoning_related_ids(record: Mapping[str, Any]) -> set[str]:
    related: set[str] = set()
    for value in _find_key_values(record, _REASONING_AGENT_KEYS):
        related.update(_collect_scalar_ids(value))
    return related


def _extract_reasoning_ego_ids(record: Mapping[str, Any]) -> set[str]:
    ego_ids: set[str] = set()
    for value in _find_key_values(record, _EGO_AGENT_KEYS):
        ego_ids.update(_collect_scalar_ids(value))
    return ego_ids


def _extract_causal_labeler_sets(payload: Any) -> list[set[str]]:
    labeler_sets: list[set[str]] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if key in _CAUSAL_AGENT_KEYS:
                ids = _collect_scalar_ids(value)
                if ids:
                    labeler_sets.append(ids)
            else:
                labeler_sets.extend(_extract_causal_labeler_sets(value))
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            labeler_sets.extend(_extract_causal_labeler_sets(item))
    return labeler_sets


def _merge_causal_sets(labeler_sets: Sequence[set[str]], *, policy: str) -> set[str]:
    if not labeler_sets:
        return set()
    if policy == "union":
        merged: set[str] = set()
        for item in labeler_sets:
            merged.update(item)
        return merged
    if policy == "majority":
        counts: Counter[str] = Counter()
        non_empty = 0
        for labeler_ids in labeler_sets:
            if not labeler_ids:
                continue
            non_empty += 1
            counts.update(labeler_ids)
        if non_empty == 0:
            return set()
        threshold = non_empty // 2 + 1
        return {agent_id for agent_id, count in counts.items() if count >= threshold}
    raise ValueError(f"Unsupported causal merge policy: {policy!r}")


@dataclass
class ScenarioMetadata:
    scenario_id: str
    record_count: int
    related_agent_ids: set[str]
    ego_agent_ids: set[str]
    causal_labeler_sets: list[set[str]]
    source_files: set[str]

    def causal_agent_ids(self, *, policy: str) -> set[str]:
        return _merge_causal_sets(self.causal_labeler_sets, policy=policy)


def _build_reasoning_index(path: Path) -> tuple[dict[str, ScenarioMetadata], dict[str, Any]]:
    index: dict[str, ScenarioMetadata] = {}
    stats = {"records_scanned": 0, "records_without_scenario_id": 0, "files": sorted(str(item) for item in _iter_dataset_files(path))}
    for record in _iter_records(path):
        stats["records_scanned"] += 1
        scenario_id = _extract_scenario_id(record)
        if scenario_id is None:
            stats["records_without_scenario_id"] += 1
            continue
        item = index.get(scenario_id)
        if item is None:
            item = ScenarioMetadata(
                scenario_id=scenario_id,
                record_count=0,
                related_agent_ids=set(),
                ego_agent_ids=set(),
                causal_labeler_sets=[],
                source_files=set(),
            )
            index[scenario_id] = item
        item.record_count += 1
        item.related_agent_ids.update(_extract_reasoning_related_ids(record))
        item.ego_agent_ids.update(_extract_reasoning_ego_ids(record))
        source_file = _normalize_id(record.get("_source_file"))
        if source_file is not None:
            item.source_files.add(source_file)
    stats["scenario_count"] = len(index)
    return index, stats


def _build_causal_index(path: Path) -> tuple[dict[str, ScenarioMetadata], dict[str, Any]]:
    index: dict[str, ScenarioMetadata] = {}
    stats = {"records_scanned": 0, "records_without_scenario_id": 0, "files": sorted(str(item) for item in _iter_dataset_files(path))}
    for record in _iter_records(path):
        stats["records_scanned"] += 1
        scenario_id = _extract_scenario_id(record)
        if scenario_id is None:
            stats["records_without_scenario_id"] += 1
            continue
        item = index.get(scenario_id)
        if item is None:
            item = ScenarioMetadata(
                scenario_id=scenario_id,
                record_count=0,
                related_agent_ids=set(),
                ego_agent_ids=set(),
                causal_labeler_sets=[],
                source_files=set(),
            )
            index[scenario_id] = item
        item.record_count += 1
        item.causal_labeler_sets.extend(_extract_causal_labeler_sets(record))
        source_file = _normalize_id(record.get("_source_file"))
        if source_file is not None:
            item.source_files.add(source_file)
    stats["scenario_count"] = len(index)
    return index, stats


def load_scenario_subset(
    *,
    scenario_ids_file: Path | None = None,
    preprocess_map_dir: Path | None = None,
    limit: int | None = None,
) -> list[str] | None:
    scenario_ids: list[str] = []
    if scenario_ids_file is not None:
        for raw_line in scenario_ids_file.read_text(encoding="utf-8").splitlines():
            normalized = _normalize_id(raw_line)
            if normalized is not None:
                scenario_ids.append(normalized)
    if preprocess_map_dir is not None:
        stems = sorted(path.stem for path in preprocess_map_dir.glob("*.npy"))
        scenario_ids.extend(stem for stem in stems if _normalize_id(stem) is not None)
    if not scenario_ids:
        return None
    deduped = list(dict.fromkeys(scenario_ids))
    if limit is not None:
        deduped = deduped[:limit]
    return deduped


def _sample(values: Iterable[str], *, limit: int = 10) -> list[str]:
    return sorted(values)[:limit]


def check_reasoning_causal_join(
    *,
    reasoning_path: Path,
    causal_path: Path,
    scenario_ids: Sequence[str] | None = None,
    causal_policy: str = "union",
    output_dir: Path | None = None,
) -> dict[str, Any]:
    reasoning_index, reasoning_stats = _build_reasoning_index(reasoning_path)
    causal_index, causal_stats = _build_causal_index(causal_path)

    reasoning_ids_all = set(reasoning_index)
    causal_ids_all = set(causal_index)
    if scenario_ids is not None:
        subset_ids = list(dict.fromkeys(scenario_ids))
    else:
        subset_ids = sorted(reasoning_ids_all)
    subset_id_set = set(subset_ids)

    reasoning_ids = reasoning_ids_all & subset_id_set
    causal_ids = causal_ids_all & subset_id_set
    joined_ids = reasoning_ids & causal_ids
    reasoning_only_ids = reasoning_ids - causal_ids
    causal_only_ids = causal_ids - reasoning_ids

    joined_rows: list[dict[str, Any]] = []
    joined_with_agent_overlap = 0
    agent_overlap_ids: set[str] = set()
    for scenario_id in sorted(joined_ids):
        reasoning = reasoning_index[scenario_id]
        causal = causal_index[scenario_id]
        causal_agent_ids = causal.causal_agent_ids(policy=causal_policy)
        overlap = reasoning.related_agent_ids & causal_agent_ids
        if overlap:
            joined_with_agent_overlap += 1
            agent_overlap_ids.update(overlap)
        joined_rows.append(
            {
                "scenario_id": scenario_id,
                "reasoning_record_count": reasoning.record_count,
                "causal_record_count": causal.record_count,
                "reasoning_related_agent_ids": sorted(reasoning.related_agent_ids),
                "reasoning_ego_agent_ids": sorted(reasoning.ego_agent_ids),
                "causal_agent_ids": sorted(causal_agent_ids),
                "causal_labeler_set_count": len([item for item in causal.causal_labeler_sets if item]),
                "agent_overlap_ids": sorted(overlap),
                "has_agent_overlap": bool(overlap),
            }
        )

    subset_reasoning_count = len(reasoning_ids)
    subset_causal_count = len(causal_ids)
    joined_count = len(joined_ids)
    summary = {
        "reasoning_path": str(reasoning_path),
        "causal_path": str(causal_path),
        "causal_policy": causal_policy,
        "reasoning_stats": reasoning_stats,
        "causal_stats": causal_stats,
        "subset": {
            "requested_scenario_count": len(subset_ids),
            "requested_scenario_ids_sample": _sample(subset_ids),
            "reasoning_scenarios_in_subset": subset_reasoning_count,
            "causal_scenarios_in_subset": subset_causal_count,
        },
        "join": {
            "joined_scenario_count": joined_count,
            "intersection_rate_vs_reasoning_subset": (joined_count / subset_reasoning_count) if subset_reasoning_count else 0.0,
            "intersection_rate_vs_causal_subset": (joined_count / subset_causal_count) if subset_causal_count else 0.0,
            "reasoning_only_scenario_count": len(reasoning_only_ids),
            "causal_only_scenario_count": len(causal_only_ids),
            "joined_scenarios_with_agent_overlap": joined_with_agent_overlap,
            "agent_overlap_rate": (joined_with_agent_overlap / joined_count) if joined_count else 0.0,
            "distinct_overlapping_agent_id_count": len(agent_overlap_ids),
            "joined_scenario_ids_sample": _sample(joined_ids),
            "reasoning_only_scenario_ids_sample": _sample(reasoning_only_ids),
            "causal_only_scenario_ids_sample": _sample(causal_only_ids),
            "overlapping_agent_ids_sample": _sample(agent_overlap_ids),
        },
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "summary.json", summary)
        (output_dir / "joined_metadata.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in joined_rows),
            encoding="utf-8",
        )
        (output_dir / "scenario_ids_joined.txt").write_text(
            "".join(f"{scenario_id}\n" for scenario_id in sorted(joined_ids)),
            encoding="utf-8",
        )
        (output_dir / "scenario_ids_reasoning_only.txt").write_text(
            "".join(f"{scenario_id}\n" for scenario_id in sorted(reasoning_only_ids)),
            encoding="utf-8",
        )
        (output_dir / "scenario_ids_causal_only.txt").write_text(
            "".join(f"{scenario_id}\n" for scenario_id in sorted(causal_only_ids)),
            encoding="utf-8",
        )
    return summary
