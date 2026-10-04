# Copyright 2026 the Awesome-Perception-Aware-RLVR authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import csv
import fcntl
from pathlib import Path
from typing import Any

from .schemas import PRIMARY_NA_STATUS, MetricResult


# Secondary columns of the global summary (not counted in the averages): the GRIT sets report
# answer accuracy (primary) and the GRIT grounding IoU, as in the GRIT paper.
AUXILIARY_COLUMNS = {
    "grit_vsr": ("grit_vsr_giou", "grounding/grit_iou"),
    "grit_tallyqa": ("grit_tallyqa_giou", "grounding/grit_iou"),
    "grit_gqa": ("grit_gqa_giou", "grounding/grit_iou"),
    "tallyqa_relabeled": ("tallyqa_relabeled_giou", "grounding/grit_iou"),
}


# run options recorded only when set (the defaults leave them empty)
OPTIONAL_RUN_FIELDNAMES = [
    "top_k",
    "grounding_instruction",
    "answer_protocol",
    "max_dynamic_patch",
    "agent_prompt_style",
    "agent_observation_min_pixels",
]

SUMMARY_FIELDNAMES = [
    "run_id",
    "output_dir",
    "model",
    "backend",
    "temperature",
    "num_samples",
    "batch_size",
    "max_batch_images",
    "max_model_len",
    "gpu_memory_utilization",
    "min_pixels",
    "max_pixels",
    "format_prompt",
    "system_prompt",
    "chat_template",
    "plain_think_tokens",
    "prompt_mode",
    "box_format",
    *OPTIONAL_RUN_FIELDNAMES,
    "interaction_mode",
    "agent_profile",
    "agent_output_contract",
    "agent_runner_version",
    "agent_max_tool_calls",
    "agent_max_response_tokens",
    "agent_max_tokens_per_turn",
    "agent_max_images_per_prompt",
    "agent_tool_image_mode",
    "judge_provider",
    "judge_model",
    "perturbation_type",
    "perturbation_summary",
    "perturbation_seed",
    "perturbation_params",
    "row_type",
    "group",
    "benchmark",
    "primary_metric",
    "raw_score",
    "normalized_score_0_100",
    "num_examples",
    "status",
    "truncated_rate",
    "truncated_count",
    "response_count",
    "sample_truncated_rate",
    "samples_with_truncation",
    "details_json",
]

GLOBAL_META_FIELDNAMES = [
    "tag",
    "model",
    "created_at",
    "output_dir",
    "system_prompt",
    "format_prompt",
    *OPTIONAL_RUN_FIELDNAMES,
    "interaction_mode",
    "agent_profile",
    "agent_output_contract",
    "agent_runner_version",
    "agent_max_tool_calls",
    "agent_max_response_tokens",
    "agent_max_tokens_per_turn",
    "agent_max_images_per_prompt",
    "agent_tool_image_mode",
]


def write_summary_csv(
    path: Path,
    results: list[MetricResult],
    *,
    run_metadata: dict[str, Any],
) -> None:
    rows = build_summary_rows(results, run_metadata=run_metadata)
    _write_summary_rows(path, rows)


def update_global_summary_csv(
    path: Path,
    results: list[MetricResult],
    *,
    run_metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    new_row, group_header, fieldnames = build_global_summary_row(results, run_metadata=run_metadata)
    run_key = _run_key(run_metadata)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        existing_group_header, existing_fieldnames, existing_rows = (
            _read_global_summary(path) if path.exists() else ([], [], [])
        )
        fieldnames, group_header = _merge_global_headers(
            existing_fieldnames,
            existing_group_header,
            fieldnames,
            group_header,
        )
        # Replace an existing run's row in place so re-scoring never disturbs the
        # (possibly hand-curated) row order; only genuinely new runs are appended.
        merged_rows = list(existing_rows)
        for index, row in enumerate(merged_rows):
            if _global_row_key(row) == run_key:
                merged_rows[index] = new_row
                break
        else:
            merged_rows.append(new_row)
        _write_global_rows(path, merged_rows, fieldnames=fieldnames, group_header=group_header)
        fcntl.flock(lock_file, fcntl.LOCK_UN)


def build_summary_rows(results: list[MetricResult], *, run_metadata: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        rows.append(_result_row(result, run_metadata, row_type="benchmark"))

    groups = sorted({result.group for result in results})
    for group in groups:
        group_results = [result for result in results if result.group == group and result.status == "ok"]
        if not group_results:
            continue
        rows.append(
            {
                **_base_metadata(_aggregate_metadata(group_results, run_metadata)),
                "row_type": "group_average",
                "group": group,
                "benchmark": f"{group}_average",
                "primary_metric": "normalized_score_0_100_mean",
                "raw_score": "",
                "normalized_score_0_100": _mean_result_scores(group_results),
                "num_examples": sum(item.num_examples for item in group_results),
                "status": "ok",
                "details_json": "",
            }
        )

    ok_results = [result for result in results if result.status == "ok"]
    if ok_results:
        rows.append(
            {
                **_base_metadata(_aggregate_metadata(ok_results, run_metadata)),
                "row_type": "overall",
                "group": "overall",
                "benchmark": "overall_average",
                "primary_metric": "normalized_score_0_100_mean",
                "raw_score": "",
                "normalized_score_0_100": _mean_result_scores(ok_results),
                "num_examples": sum(item.num_examples for item in ok_results),
                "status": "ok",
                "details_json": "",
            }
        )
    return rows


def build_global_summary_row(
    results: list[MetricResult],
    *,
    run_metadata: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    row: dict[str, Any] = {
        "tag": run_metadata.get("run_id", ""),
        "model": run_metadata.get("model", ""),
        "created_at": run_metadata.get("created_at", ""),
        "output_dir": run_metadata.get("output_dir", ""),
        "system_prompt": run_metadata.get("system_prompt", ""),
        "format_prompt": run_metadata.get("format_prompt", ""),
        **{key: run_metadata.get(key, "") for key in OPTIONAL_RUN_FIELDNAMES},
        "interaction_mode": run_metadata.get("interaction_mode", "one_shot"),
        "agent_profile": run_metadata.get("agent_profile", ""),
        "agent_output_contract": run_metadata.get("agent_output_contract", ""),
        "agent_runner_version": run_metadata.get("agent_runner_version", ""),
        "agent_max_tool_calls": run_metadata.get("agent_max_tool_calls", ""),
        "agent_max_response_tokens": run_metadata.get(
            "agent_max_response_tokens",
            "",
        ),
        "agent_max_tokens_per_turn": run_metadata.get(
            "agent_max_tokens_per_turn",
            "",
        ),
        "agent_max_images_per_prompt": run_metadata.get(
            "agent_max_images_per_prompt",
            "",
        ),
        "agent_tool_image_mode": run_metadata.get(
            "agent_tool_image_mode",
            "",
        ),
    }
    fieldnames = list(GLOBAL_META_FIELDNAMES)
    group_header = ["Meta"] * len(GLOBAL_META_FIELDNAMES)

    for group in _ordered_groups(results):
        group_results = [result for result in results if result.group == group and result.status == "ok"]
        displayed_results = [
            result for result in results if result.group == group and result.status in {"ok", PRIMARY_NA_STATUS}
        ]
        for result in displayed_results:
            fieldnames.append(result.benchmark)
            group_header.append(group)
            row[result.benchmark] = (
                "n/a" if result.status == PRIMARY_NA_STATUS else _format_global_score(_require_score(result))
            )
            auxiliary = AUXILIARY_COLUMNS.get(result.benchmark)
            if auxiliary is not None:
                auxiliary_column, detail_key = auxiliary
                fieldnames.append(auxiliary_column)
                group_header.append(group)
                value = result.details.get(detail_key)
                if isinstance(value, (int, float)):
                    row[auxiliary_column] = _format_global_score(float(value) * 100.0)
        if group_results:
            avg_name = f"{group} Avg"
            fieldnames.append(avg_name)
            group_header.append(group)
            row[avg_name] = _format_global_score(_mean_result_scores(group_results))

    ok_results = [result for result in results if result.status == "ok"]
    if ok_results:
        fieldnames.append("Overall Avg")
        group_header.append("Overall")
        row["Overall Avg"] = _format_global_score(_mean_result_scores(ok_results))

    return row, group_header, fieldnames


def _write_summary_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDNAMES})


def _write_global_rows(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: list[str],
    group_header: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_dedup_group_header(group_header))
        writer.writerow(fieldnames)
        for row in rows:
            writer.writerow([row.get(field, "") for field in fieldnames])


def _read_summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_global_summary(path: Path) -> tuple[list[str], list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return [], [], []
    group_header = _expand_group_header(rows[0])
    fieldnames = rows[1]
    data_rows = [dict(zip(fieldnames, row)) for row in rows[2:]]
    return group_header, fieldnames, data_rows


def _result_row(result: MetricResult, run_metadata: dict[str, Any], *, row_type: str) -> dict[str, Any]:
    import json

    return {
        **_base_metadata(_result_metadata(result, run_metadata)),
        "row_type": row_type,
        "group": result.group,
        "benchmark": result.benchmark,
        "primary_metric": result.primary_metric,
        "raw_score": ("n/a" if result.status == PRIMARY_NA_STATUS else result.raw_score),
        "normalized_score_0_100": ("n/a" if result.status == PRIMARY_NA_STATUS else result.normalized_score_0_100),
        "num_examples": result.num_examples,
        "status": result.status,
        "truncated_rate": result.details.get("generation/truncated_rate", ""),
        "truncated_count": result.details.get("generation/truncated_count", ""),
        "response_count": result.details.get("generation/response_count", ""),
        "sample_truncated_rate": result.details.get("generation/sample_truncated_rate", ""),
        "samples_with_truncation": result.details.get("generation/samples_with_truncation", ""),
        "details_json": json.dumps(result.details, ensure_ascii=False, sort_keys=True),
    }


def _result_metadata(result: MetricResult, run_metadata: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(run_metadata)
    metadata.update(result.metadata)
    return metadata


def _aggregate_metadata(results: list[MetricResult], run_metadata: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(run_metadata)
    for key in SUMMARY_FIELDNAMES:
        values = {str(result.metadata[key]) for result in results if key in result.metadata}
        if len(values) == 1:
            metadata[key] = next(iter(values))
        elif len(values) > 1:
            metadata[key] = "mixed"
    return metadata


def _base_metadata(run_metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run_metadata.get("run_id", ""),
        "output_dir": run_metadata.get("output_dir", ""),
        "model": run_metadata.get("model", ""),
        "backend": run_metadata.get("backend", ""),
        "temperature": run_metadata.get("temperature", ""),
        "num_samples": run_metadata.get("num_samples", ""),
        "batch_size": run_metadata.get("batch_size", ""),
        "max_batch_images": run_metadata.get("max_batch_images", ""),
        "max_model_len": run_metadata.get("max_model_len", ""),
        "gpu_memory_utilization": run_metadata.get("gpu_memory_utilization", ""),
        "min_pixels": run_metadata.get("min_pixels", ""),
        "max_pixels": run_metadata.get("max_pixels", ""),
        "format_prompt": run_metadata.get("format_prompt", ""),
        "system_prompt": run_metadata.get("system_prompt", ""),
        "chat_template": run_metadata.get("chat_template", ""),
        "plain_think_tokens": run_metadata.get("plain_think_tokens", ""),
        "prompt_mode": run_metadata.get("prompt_mode", ""),
        "box_format": run_metadata.get("box_format", ""),
        "interaction_mode": run_metadata.get("interaction_mode", "one_shot"),
        "agent_profile": run_metadata.get("agent_profile", ""),
        "agent_output_contract": run_metadata.get("agent_output_contract", ""),
        "agent_runner_version": run_metadata.get("agent_runner_version", ""),
        "agent_max_tool_calls": run_metadata.get("agent_max_tool_calls", ""),
        "agent_max_response_tokens": run_metadata.get(
            "agent_max_response_tokens",
            "",
        ),
        "agent_max_tokens_per_turn": run_metadata.get(
            "agent_max_tokens_per_turn",
            "",
        ),
        "agent_max_images_per_prompt": run_metadata.get(
            "agent_max_images_per_prompt",
            "",
        ),
        "agent_tool_image_mode": run_metadata.get(
            "agent_tool_image_mode",
            "",
        ),
        "judge_provider": run_metadata.get("judge_provider", ""),
        "judge_model": run_metadata.get("judge_model", ""),
        "perturbation_type": run_metadata.get("perturbation_type", ""),
        "perturbation_summary": run_metadata.get("perturbation_summary", ""),
        "perturbation_seed": run_metadata.get("perturbation_seed", ""),
        "perturbation_params": run_metadata.get("perturbation_params", ""),
        **{key: run_metadata.get(key, "") for key in OPTIONAL_RUN_FIELDNAMES},
    }


def _run_key(run_metadata: dict[str, Any]) -> tuple[str, str]:
    return str(run_metadata.get("output_dir", "")), str(run_metadata.get("run_id", ""))


def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("output_dir", "")), str(row.get("run_id", ""))


def _global_row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("output_dir", "")), str(row.get("tag", ""))


def _ordered_groups(results: list[MetricResult]) -> list[str]:
    groups = []
    seen = set()
    for result in results:
        if result.group not in seen:
            groups.append(result.group)
            seen.add(result.group)
    return groups


def _merge_global_headers(
    existing_fieldnames: list[str],
    existing_group_header: list[str],
    new_fieldnames: list[str],
    new_group_header: list[str],
) -> tuple[list[str], list[str]]:
    field_order = []
    seen_fields = set()
    for field in existing_fieldnames + new_fieldnames:
        if field not in seen_fields:
            field_order.append(field)
            seen_fields.add(field)

    group_by_field = {
        field: existing_group_header[index] if index < len(existing_group_header) else ""
        for index, field in enumerate(existing_fieldnames)
    }
    for index, field in enumerate(new_fieldnames):
        group_by_field[field] = new_group_header[index] if index < len(new_group_header) else ""

    ordinary_groups = []
    seen_groups = set()
    has_meta = False
    has_overall = False
    for field in field_order:
        group = group_by_field.get(field, "")
        if group == "Meta":
            has_meta = True
        elif group == "Overall":
            has_overall = True
        elif group not in seen_groups:
            ordinary_groups.append(group)
            seen_groups.add(group)

    group_order = (["Meta"] if has_meta else []) + ordinary_groups + (["Overall"] if has_overall else [])
    fieldnames = []
    group_header = []
    for group in group_order:
        group_fields = [field for field in field_order if group_by_field.get(field, "") == group]
        if group == "Meta":
            known_meta = [field for field in GLOBAL_META_FIELDNAMES if field in group_fields]
            group_fields = known_meta + [field for field in group_fields if field not in known_meta]
        group_fields = _place_auxiliary_columns(group_fields)
        average_field = "Overall Avg" if group == "Overall" else f"{group} Avg"
        if average_field in group_fields:
            group_fields = [field for field in group_fields if field != average_field] + [average_field]
        fieldnames.extend(group_fields)
        group_header.extend([group] * len(group_fields))
    return fieldnames, group_header


def _format_global_score(value: float) -> str:
    return f"{value:.2f}"


def _place_auxiliary_columns(fields: list[str]) -> list[str]:
    auxiliary_columns = {column for column, _ in AUXILIARY_COLUMNS.values()}
    ordered = [field for field in fields if field not in auxiliary_columns]
    for benchmark, (auxiliary, _) in AUXILIARY_COLUMNS.items():
        if auxiliary not in fields:
            continue
        if benchmark in ordered:
            ordered.insert(ordered.index(benchmark) + 1, auxiliary)
        else:
            ordered.append(auxiliary)
    return ordered


def _require_score(result: MetricResult) -> float:
    value = result.normalized_score_0_100
    if value is None:
        raise ValueError(f"{result.benchmark} has status=ok but no normalized score")
    return float(value)


def _mean_result_scores(results: list[MetricResult]) -> float:
    return _mean(_require_score(result) for result in results)


def _dedup_group_header(group_header: list[str]) -> list[str]:
    """Only write the group name at the first column of each run; leave the rest empty."""
    result = []
    prev = None
    for group in group_header:
        if group == prev:
            result.append("")
        else:
            result.append(group)
            prev = group
    return result


def _expand_group_header(deduped: list[str]) -> list[str]:
    """Reverse of _dedup_group_header: fill empty cells with the last non-empty group."""
    result = []
    current = ""
    for cell in deduped:
        if cell:
            current = cell
        result.append(current)
    return result


def _mean(values) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0
