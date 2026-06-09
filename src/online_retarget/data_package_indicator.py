"""Versioned data-package indicator parsing for supervised motionlib training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


SCHEMA = "online_retarget.package_indicator.v1"
SPEC_VALUES = {"kin", "phy"}
CATEGORY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
REQUIRED_COLUMNS = (
    "pair_id",
    "relative_path",
    "robot_relative_path",
    "soma_relative_path",
    "source_bvh",
)


@dataclass(frozen=True)
class PackageIndicatorRow:
    pair_id: str
    relative_path: str
    robot_relative_path: str
    soma_relative_path: str
    source_bvh: str
    line_number: int


@dataclass(frozen=True)
class PackageIndicator:
    path: Path
    schema: str
    spec: str
    category: str
    columns: tuple[str, ...]
    rows: tuple[PackageIndicatorRow, ...]
    indicator_sha256: str
    package_rows_sha256: str


def normalize_package_path(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_pair_id(robot_relative_path: Any, soma_relative_path: Any, source_bvh: Any) -> str:
    payload = "\0".join(
        (
            normalize_package_path(robot_relative_path),
            normalize_package_path(soma_relative_path),
            normalize_package_path(source_bvh),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def row_package_fields(row: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    robot_relative_path = normalize_package_path(row.get("robot_relative_path"))
    soma_relative_path = normalize_package_path(row.get("soma_relative_path"))
    source_bvh = normalize_package_path(row.get("source_bvh") or row.get("source_soma_proportional_path"))
    pair_id = normalize_package_path(row.get("data_package_pair_id")) or package_pair_id(
        robot_relative_path,
        soma_relative_path,
        source_bvh,
    )
    relative_path = normalize_package_path(row.get("relative_path") or robot_relative_path)
    return pair_id, relative_path, robot_relative_path, soma_relative_path, source_bvh


def package_rows_sha256(rows: Sequence[Mapping[str, Any] | PackageIndicatorRow]) -> str:
    lines: list[str] = []
    for row in rows:
        if isinstance(row, PackageIndicatorRow):
            fields = (
                row.pair_id,
                row.relative_path,
                row.robot_relative_path,
                row.soma_relative_path,
                row.source_bvh,
            )
        else:
            fields = row_package_fields(row)
        lines.append("\t".join(normalize_package_path(field) for field in fields))
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _parse_header_comment(line: str) -> tuple[str, str] | None:
    text = line.lstrip()[1:].strip()
    if "=" not in text:
        return None
    key, value = text.split("=", 1)
    return key.strip(), value.strip()


def _require_slug(kind: str, value: str, allowed: set[str] | None = None) -> str:
    value = value.strip()
    if allowed is not None and value not in allowed:
        raise ValueError(f"invalid data_package {kind}: {value!r}")
    if allowed is None and not CATEGORY_RE.match(value):
        raise ValueError(f"invalid data_package {kind}: {value!r}")
    return value


def parse_package_indicator(path: Path | str) -> PackageIndicator:
    indicator_path = Path(path).expanduser()
    if not indicator_path.exists():
        raise FileNotFoundError(f"data_package indicator is missing: {indicator_path}")

    header_comments: dict[str, str] = {}
    columns: list[str] | None = None
    rows: list[PackageIndicatorRow] = []
    seen_pair_ids: set[str] = set()
    with indicator_path.open("r", encoding="utf-8", newline="") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue
            if line.lstrip().startswith("#"):
                parsed = _parse_header_comment(line)
                if parsed is not None:
                    header_comments[parsed[0]] = parsed[1]
                continue
            values = line.split("\t")
            if columns is None:
                columns = values
                duplicate_columns = sorted({column for column in columns if columns.count(column) > 1})
                if duplicate_columns:
                    raise ValueError(f"data_package indicator duplicate columns: {duplicate_columns}")
                missing_columns = [column for column in REQUIRED_COLUMNS if column not in columns]
                if missing_columns:
                    raise ValueError(f"data_package indicator missing required columns: {missing_columns}")
                continue
            if len(values) != len(columns):
                raise ValueError(
                    f"data_package indicator line {line_number} has {len(values)} fields, expected {len(columns)}"
                )
            record = dict(zip(columns, values))
            required_values = {
                key: normalize_package_path(record.get(key, ""))
                for key in REQUIRED_COLUMNS
            }
            for key in ("pair_id", "relative_path", "robot_relative_path", "soma_relative_path"):
                if not required_values[key]:
                    raise ValueError(f"data_package indicator line {line_number} has empty {key}")
            expected_pair_id = package_pair_id(
                required_values["robot_relative_path"],
                required_values["soma_relative_path"],
                required_values["source_bvh"],
            )
            if required_values["pair_id"] != expected_pair_id:
                raise ValueError(
                    f"data_package indicator line {line_number} pair_id mismatch: "
                    f"{required_values['pair_id']} != {expected_pair_id}"
                )
            if required_values["pair_id"] in seen_pair_ids:
                raise ValueError(f"data_package indicator duplicate pair_id: {required_values['pair_id']}")
            seen_pair_ids.add(required_values["pair_id"])
            rows.append(
                PackageIndicatorRow(
                    pair_id=required_values["pair_id"],
                    relative_path=required_values["relative_path"],
                    robot_relative_path=required_values["robot_relative_path"],
                    soma_relative_path=required_values["soma_relative_path"],
                    source_bvh=required_values["source_bvh"],
                    line_number=line_number,
                )
            )

    if columns is None:
        raise ValueError(f"data_package indicator has no TSV header: {indicator_path}")
    schema = header_comments.get("schema", "")
    if schema != SCHEMA:
        raise ValueError(f"data_package indicator schema mismatch: {schema!r} != {SCHEMA!r}")
    spec = _require_slug("spec", header_comments.get("spec", ""), SPEC_VALUES)
    category = _require_slug("category", header_comments.get("category", ""))
    return PackageIndicator(
        path=indicator_path,
        schema=schema,
        spec=spec,
        category=category,
        columns=tuple(columns),
        rows=tuple(rows),
        indicator_sha256=file_sha256(indicator_path),
        package_rows_sha256=package_rows_sha256(rows),
    )


def data_package_config(input_data: Mapping[str, Any]) -> Mapping[str, Any] | None:
    cfg = input_data.get("data_package")
    if cfg in (None, "", False):
        return None
    if not isinstance(cfg, Mapping):
        raise TypeError("input_data.data_package must be an object")
    spec = _require_slug("spec", str(cfg.get("spec", "")), SPEC_VALUES)
    category = _require_slug("category", str(cfg.get("category", "")))
    indicator = str(cfg.get("indicator", "")).strip()
    if not indicator:
        raise ValueError("input_data.data_package.indicator is required")
    missing_policy = str(cfg.get("missing_policy", "error")).strip() or "error"
    if missing_policy != "error":
        raise ValueError("input_data.data_package.missing_policy currently supports only 'error'")
    return {
        "spec": spec,
        "category": category,
        "indicator": indicator,
        "missing_policy": missing_policy,
    }


def _copy_package_row(row: Mapping[str, Any], indicator_row: PackageIndicatorRow) -> dict[str, Any]:
    copied = dict(row)
    copied["data_package_pair_id"] = indicator_row.pair_id
    copied["data_package_spec"] = ""
    copied["data_package_category"] = ""
    return copied


def filter_rows_by_data_package_config(
    rows: Sequence[Mapping[str, Any]],
    input_data: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    cfg = data_package_config(input_data)
    if cfg is None:
        return [dict(row) for row in rows], None

    indicator = parse_package_indicator(cfg["indicator"])
    if indicator.spec != cfg["spec"] or indicator.category != cfg["category"]:
        raise ValueError(
            "data_package indicator identity mismatch: "
            f"config={cfg['spec']}/{cfg['category']} indicator={indicator.spec}/{indicator.category}"
        )

    rows_by_pair_id: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        pair_id, _relative_path, _robot_relative_path, _soma_relative_path, _source_bvh = row_package_fields(row)
        rows_by_pair_id.setdefault(pair_id, []).append(row)

    selected_before_max: list[dict[str, Any]] = []
    errors: list[str] = []
    for indicator_row in indicator.rows:
        matches = rows_by_pair_id.get(indicator_row.pair_id, [])
        if not matches:
            errors.append(f"missing base row for pair_id={indicator_row.pair_id}")
            continue
        if len(matches) > 1:
            errors.append(f"duplicate base rows for pair_id={indicator_row.pair_id}")
            continue
        row = matches[0]
        pair_id, relative_path, robot_relative_path, soma_relative_path, source_bvh = row_package_fields(row)
        if (
            relative_path != indicator_row.relative_path
            or robot_relative_path != indicator_row.robot_relative_path
            or soma_relative_path != indicator_row.soma_relative_path
        ):
            errors.append(
                "path mismatch for pair_id="
                f"{indicator_row.pair_id}: base={relative_path}/{robot_relative_path}/{soma_relative_path} "
                f"indicator={indicator_row.relative_path}/{indicator_row.robot_relative_path}/"
                f"{indicator_row.soma_relative_path}"
            )
            continue
        if indicator_row.source_bvh and source_bvh != indicator_row.source_bvh:
            errors.append(
                f"source_bvh mismatch for pair_id={indicator_row.pair_id}: "
                f"base={source_bvh!r} indicator={indicator_row.source_bvh!r}"
            )
            continue
        selected = _copy_package_row(row, indicator_row)
        selected["data_package_spec"] = indicator.spec
        selected["data_package_category"] = indicator.category
        selected_before_max.append(selected)

    if errors:
        preview = "; ".join(errors[:5])
        suffix = f"; plus {len(errors) - 5} more" if len(errors) > 5 else ""
        raise ValueError(f"data_package paired-row validation failed: {preview}{suffix}")

    max_clips = int(input_data.get("max_clips", 0) or 0)
    if max_clips > 0:
        selected_rows = selected_before_max[:max_clips]
    else:
        selected_rows = selected_before_max
    summary = data_package_manifest_summary(
        cfg,
        indicator,
        selected_rows,
        candidate_row_count=len(rows),
        matched_row_count=len(selected_before_max),
        max_clips=max_clips,
    )
    return selected_rows, summary


def data_package_manifest_summary(
    cfg: Mapping[str, Any],
    indicator: PackageIndicator,
    selected_rows: Sequence[Mapping[str, Any]],
    *,
    candidate_row_count: int | None = None,
    matched_row_count: int | None = None,
    max_clips: int = 0,
) -> dict[str, Any]:
    selected_row_count = len(selected_rows)
    indicator_row_count = len(indicator.rows)
    matched = indicator_row_count if matched_row_count is None else int(matched_row_count)
    rejected_row_count = max(0, matched - selected_row_count)
    return {
        "schema": indicator.schema,
        "spec": str(cfg["spec"]),
        "category": str(cfg["category"]),
        "indicator": str(indicator.path),
        "indicator_sha256": indicator.indicator_sha256,
        "indicator_row_count": indicator_row_count,
        "candidate_row_count": candidate_row_count,
        "matched_row_count": matched,
        "selected_row_count": selected_row_count,
        "missing_row_count": 0,
        "rejected_row_count": rejected_row_count,
        "max_clips": int(max_clips),
        "max_clips_applied": bool(max_clips > 0 and matched > selected_row_count),
        "package_rows_sha256": package_rows_sha256(selected_rows),
    }


def manifest_summary_from_selected_rows(
    input_data: Mapping[str, Any],
    selected_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    cfg = data_package_config(input_data)
    if cfg is None:
        return None
    indicator = parse_package_indicator(cfg["indicator"])
    if indicator.spec != cfg["spec"] or indicator.category != cfg["category"]:
        raise ValueError(
            "data_package indicator identity mismatch: "
            f"config={cfg['spec']}/{cfg['category']} indicator={indicator.spec}/{indicator.category}"
        )
    max_clips = int(input_data.get("max_clips", 0) or 0)
    return data_package_manifest_summary(
        cfg,
        indicator,
        selected_rows,
        matched_row_count=len(selected_rows),
        max_clips=max_clips,
    )
