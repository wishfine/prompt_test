#!/usr/bin/env python3
"""Merge physics raw JSONL sources before a single teacher-labeling run.

The original sampling ``difficulty`` is intentionally preserved as provenance
only.  It must never be used as a teacher label.  Every retained record is
sent through the same frozen production teacher afterwards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REQUIRED_FIELDS = {
    "question_id",
    "parent_id",
    "stem",
    "options",
    "analysis",
    "sub_questions",
    "stem_pic_url",
    "analysis_pic_url",
    "difficulty",
}
CONTENT_FIELDS = (
    "stem",
    "options",
    "analysis",
    "sub_questions",
    "stem_pic_url",
    "analysis_pic_url",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(unicodedata.normalize("NFKC", value).split())
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): normalize_value(item) for key, item in sorted(value.items())}
    return value


def content_digest(record: dict[str, Any]) -> str:
    payload = {field: normalize_value(record.get(field)) for field in CONTENT_FIELDS}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary_path = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary_path, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge raw physics sources with ID/content de-duplication before teacher labeling."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Raw JSONL input. Repeat once per source; order determines duplicate precedence.",
    )
    parser.add_argument("--output", required=True, help="Merged JSONL output path.")
    parser.add_argument("--manifest", required=True, help="JSON manifest output path.")
    parser.add_argument(
        "--quarantine",
        required=True,
        help="JSONL for duplicate/conflicting/invalid input records.",
    )
    args = parser.parse_args()

    inputs = [Path(value).expanduser().resolve() for value in args.input]
    output_path = Path(args.output).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    quarantine_path = Path(args.quarantine).expanduser().resolve()
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Raw input does not exist: {path}")

    accepted: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    seen_ids: dict[str, tuple[str, str]] = {}
    seen_content: dict[str, str] = {}
    per_source: dict[str, Counter[str]] = defaultdict(Counter)

    for source_index, path in enumerate(inputs):
        source_key = str(path)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                per_source[source_key]["read"] += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    quarantine.append(
                        {
                            "reason": "invalid_json",
                            "source_file": source_key,
                            "source_line": line_number,
                            "error": str(exc),
                        }
                    )
                    per_source[source_key]["invalid_json"] += 1
                    continue
                if not isinstance(record, dict):
                    quarantine.append(
                        {
                            "reason": "record_not_object",
                            "source_file": source_key,
                            "source_line": line_number,
                        }
                    )
                    per_source[source_key]["record_not_object"] += 1
                    continue
                missing = sorted(REQUIRED_FIELDS - set(record))
                if missing:
                    quarantine.append(
                        {
                            "reason": "missing_required_fields",
                            "source_file": source_key,
                            "source_line": line_number,
                            "question_id": record.get("question_id"),
                            "missing_fields": missing,
                        }
                    )
                    per_source[source_key]["missing_required_fields"] += 1
                    continue

                question_id = str(record["question_id"]).strip()
                parent_id = str(record["parent_id"]).strip()
                if not question_id or not parent_id or not isinstance(record["sub_questions"], list):
                    quarantine.append(
                        {
                            "reason": "invalid_identifier_or_sub_questions",
                            "source_file": source_key,
                            "source_line": line_number,
                            "question_id": record.get("question_id"),
                        }
                    )
                    per_source[source_key]["invalid_identifier_or_sub_questions"] += 1
                    continue

                record = dict(record)
                record["question_id"] = question_id
                record["parent_id"] = parent_id
                digest = content_digest(record)
                provenance = {
                    "source_file": source_key,
                    "source_index": source_index,
                    "source_line": line_number,
                    "raw_question_id": question_id,
                    "raw_parent_id": parent_id,
                    "raw_difficulty": record.get("difficulty"),
                }

                if question_id in seen_ids:
                    prior_digest, prior_id = seen_ids[question_id]
                    reason = "duplicate_question_id" if digest == prior_digest else "conflicting_question_id"
                    quarantine.append(
                        {
                            "reason": reason,
                            "question_id": question_id,
                            "duplicate_of": prior_id,
                            "source_file": source_key,
                            "source_line": line_number,
                        }
                    )
                    per_source[source_key][reason] += 1
                    continue
                if digest in seen_content:
                    quarantine.append(
                        {
                            "reason": "duplicate_normalized_content",
                            "question_id": question_id,
                            "duplicate_of": seen_content[digest],
                            "source_file": source_key,
                            "source_line": line_number,
                        }
                    )
                    per_source[source_key]["duplicate_normalized_content"] += 1
                    continue

                record["_merge_provenance"] = provenance
                seen_ids[question_id] = (digest, question_id)
                seen_content[digest] = question_id
                accepted.append(record)
                per_source[source_key]["accepted"] += 1

    atomic_write_jsonl(output_path, accepted)
    atomic_write_jsonl(quarantine_path, quarantine)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "physics_raw_merge_v1",
        "inputs": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "stats": dict(per_source[str(path)]),
            }
            for path in inputs
        ],
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "quarantine": str(quarantine_path),
        "records": len(accepted),
        "quarantined_records": len(quarantine),
        "deduplication": {
            "priority": "input order; first retained record wins",
            "id_key": "question_id",
            "content_key": "sha256(NFKC/whitespace-normalized structured content)",
        },
        "raw_difficulty": {
            "preserved": True,
            "usage": "provenance only; never a teacher or student training label",
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
