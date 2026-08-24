# -*- coding: utf-8 -*-
"""高中物理老师反馈题集导入工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def import_teacher_feedback_set(
    source: Path | str, target: Path | str
) -> dict[str, Any]:
    source_path = Path(source)
    target_path = Path(target)
    seen_ids = set()
    rows = []
    for line in source_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        qid = str(row.get("question_id") or "")
        if qid in seen_ids:
            raise ValueError(f"重复 question_id: {qid}")
        seen_ids.add(qid)
        rows.append(row)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    return {
        "imported_count": len(rows),
        "question_id_count": len(seen_ids),
    }
