#!/usr/bin/env python3
"""Batch-sort existing junior-English knowledge-point labels with an LLM.

The model is only allowed to permute the semicolon-separated items in the
source ``output`` field.  Every response is validated as a multiset before it
is written, so hallucinated, missing, or rewritten labels cannot silently
enter the result file.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import json_repair
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = PROJECT_ROOT / "prompts" / "初中英语知识点排序提示词.txt"


def split_output_labels(value: str) -> list[str]:
    """Split the source output without changing text inside a label."""
    if not isinstance(value, str):
        raise ValueError("output 必须是字符串")
    return [part.strip() for part in value.split(";") if part.strip()]


def validate_ordered_output(source_output: str, payload: object) -> list[str]:
    """Validate the model contract and return the exact ordered labels."""
    source_labels = split_output_labels(source_output)
    if not isinstance(payload, dict):
        raise ValueError("模型响应必须是 JSON 对象")
    ordered = payload.get("ordered_output")
    if not isinstance(ordered, list):
        raise ValueError("ordered_output 必须是字符串数组")
    if not all(isinstance(label, str) and label for label in ordered):
        raise ValueError("ordered_output 中每项都必须是非空字符串")
    if len(ordered) != len(source_labels):
        raise ValueError(
            f"标签数量不一致：原始 {len(source_labels)}，模型 {len(ordered)}"
        )
    if Counter(ordered) != Counter(source_labels):
        raise ValueError("模型改变了标签内容，未通过多重集合校验")
    return ordered


def parse_model_json(text: str) -> dict[str, Any]:
    """Parse strict JSON first, then repair minor formatting noise."""
    if not text or not text.strip():
        raise ValueError("模型响应为空")
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        value = json_repair.repair_json(candidate, return_objects=True)
    if not isinstance(value, dict):
        raise ValueError("模型响应不是 JSON 对象")
    return value


def _message_text(response: Any) -> str:
    message = response.choices[0].message
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict)
        )
    return str(content or "")


def _user_content(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, separators=(",", ":"))


def _call_once(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    row: dict[str, Any],
    temperature: float | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _user_content(row)},
        ],
        "response_format": {"type": "json_object"},
    }
    if temperature is not None and "lite" not in model.lower():
        kwargs["temperature"] = temperature
    if "doubao" in model.lower():
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    response = client.chat.completions.create(**kwargs)
    return parse_model_json(_message_text(response))


def sort_one_row(
    client: OpenAI,
    row: dict[str, Any],
    *,
    model: str,
    system_prompt: str,
    retries: int,
    retry_delay: float,
    temperature: float | None,
) -> dict[str, Any]:
    """Call the model, validate the permutation, and always return a row."""
    result = dict(row)
    result["original_output"] = row.get("output")
    result["sort_model"] = model
    last_error = ""
    for attempt in range(max(1, retries)):
        try:
            payload = _call_once(
                client,
                model=model,
                system_prompt=system_prompt,
                row=row,
                temperature=temperature,
            )
            ordered = validate_ordered_output(str(row.get("output") or ""), payload)
            result.update(
                {
                    "output": ";".join(ordered),
                    "ordered_output": ordered,
                    "sorted_output": ";".join(ordered),
                    "sort_status": "success",
                    "sort_error": None,
                }
            )
            return result
        except Exception as exc:  # network, parse, and contract errors are retryable
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < max(1, retries):
                time.sleep(retry_delay * (2**attempt))
    result.update(
        {
            "ordered_output": None,
            "sorted_output": None,
            "sort_status": "error",
            "sort_error": last_error or "未知错误",
        }
    )
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_number} 行不是合法 JSON：{exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"第 {line_number} 行不是 JSON 对象")
            rows.append(value)
    return rows


def _row_key(row: dict[str, Any], index: int) -> str:
    question_id = str(row.get("question_id") or row.get("id") or "").strip()
    return question_id or f"__line__{index}"


def _load_resume_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(_read_jsonl(path)):
        if row.get("sort_status") in {"success", "error"}:
            result[_row_key(row, index)] = row
    return result


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    prompt_path = Path(args.prompt).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入 JSONL：{input_path}")
    if not prompt_path.exists():
        raise FileNotFoundError(f"找不到 Prompt：{prompt_path}")

    rows = _read_jsonl(input_path)
    system_prompt = prompt_path.read_text(encoding="utf-8")
    api_key = args.api_key or os.getenv("API_KEY", "not-needed")
    base_url = (args.base_url or os.getenv("BASE_URL", "http://172.22.0.35:4466/v1")).rstrip("/")
    model = args.model or os.getenv("MODEL_NAME", "doubao-seed-2.0-lite")
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=args.timeout, max_retries=0)

    resumed = _load_resume_rows(output_path) if args.resume else {}
    results: list[dict[str, Any] | None] = [None] * len(rows)
    pending: list[tuple[int, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        previous = resumed.get(_row_key(row, index))
        if previous and previous.get("sort_status") == "success":
            results[index] = previous
        else:
            pending.append((index, row))

    print(f"输入题目：{len(rows)}；待调用：{len(pending)}；复用：{len(rows) - len(pending)}")
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = {
            executor.submit(
                sort_one_row,
                client,
                row,
                model=model,
                system_prompt=system_prompt,
                retries=args.retries,
                retry_delay=args.retry_delay,
                temperature=args.temperature,
            ): index
            for index, row in pending
        }
        for completed, future in enumerate(as_completed(futures), 1):
            index = futures[future]
            results[index] = future.result()
            status = results[index].get("sort_status") if results[index] else "error"
            print(f"[{completed}/{len(pending)}] {index + 1}/{len(rows)} {status}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in results:
            if row is not None:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    success_count = sum(row is not None and row.get("sort_status") == "success" for row in results)
    print(f"完成：成功 {success_count}，失败 {len(rows) - success_count}，输出：{output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="原始 JSONL")
    parser.add_argument("--output", required=True, help="排序结果 JSONL")
    parser.add_argument("--prompt", default=str(DEFAULT_PROMPT), help="system prompt 文件")
    parser.add_argument("--model", default="", help="默认 doubao-seed-2.0-lite，可换 deepseek-v4-flash")
    parser.add_argument("--base-url", default="", help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default="", help="API key；默认读取 API_KEY")
    parser.add_argument("--max-workers", type=int, default=20)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--temperature", type=float, default=None, help="仅非 Lite 模型显式发送")
    parser.add_argument("--resume", action="store_true", help="复用输出中已成功的题目")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
