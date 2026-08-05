# -*- coding: utf-8 -*-
"""对高中化学运行结果进行独立 AI 盲标，不读取原流程的难度结论。"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

try:
    import aiohttp
    import json_repair
    from dotenv import load_dotenv
    from tqdm.asyncio import tqdm
except ImportError as exc:  # pragma: no cover - 在项目 venv 中运行
    raise RuntimeError(
        "缺少运行依赖，请安装 aiohttp、json-repair、python-dotenv、tqdm"
    ) from exc

from high_chemistry_pipeline_core import prepare_question


load_dotenv()

API_KEY = os.getenv("API_KEY", "not-needed")
BASE_URL = os.getenv("BASE_URL", "http://172.22.0.35:4466/v1").rstrip("/") + "/"
MODEL_NAME = os.getenv("BLIND_LABEL_MODEL_NAME") or os.getenv("MODEL_NAME", "doubao-seed-2.0-lite")
TEMPERATURE_RAW = os.getenv("BLIND_LABEL_TEMPERATURE", "")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_PROMPT = PROJECT_ROOT / "prompts" / "高中化学AI盲标提示词.txt"
DEFAULT_OUTPUT = OUTPUTS_ROOT / "model_runs" / "high_chemistry_ai_reference.jsonl"
DEFAULT_ERRORS = OUTPUTS_ROOT / "model_runs" / "high_chemistry_ai_reference_errors.jsonl"
LEVEL_NAMES = {1: "送分题", 2: "基础题", 3: "中等题", 4: "拔高题", 5: "压轴题"}
FILE_LOCK = asyncio.Lock()
DIFFICULTY_METADATA_RE = re.compile(
    r"送分题|基础题|中等题|拔高题|压轴题|难度[1-5]档|"
    r"正确率|得分率|区分度|难度(?:较|很|偏|较为|等级|档位)"
)


def resolve_temperature() -> float | None:
    if TEMPERATURE_RAW.strip():
        return float(TEMPERATURE_RAW)
    return 1.0 if "lite" in MODEL_NAME.lower() else None


TEMPERATURE = resolve_temperature()


def load_prompt(path: str | Path) -> str:
    prompt_path = Path(path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"找不到盲标 Prompt：{prompt_path}")
    namespace: dict[str, Any] = {}
    source = prompt_path.read_text(encoding="utf-8")
    exec(compile(source, str(prompt_path), "exec"), namespace)
    prompt = namespace.get("BLIND_LABEL_PROMPT")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt 缺少 BLIND_LABEL_PROMPT")
    return prompt


def load_questions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_number} 行不是合法 JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path} 第 {line_number} 行不是对象")
            rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_run_config(args: argparse.Namespace, input_path: Path, prompt_path: Path) -> dict[str, Any]:
    return {
        "labeler_version": "high_chemistry_blind_label_v2",
        "input_path": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "prompt_path": str(prompt_path.resolve()),
        "prompt_sha256": sha256_file(prompt_path),
        "model_name": MODEL_NAME,
        "temperature": TEMPERATURE,
        "image_mode": args.image_mode,
        "timeout": args.timeout,
        "retries_total": args.retries,
    }


def run_signature(config: dict[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_processed_ids(path: Path, signature: str) -> set[str]:
    if not path.exists():
        return set()
    rows = load_questions(path)
    signatures = {str(row.get("run_signature") or "") for row in rows}
    if not signatures or "" in signatures or signatures != {signature}:
        raise ValueError(
            "既有盲标输出的运行签名缺失或与本次不一致；"
            "请使用新的 --output 文件，避免混合不同运行。"
        )
    return {str(row.get("question_id") or "") for row in rows if str(row.get("question_id") or "")}


async def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    async with FILE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _redact_difficulty_metadata(text: Any) -> tuple[str, int]:
    value = str(text or "")
    segments = re.split(r"(?<=[。！？!?\n])", value)
    retained: list[str] = []
    removed = 0
    for segment in segments:
        if DIFFICULTY_METADATA_RE.search(segment):
            removed += 1
        else:
            retained.append(segment)
    return "".join(retained).strip(), removed


def _sanitize_analysis(question: dict[str, Any]) -> int:
    removed = 0
    if "analysis" in question:
        question["analysis"], count = _redact_difficulty_metadata(question["analysis"])
        removed += count
    for subquestion in question.get("sub_questions") or []:
        if isinstance(subquestion, dict):
            removed += _sanitize_analysis(subquestion)
    return removed


def _declared_image_urls(question: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    for row in [question, *(question.get("sub_questions") or [])]:
        if not isinstance(row, dict):
            continue
        for field in ("stem_image_url", "stem_pic_url", "analysis_image_url", "analysis_pic_url"):
            url = str(row.get(field) or "").strip()
            if url and url not in urls:
                urls.append(url)
    return urls


def question_payload(source: dict[str, Any], image_mode: str) -> tuple[dict[str, Any], dict[str, Any], list[str], dict[str, Any]]:
    prepared = prepare_question(source, image_mode=image_mode)
    question = copy.deepcopy(prepared.question)
    redacted_segments = _sanitize_analysis(question)
    declared_images = _declared_image_urls(question)
    image_required_but_missing = (
        prepared.input_quality["content_mode"] == "image_dependent"
        or (
            prepared.input_quality["image_required"]
            and prepared.input_quality["image_available"]
            and not prepared.selected_image_urls
        )
    ) and not prepared.selected_image_urls
    image_metadata = {
        "declared_image_count": len(declared_images),
        "selected_image_count": len(prepared.selected_image_urls),
        "image_required": prepared.input_quality["image_required"],
        "image_required_but_missing": image_required_but_missing,
        "analysis_metadata_segments_redacted": redacted_segments,
    }
    return question, prepared.input_quality, prepared.selected_image_urls, image_metadata


def build_question_text(question: dict[str, Any], input_quality: dict[str, Any]) -> str:
    return (
        "【输入质量】\n" + json.dumps(input_quality, ensure_ascii=False, indent=2)
        + "\n\n【题目数据】\n" + json.dumps(question, ensure_ascii=False, indent=2)
    )


def content_with_images(text: str, image_urls: list[str]) -> str | list[dict[str, Any]]:
    if not image_urls:
        return text
    return [{"type": "input_text", "text": text}] + [
        {"type": "input_image", "image_url": url} for url in image_urls
    ]


def extract_output_text(body: dict[str, Any]) -> str:
    if isinstance(body.get("output_text"), str):
        return body["output_text"]
    chunks: list[str] = []
    for item in body.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                chunks.append(content["text"])
    return "\n".join(chunks)


def usage(body: dict[str, Any]) -> dict[str, int]:
    raw = body.get("usage") or {}
    return {key: int(raw.get(key, 0) or 0) for key in ("input_tokens", "output_tokens", "total_tokens")}


def validate_label(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("盲标输出必须为 JSON 对象")
    try:
        level = int(value.get("standard_level"))
    except (TypeError, ValueError) as exc:
        raise ValueError("standard_level 必须为 1—5 整数") from exc
    if level not in LEVEL_NAMES:
        raise ValueError("standard_level 必须为 1—5 整数")
    reason = str(value.get("reason") or "").strip()
    confidence = str(value.get("confidence") or "").strip()
    if not reason:
        raise ValueError("reason 不得为空")
    if confidence not in {"高", "中", "低"}:
        raise ValueError("confidence 必须为 高/中/低")
    return {"standard_level": level, "reason": reason, "confidence": confidence}


async def call_label(
    session: aiohttp.ClientSession,
    *,
    prompt: str,
    question_text: str,
    image_urls: list[str],
    timeout: int,
    retries: int,
) -> tuple[dict[str, Any], dict[str, int], float]:
    """最多尝试 retries 次；首次获得合法 JSON 后立即返回，不重新抽样。"""
    started = time.time()
    total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    repair = ""
    last_error = ""
    for attempt in range(retries):
        text = prompt + "\n\n" + question_text
        if repair:
            text += "\n\n【格式修复要求】\n" + repair + "\n只输出完整合法 JSON。"
        payload: dict[str, Any] = {
            "model": MODEL_NAME,
            "input": [{"role": "user", "content": content_with_images(text, image_urls)}],
            "thinking": {"type": "disabled"},
            "max_output_tokens": 800,
        }
        if TEMPERATURE is not None:
            payload["temperature"] = TEMPERATURE
        try:
            async with session.post(
                f"{BASE_URL}responses",
                json=payload,
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response:
                raw = await response.text()
                if response.status != 200:
                    last_error = f"HTTP {response.status}: {raw[:400]}"
                else:
                    body = json.loads(raw)
                    current = usage(body)
                    for key in total:
                        total[key] += current[key]
                    try:
                        return validate_label(json_repair.loads(extract_output_text(body))), total, time.time() - started
                    except ValueError as exc:
                        last_error = str(exc)
                        repair = f"上一次输出校验失败：{exc}"
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retries - 1:
            await asyncio.sleep(2 ** attempt + random.random())
    raise RuntimeError(f"盲标请求失败：{last_error}")


async def process_one(
    source: dict[str, Any], *, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore,
    prompt: str, output: Path, errors: Path, image_mode: str, timeout: int, retries: int,
    signature: str, config: dict[str, Any],
) -> None:
    async with semaphore:
        question, input_quality, image_urls, image_metadata = question_payload(source, image_mode)
        question_id = str(question.get("question_id") or "")
        try:
            if image_metadata["image_required_but_missing"]:
                raise ValueError("盲标所需图片未被纳入，拒绝依据残缺文本猜测")
            label, api_usage, elapsed = await call_label(
                session, prompt=prompt, question_text=build_question_text(question, input_quality),
                image_urls=image_urls, timeout=timeout, retries=retries,
            )
            await append_jsonl(output, {
                "question_id": question_id,
                "standard_level": label["standard_level"],
                "standard_level_name": LEVEL_NAMES[label["standard_level"]],
                "reason": label["reason"],
                "confidence": label["confidence"],
                "input_quality": input_quality,
                "selected_image_urls": image_urls,
                "image_metadata": image_metadata,
                "run_signature": signature,
                "run_config": config,
                "labeler_version": config["labeler_version"],
                "model_name": MODEL_NAME,
                "temperature": TEMPERATURE,
                "api_time_seconds": round(elapsed, 2),
                "api_usage": api_usage,
            })
        except Exception as exc:
            await append_jsonl(errors, {
                "question_id": question_id,
                "error_type": type(exc).__name__, "error": str(exc),
                "input_quality": input_quality,
                "image_metadata": image_metadata,
                "run_signature": signature,
                "run_config": config,
            })


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="高中化学题目独立 AI 盲标")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--errors", default=str(DEFAULT_ERRORS))
    parser.add_argument("--prompt", default=str(DEFAULT_PROMPT))
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--image-mode", choices=("off", "auto", "all"), default="auto")
    return parser


async def run(args: argparse.Namespace) -> None:
    input_path, output_path, error_path = Path(args.input), Path(args.output), Path(args.errors)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在：{input_path}")
    prompt = load_prompt(args.prompt)
    prompt_path = Path(args.prompt)
    config = build_run_config(args, input_path, prompt_path)
    signature = run_signature(config)
    rows = load_questions(input_path)
    processed = load_processed_ids(output_path, signature)
    pending = [row for row in rows if str(row.get("question_id") or "") not in processed]
    print(f"盲标运行签名：{signature}")
    print(f"输入 SHA256：{config['input_sha256']}")
    print(f"Prompt SHA256：{config['prompt_sha256']}")
    print(f"模型：{MODEL_NAME}；temperature：{TEMPERATURE}；总尝试次数：{args.retries}")
    print(f"加载题目：{len(rows)}；已盲标：{len(processed)}；待盲标：{len(pending)}")
    if not pending:
        return
    semaphore = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=max(2, args.concurrency * 2))
    async with aiohttp.ClientSession(connector=connector) as session:
        progress = tqdm(total=len(pending), desc="High Chemistry Blind Label", unit="item")
        tasks = [asyncio.create_task(process_one(
            row, session=session, semaphore=semaphore, prompt=prompt, output=output_path,
            errors=error_path, image_mode=args.image_mode, timeout=args.timeout, retries=args.retries,
            signature=signature, config=config,
        )) for row in pending]
        for task in asyncio.as_completed(tasks):
            await task
            progress.update(1)
        progress.close()
    print(f"盲标结果：{output_path.resolve()}")
    print(f"盲标错误：{error_path.resolve()}")


def main() -> None:
    args = build_parser().parse_args()
    started = time.time()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("收到中断信号，已安全退出")
    finally:
        print(f"耗时：{(time.time() - started) / 60:.2f}分钟")


if __name__ == "__main__":
    main()
