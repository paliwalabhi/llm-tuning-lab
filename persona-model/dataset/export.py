import json
import re
from datetime import datetime, timezone
from pathlib import Path

from questions import QUESTIONS


def _word_count(text: str) -> int:
    return len(text.split()) if text else 0


def _derive_system_prompt(responses: dict) -> str:
    """Heuristic system prompt from response patterns — no LLM call needed."""
    all_text = " ".join(v for v in responses.values() if v)

    word_count = _word_count(all_text)
    sentences = re.split(r"[.!?]+", all_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    avg_len = (word_count / len(sentences)) if sentences else 15

    informal_markers = len(re.findall(r"\b(hey|yeah|yep|nope|kinda|gonna|wanna|tbh|lol|haha|btw|fyi)\b", all_text, re.I))
    formal_markers = len(re.findall(r"\b(therefore|furthermore|regarding|pursuant|hereby|sincerely|accordingly)\b", all_text, re.I))
    hedges = len(re.findall(r"\b(maybe|perhaps|might|could|probably|i think|i believe|not sure|seems like)\b", all_text, re.I))
    bullets = all_text.count("-") + all_text.count("•") + all_text.count("*")

    if informal_markers > formal_markers:
        tone = "casual and direct"
    elif formal_markers > informal_markers:
        tone = "professional and structured"
    else:
        tone = "balanced — professional when needed, casual with people they know"

    if avg_len < 12:
        sentence_style = "short, punchy sentences"
    elif avg_len > 22:
        sentence_style = "longer, more detailed sentences"
    else:
        sentence_style = "medium-length sentences"

    hedge_note = " They tend to qualify opinions rather than state them as absolutes." if hedges > 4 else ""
    format_note = " They often use lists and structured formatting." if bullets > 5 else ""

    return (
        f"You are a writer with a {tone} communication style, preferring {sentence_style}."
        f"{hedge_note}{format_note} Mirror this person's vocabulary, rhythm, and tone exactly."
    )


def build_primary_json(responses: dict) -> dict:
    answered = {k: v for k, v in responses.items() if v is not None}
    system_prompt = _derive_system_prompt(answered)

    samples = []
    for q in QUESTIONS:
        qid = str(q["id"])
        response = responses.get(qid)
        samples.append({
            "id": q["id"],
            "category": q["category"],
            "category_code": q["category_code"],
            "question": q["prompt"],
            "response": response or "",
            "word_count": _word_count(response) if response else 0,
            "has_response": response is not None,
        })

    categories = sorted({q["category_code"] for q in QUESTIONS})

    return {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "metadata": {
            "total_questions": len(QUESTIONS),
            "questions_answered": len(answered),
            "collection_tool": "collect_style.py",
            "suggested_system_prompt": system_prompt,
            "categories": categories,
        },
        "style_samples": samples,
    }


def write_primary_json(data: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def build_jsonl_records(responses: dict) -> list:
    data = build_primary_json(responses)
    system_prompt = data["metadata"]["suggested_system_prompt"]
    records = []
    for sample in data["style_samples"]:
        if not sample["has_response"]:
            continue
        records.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": sample["question"]},
                {"role": "assistant", "content": sample["response"]},
            ]
        })
    return records


def write_jsonl(records: list, output_path: Path) -> None:
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_export(responses: dict, output_dir: Path) -> tuple:
    json_path = output_dir / "style_dataset.json"
    jsonl_path = output_dir / "style_dataset.jsonl"

    data = build_primary_json(responses)
    write_primary_json(data, json_path)

    records = build_jsonl_records(responses)
    write_jsonl(records, jsonl_path)

    return json_path, jsonl_path
