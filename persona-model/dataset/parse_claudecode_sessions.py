"""
Parse Claude Code session JSONL transcripts into training JSONL.

Usage:
    python parse_claudecode_sessions.py <sessions_dir> [output.jsonl]

<sessions_dir> is a Claude Code project directory like:
    C:/Users/aashi/.claude/projects/c--Users-aashi-Documents-open-source-Interviewd/

Output defaults to persona-model/dataset/claude_conversations.jsonl (appends).

Only top-level *.jsonl files are processed (subagents/ directories are skipped).
Only main-chain messages (isSidechain=false) are used.
Training pairs: assistant turn -> human reply, so the model learns to
respond in the user's voice given a prompt.
"""

import json
import sys
from pathlib import Path

DATASET_DIR = Path(__file__).parent
DEFAULT_OUTPUT = DATASET_DIR / "claude_conversations.jsonl"

SYSTEM_PROMPT = (
    "You are a person with a distinct writing style. "
    "Respond naturally and authentically, matching this person's voice, "
    "tone, vocabulary, and communication patterns exactly."
)

MIN_CHARS = 30
MAX_CHARS = 8000


def extract_text_from_content(content: list) -> str:
    """Pull only text-type blocks; skip thinking, tool_use, tool_result."""
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            t = (block.get("text") or "").strip()
            if t:
                parts.append(t)
    return "\n".join(parts)


def is_usable(text: str) -> bool:
    return MIN_CHARS <= len(text) <= MAX_CHARS


def parse_session_file(path: Path) -> list:
    """Parse one session JSONL file into ordered (role, text) turns."""
    turns = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Only main chain, only user/assistant message records
        if record.get("isSidechain"):
            continue
        if record.get("type") not in ("user", "assistant"):
            continue

        message = record.get("message", {})
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue

        content = message.get("content", [])
        # content can be a plain string in some older formats
        if isinstance(content, str):
            text = content.strip()
        else:
            text = extract_text_from_content(content)

        if not text:
            continue

        turns.append((role, text))

    return turns


def turns_to_records(turns: list) -> list:
    """Convert ordered (role, text) turns into training JSONL records."""
    records = []
    for i, (role, text) in enumerate(turns):
        if role != "user":
            continue
        if not is_usable(text):
            continue

        # Find the immediately preceding assistant turn
        prev_assistant = None
        for j in range(i - 1, -1, -1):
            if turns[j][0] == "assistant":
                prev_assistant = turns[j][1]
                break

        if prev_assistant and is_usable(prev_assistant):
            user_content = prev_assistant
        else:
            user_content = "Start a new message or question on any topic."

        records.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": text},
            ]
        })
    return records


def load_existing_records(output_path: Path) -> set:
    """Return a set of assistant-content strings already in the output file."""
    seen = set()
    if not output_path.exists():
        return seen
    for line in output_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            for msg in record.get("messages", []):
                if msg.get("role") == "assistant":
                    seen.add(msg["content"])
        except (json.JSONDecodeError, KeyError):
            continue
    return seen


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_claudecode_sessions.py <sessions_dir> [output.jsonl]")
        sys.exit(1)

    sessions_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT

    if not sessions_dir.exists():
        print(f"Error: directory not found: {sessions_dir}")
        sys.exit(1)

    # Only top-level *.jsonl files — skip subagents/ subdirectories
    session_files = [
        f for f in sessions_dir.glob("*.jsonl")
        if f.is_file()
    ]

    if not session_files:
        print(f"No session JSONL files found in: {sessions_dir}")
        sys.exit(0)

    existing = load_existing_records(output_path)
    all_records = []

    for f in sorted(session_files):
        turns = parse_session_file(f)
        records = turns_to_records(turns)
        all_records.extend(records)

    # Deduplicate against existing output
    new_records = [
        r for r in all_records
        if next((m["content"] for m in r["messages"] if m["role"] == "assistant"), "") not in existing
    ]

    if not new_records:
        print(f"No new records to add (all {len(all_records)} already present).")
        return

    with output_path.open("a", encoding="utf-8") as f:
        for record in new_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    total_after = len(existing) + len(new_records)
    print(f"Processed {len(session_files)} session files")
    print(f"Found {len(all_records)} records, {len(new_records)} new (skipped {len(all_records) - len(new_records)} duplicates)")
    print(f"Output: {output_path} ({total_after} total records)")


if __name__ == "__main__":
    main()
