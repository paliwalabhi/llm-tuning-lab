"""
Parse a Claude.ai conversations export into training JSONL.

Usage:
    python parse_claude_export.py <path/to/conversations.json> [output.jsonl]

Output defaults to persona-model/dataset/claude_conversations.jsonl

Training format: each record is a conversation turn where
  user   = the preceding Claude response (the prompt the persona was replying to)
  assistant = the human's actual message (the persona's voice we're training)

First messages in a conversation (no prior context) are included with a
generic system prompt so the model learns unprompted style too.
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

MIN_CHARS = 30  # skip very short messages (reactions, single words)
MAX_CHARS = 8000  # skip extremely long messages (pastes, code dumps)


def extract_text(message: dict) -> str:
    """Get clean text from a message, preferring the top-level text field."""
    text = (message.get("text") or "").strip()
    if not text:
        # fall back to concatenating text-type content blocks
        blocks = [
            block.get("text", "")
            for block in message.get("content", [])
            if block.get("type") == "text"
        ]
        text = "\n".join(b.strip() for b in blocks if b.strip())
    return text


def is_usable(text: str) -> bool:
    return MIN_CHARS <= len(text) <= MAX_CHARS


def parse_conversations(conversations: list) -> list:
    records = []

    for convo in conversations:
        messages = convo.get("chat_messages", [])
        if not messages:
            continue

        # Build a flat ordered list of (sender, text) pairs, skipping blanks
        turns = []
        for msg in messages:
            sender = msg.get("sender")
            if sender not in ("human", "assistant"):
                continue
            text = extract_text(msg)
            if not text:
                continue
            turns.append((sender, text))

        if not turns:
            continue

        for i, (sender, text) in enumerate(turns):
            if sender != "human":
                continue
            if not is_usable(text):
                continue

            # Find the immediately preceding assistant turn as context
            prev_assistant = None
            for j in range(i - 1, -1, -1):
                if turns[j][0] == "assistant":
                    prev_assistant = turns[j][1]
                    break

            if prev_assistant and is_usable(prev_assistant):
                # Paired turn: Claude said something → user replied
                records.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prev_assistant},
                        {"role": "assistant", "content": text},
                    ]
                })
            else:
                # First turn or no usable prior context — treat as unprompted
                records.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": "Start a new message or question on any topic."},
                        {"role": "assistant", "content": text},
                    ]
                })

    return records


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_claude_export.py <conversations.json> [output.jsonl]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT

    if not input_path.exists():
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    raw = json.loads(input_path.read_text(encoding="utf-8"))
    conversations = raw if isinstance(raw, list) else raw.get("conversations", [])

    records = parse_conversations(conversations)

    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Parsed {len(conversations)} conversations → {len(records)} training records")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
