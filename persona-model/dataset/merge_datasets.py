"""
Merge multiple training JSONL files into one deduplicated dataset.

Usage:
    python merge_datasets.py <file1.jsonl> <file2.jsonl> ... [-o output.jsonl]

Output defaults to persona-model/dataset/persona_training.jsonl
"""

import json
import sys
from pathlib import Path

DATASET_DIR = Path(__file__).parent
DEFAULT_OUTPUT = DATASET_DIR / "persona_training.jsonl"


def load_jsonl(path: Path) -> list:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python merge_datasets.py <file1.jsonl> [file2.jsonl ...] [-o output.jsonl]")
        sys.exit(1)

    output_path = DEFAULT_OUTPUT
    input_paths = []
    i = 0
    while i < len(args):
        if args[i] == "-o" and i + 1 < len(args):
            output_path = Path(args[i + 1])
            i += 2
        else:
            input_paths.append(Path(args[i]))
            i += 1

    all_records = []
    for path in input_paths:
        if not path.exists():
            print(f"Warning: file not found, skipping: {path}")
            continue
        records = load_jsonl(path)
        print(f"  {path.name}: {len(records)} records")
        all_records.extend(records)

    # Deduplicate by assistant content
    seen = set()
    unique = []
    for record in all_records:
        key = next((m["content"] for m in record.get("messages", []) if m.get("role") == "assistant"), None)
        if key and key not in seen:
            seen.add(key)
            unique.append(record)

    output_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in unique) + "\n",
        encoding="utf-8",
    )

    dupes = len(all_records) - len(unique)
    print(f"\nTotal input : {len(all_records)}")
    print(f"Duplicates  : {dupes}")
    print(f"Final output: {len(unique)} records -> {output_path}")


if __name__ == "__main__":
    main()
