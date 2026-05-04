"""
Writing style data collection CLI.

Usage:
    python collect_style.py

Answers are saved after every question. Ctrl+C saves progress and exits cleanly.
Re-run at any time to resume where you left off.
"""

import json
import os
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

from export import run_export
from questions import QUESTIONS

DATASET_DIR = Path(__file__).parent
PROGRESS_FILE = DATASET_DIR / "responses_progress.json"
TOTAL = len(QUESTIONS)


# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------

try:
    import shutil
    _COLS = shutil.get_terminal_size((80, 20)).columns
except Exception:
    _COLS = 80

_COLS = min(_COLS, 100)


def _divider(char="="):
    print(char * _COLS)


def _wrap(text: str, indent: int = 0) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=_COLS - indent, initial_indent=prefix, subsequent_indent=prefix)


# ---------------------------------------------------------------------------
# Progress persistence
# ---------------------------------------------------------------------------

def load_progress() -> dict:
    if not PROGRESS_FILE.exists():
        return {}
    try:
        data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        return data.get("responses", {})
    except (json.JSONDecodeError, KeyError):
        backup = PROGRESS_FILE.with_suffix(".json.bak")
        print(f"\nWarning: progress file appears corrupted.")
        print(f"  [1] Start fresh (corrupted file saved to {backup.name})")
        print(f"  [2] Exit and inspect manually")
        choice = input("\nChoice [1/2]: ").strip()
        if choice == "1":
            PROGRESS_FILE.rename(backup)
            return {}
        else:
            sys.exit(0)


def save_progress(responses: dict) -> None:
    data = {
        "schema_version": "1.0",
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "responses": responses,
    }
    tmp = PROGRESS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, PROGRESS_FILE)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_welcome(already_answered: int) -> None:
    print()
    _divider()
    print(_wrap("WRITING STYLE QUESTIONNAIRE"))
    _divider()
    print()
    if already_answered > 0:
        print(_wrap(f"Resuming from question {already_answered + 1}/{TOTAL} — you've already completed {already_answered} question{'s' if already_answered != 1 else ''}."))
    else:
        print(_wrap(
            "Answer each question the way you'd naturally write or speak. "
            "There are no right answers — your raw, unedited style is the point."
        ))
        print()
        print(_wrap("  - Press Enter twice (or type \".\" on a blank line) to submit an answer."))
        print(_wrap("  - Type \"skip\" on a blank line to skip a question."))
        print(_wrap("  - Press Ctrl+C at any time to save and exit."))
    print()


def display_question(q: dict, current: int) -> None:
    print()
    _divider("-")
    print(f"  [{current}/{TOTAL}]  {q['category']}")
    _divider("-")
    print()
    print(_wrap(q["prompt"], indent=2))
    print()
    print(_wrap(q["hint"], indent=2))
    print()
    print(_wrap("Your response (Enter twice or \".\" to submit, \"skip\" to skip):", indent=2))
    print()


def display_completion(answered: int) -> None:
    print()
    _divider()
    print(_wrap(f"Done! You answered {answered}/{TOTAL} questions."))
    _divider()
    print()


def display_outputs(json_path: Path, jsonl_path: Path) -> None:
    print(_wrap("Output files:"))
    print(f"  JSON  -> {json_path}")
    print(f"  JSONL -> {jsonl_path}")
    print()
    print(_wrap("The JSONL file is ready for fine-tuning with OpenAI, HuggingFace TRL, or any framework that accepts the chat messages format."))
    print()


# ---------------------------------------------------------------------------
# Input collection
# ---------------------------------------------------------------------------

def collect_multiline_input(partial_lines: list = None) -> str | None:
    """
    Returns the submitted text, or None if skipped.
    Raises KeyboardInterrupt propagated from the caller if Ctrl+C pressed.
    """
    lines = list(partial_lines) if partial_lines else []
    blank_count = 0

    while True:
        try:
            line = input()
        except KeyboardInterrupt:
            raise
        except EOFError:
            break

        if line.strip() == ".":
            break

        if line.strip().lower() == "skip":
            return None

        if line == "":
            blank_count += 1
            if blank_count >= 2:
                break
            lines.append(line)
        else:
            blank_count = 0
            lines.append(line)

    text = "\n".join(lines).strip()

    if not text:
        print()
        print(_wrap("  Looks like that was empty. Try again, or type \"skip\" to leave it blank.", indent=2))
        print()
        return collect_multiline_input()

    return text


# ---------------------------------------------------------------------------
# Re-run menu (all questions already answered)
# ---------------------------------------------------------------------------

def already_complete_menu(responses: dict) -> dict:
    print()
    print(_wrap("You've already answered all 25 questions. What would you like to do?"))
    print()
    print("  [1] Re-export the output files")
    print("  [2] Re-answer a specific question")
    print("  [3] Start completely over")
    print("  [4] Exit")
    print()

    choice = input("Choice [1-4]: ").strip()

    if choice == "1":
        return responses

    elif choice == "2":
        print()
        qid_str = input("Enter a question number (1-25): ").strip()
        try:
            qid = int(qid_str)
            if not (1 <= qid <= TOTAL):
                raise ValueError
        except ValueError:
            print("Invalid number. Exiting.")
            sys.exit(0)
        q = next(q for q in QUESTIONS if q["id"] == qid)
        display_question(q, qid)
        try:
            response = collect_multiline_input()
        except KeyboardInterrupt:
            print("\n\nSaving and exiting...")
            save_progress(responses)
            sys.exit(0)
        responses[str(qid)] = response
        save_progress(responses)
        return responses

    elif choice == "3":
        backup = PROGRESS_FILE.with_suffix(".json.bak")
        if PROGRESS_FILE.exists():
            os.replace(PROGRESS_FILE, backup)
        return {}

    else:
        sys.exit(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    responses = load_progress()
    already_answered = sum(1 for v in responses.values() if v is not None)

    if already_answered == TOTAL:
        responses = already_complete_menu(responses)
        already_answered = sum(1 for v in responses.values() if v is not None)
        if already_answered == TOTAL and responses:
            json_path, jsonl_path = run_export(responses, DATASET_DIR)
            display_outputs(json_path, jsonl_path)
            return

    display_welcome(already_answered)

    current_num = already_answered

    for q in QUESTIONS:
        qid = str(q["id"])
        if qid in responses:
            continue

        current_num += 1
        display_question(q, current_num)

        try:
            response = collect_multiline_input()
        except KeyboardInterrupt:
            print("\n\nSaving progress and exiting...")
            save_progress(responses)
            sys.exit(0)

        responses[qid] = response
        save_progress(responses)

    answered_count = sum(1 for v in responses.values() if v is not None)
    display_completion(answered_count)

    json_path, jsonl_path = run_export(responses, DATASET_DIR)
    display_outputs(json_path, jsonl_path)


if __name__ == "__main__":
    main()
