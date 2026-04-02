from __future__ import annotations

import json
import re
from pathlib import Path

from models.transformer import words_to_sentence


def _load_allowed_tokens() -> dict[str, str]:
    path = Path(__file__).resolve().parents[1] / "data" / "lsa64_labels.json"
    tokens: dict[str, str] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for key in data:
            token = re.sub(r"[^A-Z0-9]+", "_", key.upper()).strip("_")
            if token:
                tokens[token.lower()] = data[key]
    base = ["COUSIN", "EAT", "FINISH", "NICE", "TEACHER"]
    for item in base:
        tokens.setdefault(item.lower(), item.title())
    return tokens


ALLOWED = _load_allowed_tokens()


def main() -> None:
    print("Module-3 NLP test")
    print("Allowed words:", ", ".join(sorted(ALLOWED.values())))
    print("Type words separated by spaces. Type 'exit' to quit.\n")

    while True:
        raw = input("Words> ").strip()
        if raw.lower() in {"exit", "quit"}:
            break
        if not raw:
            print("Please enter at least one word.\n")
            continue

        tokens = [re.sub(r"[^A-Z0-9]+", "_", t.upper()).strip("_") for t in raw.split()]
        unknown = [t for t in tokens if t.lower() not in ALLOWED]
        if unknown:
            print("Unknown words:", ", ".join(sorted(set(unknown))))
            print("Use only:", ", ".join(sorted(ALLOWED.values())), "\n")
            continue

        sentence = words_to_sentence(tokens)
        print("Sentence:", sentence, "\n")


if __name__ == "__main__":
    main()

