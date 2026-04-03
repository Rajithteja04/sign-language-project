from __future__ import annotations

import json
import re
from pathlib import Path

from models.transformer import words_to_sentence


def _load_allowed_tokens() -> dict[str, dict[str, str]]:
    path = Path(__file__).resolve().parents[1] / "data" / "lsa64_labels.json"
    tokens: dict[str, dict[str, str]] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for key, meta in data.items():
            token = re.sub(r"[^A-Z0-9]+", "_", key.upper()).strip("_")
            if token:
                if isinstance(meta, dict):
                    phrase = meta.get("phrase", key.title())
                    role = meta.get("role", "object")
                else:
                    phrase = str(meta)
                    role = "object"
                tokens[token.lower()] = {
                    "token": token,
                    "phrase": phrase,
                    "role": role,
                }
    base = ["COUSIN", "EAT", "FINISH", "NICE", "TEACHER"]
    for item in base:
        tokens.setdefault(
            item.lower(),
            {"token": item, "phrase": item.title(), "role": "msasl"},
        )
    return tokens


ALLOWED = _load_allowed_tokens()


def main() -> None:
    print("Module-3 NLP test")
    print(
        "Type words separated by spaces. Type 'list' to view tokens, 'exit' to quit.\n"
    )

    while True:
        raw = input("Words> ").strip()
        if raw.lower() in {"exit", "quit"}:
            break
        if raw.lower() == "list":
            _print_allowed()
            continue
        if not raw:
            print("Please enter at least one word.\n")
            continue

        tokens = [re.sub(r"[^A-Z0-9]+", "_", t.upper()).strip("_") for t in raw.split()]
        unknown = [t for t in tokens if t.lower() not in ALLOWED]
        if unknown:
            print("Unknown words:", ", ".join(sorted(set(unknown))))
            print("Use only known gloss tokens. Type 'list' to view them.\n")
            continue

        sentence = words_to_sentence(tokens)
        print("Sentence:", sentence, "\n")


def _print_allowed() -> None:
    grouped: dict[str, list[str]] = {}
    for meta in ALLOWED.values():
        display = f"{meta['token']} ({meta['phrase']})"
        grouped.setdefault(meta["role"], []).append(display)
    print("Allowed tokens by role:")
    for role in sorted(grouped):
        tokens = ", ".join(sorted(grouped[role]))
        print(f"  {role.title():<10}: {tokens}")
    print()


if __name__ == "__main__":
    main()

