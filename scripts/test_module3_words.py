from __future__ import annotations

from models.transformer import words_to_sentence


ALLOWED = {"cousin", "eat", "finish", "nice", "teacher"}


def main() -> None:
    print("Module-3 NLP test")
    print("Allowed words:", ", ".join(sorted(ALLOWED)))
    print("Type words separated by spaces. Type 'exit' to quit.\n")

    while True:
        raw = input("Words> ").strip()
        if raw.lower() in {"exit", "quit"}:
            break
        if not raw:
            print("Please enter at least one word.\n")
            continue

        tokens = [t.lower() for t in raw.split()]
        unknown = [t for t in tokens if t not in ALLOWED]
        if unknown:
            print("Unknown words:", ", ".join(sorted(set(unknown))))
            print("Use only:", ", ".join(sorted(ALLOWED)), "\n")
            continue

        sentence = words_to_sentence(tokens)
        print("Sentence:", sentence, "\n")


if __name__ == "__main__":
    main()

