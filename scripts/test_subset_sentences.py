from __future__ import annotations

from models.transformer import words_to_sentence


def main() -> None:
    print("LSA64 Subset-10 sentence tester")
    print("Type gloss tokens separated by spaces (e.g., HELP GIVE WATER).")
    print("Type 'exit' to quit.\n")

    while True:
        raw = input("Tokens> ").strip()
        if not raw:
            print("Enter at least one token.\n")
            continue
        if raw.lower() in {"exit", "quit"}:
            break

        tokens = raw.upper().split()
        sentence = words_to_sentence(tokens)
        print("Sentence:", sentence or "(empty)")
        print()


if __name__ == "__main__":
    main()
