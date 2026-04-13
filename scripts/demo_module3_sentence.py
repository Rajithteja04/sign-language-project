from __future__ import annotations

from models.transformer import words_to_sentence


def main() -> None:
    print("Module 3 – NLP Sentence Generator Demo")
    print("Enter gloss tokens (uppercase words). Type 'exit' to quit.\n")

    while True:
        raw = input("Tokens> ").strip()
        if not raw:
            print("Enter one or more tokens.\n")
            continue
        if raw.lower() in {"exit", "quit"}:
            break

        tokens = [tok.upper() for tok in raw.split()]
        sentence = words_to_sentence(tokens)
        print("Sentence:", sentence or "(no output)")
        print()


if __name__ == "__main__":
    main()
