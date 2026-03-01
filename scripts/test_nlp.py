import sys

from models.transformer import gloss_to_sentence

def main():
    print("Enter gloss sentences to test the model(type 'exit' to quit)")
    while True:
        gloss = input("Gloss> ")
        if gloss.lower() in ("exit", "quit"):
            break
        result = gloss_to_sentence(gloss)
        print("English:", result)


if __name__ == "__main__":
    main()