import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gemini_client import generate_response


def main():
    # Uma chamada curta para ver se a chave e o cliente funcionam.
    response = generate_response(
        "Explique em uma frase o que é Inteligência Artificial Generativa."
    )
    print(response)


if __name__ == "__main__":
    main()
