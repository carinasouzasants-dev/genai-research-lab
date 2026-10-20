from src.gemini_client import generate_response


response = generate_response(
    "Explique em uma frase o que é Inteligência Artificial Generativa."
)

print(response)