import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import MODEL_FLASH, MODEL_FLASH_37, RESULTS_DIR
from src.gemini_client import generate_response
from src.results import save_json


MODELS = {
    "Gemini 3.8 Flash": MODEL_FLASH,
    "Gemini 3.7 Flash": MODEL_FLASH_37,
}


PROMPT = """
Você é um analista de Inteligência Artificial aplicado ao
agronegócio.

Uma propriedade agrícola apresentou uma queda significativa
na produtividade durante determinado período.

Analise o problema e produza uma resposta objetiva.

Estruture sua resposta em exatamente quatro seções:

1. Possíveis causas
2. Dados que deveriam ser investigados
3. Ações recomendadas
4. Como a Inteligência Artificial poderia apoiar a análise

Não invente dados específicos da propriedade.
Diferencie claramente hipóteses de informações que precisam
ser verificadas.
"""


def run_experiment(model_name, model_id):
    print("\n" + "=" * 70)
    print(f"MODELO: {model_name}")
    print(f"ID: {model_id}")
    print("=" * 70)

    start_time = time.perf_counter()
    text = generate_response(PROMPT, model=model_id)
    elapsed_time = time.perf_counter() - start_time

    print("\nRESPOSTA:\n")
    print(text)
    print("\n" + "-" * 70)
    print(f"TEMPO DE RESPOSTA: {elapsed_time:.2f}s")
    print(f"CARACTERES: {len(text)}")
    print("-" * 70)

    return {
        "model_name": model_name,
        "model_id": model_id,
        "response": text,
        "response_time_seconds": round(elapsed_time, 2),
        "response_length_characters": len(text),
    }


def main():
    results = [
        run_experiment(model_name, model_id)
        for model_name, model_id in MODELS.items()
    ]

    output_path = save_json(
        RESULTS_DIR / "model_comparison_03.json",
        results,
    )

    print("\n" + "=" * 70)
    print("EXPERIMENTO 03 CONCLUÍDO")
    print("=" * 70)

    for result in results:
        print(
            f"\n{result['model_name']}:"
            f"\n  Tempo: {result['response_time_seconds']}s"
            f"\n  Caracteres: {result['response_length_characters']}"
        )

    print(f"\nResultados salvos em: {output_path}")


if __name__ == "__main__":
    main()
