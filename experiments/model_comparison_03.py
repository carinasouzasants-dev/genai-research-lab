import json
import time

from google import genai
from dotenv import load_dotenv
import os


# ============================================================
# CONFIGURAÇÃO
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY não encontrada. "
        "Verifique o arquivo .env."
    )

client = genai.Client(api_key=api_key)


# ============================================================
# MODELOS
# ============================================================

MODELS = {
    "Gemini 3.8 Flash": "gemini-3.8-flash",
    "Gemini 3.7 Flash": "gemini-3.7-flash",
}


# ============================================================
# PROBLEMA EXPERIMENTAL
# ============================================================

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


# ============================================================
# EXECUÇÃO DO EXPERIMENTO
# ============================================================

def run_experiment(model_name: str, model_id: str):

    print("\n" + "=" * 70)
    print(f"MODELO: {model_name}")
    print(f"ID: {model_id}")
    print("=" * 70)

    start_time = time.perf_counter()

    response = client.models.generate_content(
        model=model_id,
        contents=PROMPT,
    )

    elapsed_time = time.perf_counter() - start_time

    text = response.text

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


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    results = []

    for model_name, model_id in MODELS.items():

        result = run_experiment(
            model_name,
            model_id,
        )

        results.append(result)


    # ========================================================
    # SALVAR RESULTADOS
    # ========================================================

    output_path = "results/model_comparison_03.json"

    with open(output_path, "w", encoding="utf-8") as file:

        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=4,
        )


    # ========================================================
    # RESUMO
    # ========================================================

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