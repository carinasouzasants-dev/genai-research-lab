import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RESULTS_DIR
from src.gemini_client import generate_response
from src.results import save_json


PROBLEM = """
Uma propriedade agrícola apresentou uma queda significativa
na produtividade durante determinado período.

Analise o problema e sugira possíveis causas e ações.
"""


def run_experiment(name, prompt):
    print("\n" + "=" * 60)
    print(f"EXPERIMENTO: {name}")
    print("=" * 60)

    start = time.perf_counter()
    response = generate_response(prompt)
    elapsed = time.perf_counter() - start

    print("\nRESPOSTA:")
    print(response)
    print(f"\nTEMPO DE RESPOSTA: {elapsed:.2f}s")

    return {
        "name": name,
        "prompt": prompt.strip(),
        "response": response,
        "response_time_seconds": round(elapsed, 2),
        "response_length_characters": len(response),
    }


zero_shot_prompt = f"""
Analise o seguinte problema agrícola e sugira possíveis
causas e ações.

Problema:
{PROBLEM}
"""


few_shot_prompt = f"""
Você é um analista especializado em problemas agrícolas.

Exemplo 1:

Problema:
Baixa produtividade devido à falta de água.

Análise:
A disponibilidade hídrica pode afetar diretamente
o desenvolvimento das culturas.

Ação:
Avaliar indicadores de irrigação e disponibilidade de água.


Exemplo 2:

Problema:
Queda de produtividade associada a pragas.

Análise:
A presença de pragas pode comprometer o desenvolvimento
das plantas.

Ação:
Avaliar ocorrência, distribuição e controle das pragas.


Agora analise este problema:

{PROBLEM}

Forneça:

1. Possíveis causas
2. Evidências que deveriam ser investigadas
3. Ações recomendadas
"""


def main():
    results = [
        run_experiment("Zero-shot", zero_shot_prompt),
        run_experiment("Few-shot", few_shot_prompt),
    ]

    output_path = save_json(
        RESULTS_DIR / "prompt_comparison_01.json",
        {
            "experiment": "01 - Prompt comparison",
            "results": results,
        },
    )

    print("\n" + "=" * 60)
    print("EXPERIMENTO 01 CONCLUÍDO")
    print("=" * 60)
    print(f"\nResultado salvo em: {output_path}")


if __name__ == "__main__":
    main()
