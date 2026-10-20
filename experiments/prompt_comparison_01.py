import time

from src.gemini_client import generate_response


PROBLEM = """
Uma propriedade agrícola apresentou uma queda significativa
na produtividade durante determinado período.

Analise o problema e sugira possíveis causas e ações.
"""


def run_experiment(name: str, prompt: str):
    print("\n" + "=" * 60)
    print(f"EXPERIMENTO: {name}")
    print("=" * 60)

    start = time.perf_counter()

    response = generate_response(prompt)

    elapsed = time.perf_counter() - start

    print("\nRESPOSTA:")
    print(response)

    print(f"\nTEMPO DE RESPOSTA: {elapsed:.2f}s")

    return response, elapsed


# Experimento 1: Zero-shot
zero_shot_prompt = f"""
Analise o seguinte problema agrícola e sugira possíveis
causas e ações.

Problema:
{PROBLEM}
"""


# Experimento 2: Few-shot
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


if __name__ == "__main__":

    zero_shot_response, zero_shot_time = run_experiment(
        "Zero-shot",
        zero_shot_prompt,
    )

    few_shot_response, few_shot_time = run_experiment(
        "Few-shot",
        few_shot_prompt,
    )