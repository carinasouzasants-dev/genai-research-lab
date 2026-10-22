import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel

from src.config import MODEL_FLASH, RESULTS_DIR
from src.gemini_client import generate_content
from src.results import save_json


class ModelEvaluation(BaseModel):
    clareza: int
    completude: int
    aderencia_as_instrucoes: int
    qualidade_das_informacoes: int
    aplicabilidade: int
    uso_responsavel_de_hipoteses: int
    justificativa: str


class EvaluationResult(BaseModel):
    avaliacao_modelo_1: ModelEvaluation
    avaliacao_modelo_2: ModelEvaluation
    melhor_modelo: str
    justificativa_comparativa: str


CRITERIA = [
    "clareza",
    "completude",
    "aderencia_as_instrucoes",
    "qualidade_das_informacoes",
    "aplicabilidade",
    "uso_responsavel_de_hipoteses",
]


RUBRIC = """
Avalie cada resposta utilizando uma escala de 1 a 5.

1 = muito fraco
2 = fraco
3 = adequado
4 = bom
5 = excelente

Critérios:

1. Clareza
A resposta é clara, organizada e fácil de compreender?

2. Completude
A resposta aborda adequadamente todos os pontos solicitados?

3. Aderência às instruções
A resposta segue as instruções do prompt?

4. Qualidade das informações
As informações apresentadas são relevantes, coerentes e tecnicamente plausíveis?

5. Aplicabilidade
As sugestões poderiam apoiar uma investigação ou tomada de decisão real?

6. Uso responsável de hipóteses
A resposta diferencia hipóteses de informações que precisam ser verificadas
e evita apresentar dados inventados como fatos?

Não avalie qual modelo você prefere.
Avalie exclusivamente as respostas apresentadas.
"""


def calculate_score(model_evaluation):
    total = 0

    for criterion in CRITERIA:
        value = getattr(model_evaluation, criterion)
        # A rubrica só aceita de 1 a 5. Fora disso, a nota não vale.
        if not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError(
                f"Nota fora da escala 1-5 em {criterion}: {value}"
            )
        total += value

    return total


def main():
    input_path = RESULTS_DIR / "model_comparison_03.json"

    with open(input_path, "r", encoding="utf-8") as file:
        model_results = json.load(file)

    model_1 = model_results[0]
    model_2 = model_results[1]

    evaluation_prompt = f"""
Você é um avaliador de sistemas de Inteligência Artificial Generativa.

Seu objetivo é avaliar duas respostas produzidas por modelos diferentes
para exatamente o mesmo problema.

{RUBRIC}

------------------------------------------------------------
RESPOSTA DO MODELO 1
------------------------------------------------------------

Modelo:
{model_1["model_name"]}

Resposta:

{model_1["response"]}


------------------------------------------------------------
RESPOSTA DO MODELO 2
------------------------------------------------------------

Modelo:
{model_2["model_name"]}

Resposta:

{model_2["response"]}


------------------------------------------------------------
TAREFA
------------------------------------------------------------

Avalie individualmente cada resposta utilizando os seis critérios.

Para cada critério, forneça uma nota inteira entre 1 e 5.

Depois:

1. Compare as pontuações.
2. Identifique qual modelo apresentou a melhor resposta geral.
3. Explique brevemente os principais motivos da comparação.

A avaliação deve ser baseada somente nas respostas.
Não invente informações que não estejam presentes nelas.
"""

    response = generate_content(
        evaluation_prompt,
        model=MODEL_FLASH,
        config={
            "response_mime_type": "application/json",
            "response_schema": EvaluationResult,
        },
    )

    evaluation = EvaluationResult.model_validate_json(response.text)
    score_model_1 = calculate_score(evaluation.avaliacao_modelo_1)
    score_model_2 = calculate_score(evaluation.avaliacao_modelo_2)

    print("\n" + "=" * 70)
    print("EXPERIMENTO 04 - EVALUATION")
    print("=" * 70)
    print(f"\nMODELO 1: {model_1['model_name']}")
    print(f"Pontuação: {score_model_1}/30")
    print("\nAvaliação:")
    print(evaluation.avaliacao_modelo_1.model_dump_json(indent=4))
    print("\n" + "-" * 70)
    print(f"\nMODELO 2: {model_2['model_name']}")
    print(f"Pontuação: {score_model_2}/30")
    print("\nAvaliação:")
    print(evaluation.avaliacao_modelo_2.model_dump_json(indent=4))
    print("\n" + "=" * 70)
    print("COMPARAÇÃO FINAL")
    print("=" * 70)
    print(f"\nMelhor modelo: {evaluation.melhor_modelo}")
    print("\nJustificativa:")
    print(evaluation.justificativa_comparativa)

    output_path = save_json(
        RESULTS_DIR / "evaluation_04.json",
        {
            "experiment": "04 - Evaluation",
            "models": [
                {
                    "model_name": model_1["model_name"],
                    "score": score_model_1,
                    "evaluation": evaluation.avaliacao_modelo_1.model_dump(),
                },
                {
                    "model_name": model_2["model_name"],
                    "score": score_model_2,
                    "evaluation": evaluation.avaliacao_modelo_2.model_dump(),
                },
            ],
            "best_model": evaluation.melhor_modelo,
            "comparative_justification": (
                evaluation.justificativa_comparativa
            ),
        },
    )

    print("\n" + "=" * 70)
    print("AVALIAÇÃO SALVA")
    print("=" * 70)
    print(f"\nArquivo: {output_path}")


if __name__ == "__main__":
    main()
