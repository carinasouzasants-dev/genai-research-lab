import json

from google import genai
from pydantic import BaseModel
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
# ESTRUTURA DA AVALIAÇÃO
# ============================================================

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


# ============================================================
# CARREGAR RESULTADOS DO EXPERIMENTO 03
# ============================================================

input_path = "results/model_comparison_03.json"

with open(input_path, "r", encoding="utf-8") as file:
    model_results = json.load(file)


model_1 = model_results[0]
model_2 = model_results[1]


# ============================================================
# CRITÉRIOS DE AVALIAÇÃO
# ============================================================

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


# ============================================================
# PROMPT DE AVALIAÇÃO
# ============================================================

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


# ============================================================
# EXECUTAR AVALIAÇÃO
# ============================================================

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=evaluation_prompt,
    config={
        "response_mime_type": "application/json",
        "response_schema": EvaluationResult,
    },
)


# ============================================================
# VALIDAR RESULTADO
# ============================================================

evaluation = EvaluationResult.model_validate_json(response.text)


# ============================================================
# CALCULAR PONTUAÇÕES
# ============================================================

criteria = [
    "clareza",
    "completude",
    "aderencia_as_instrucoes",
    "qualidade_das_informacoes",
    "aplicabilidade",
    "uso_responsavel_de_hipoteses",
]


def calculate_score(model_evaluation):
    return sum(
        getattr(model_evaluation, criterion)
        for criterion in criteria
    )


score_model_1 = calculate_score(
    evaluation.avaliacao_modelo_1
)

score_model_2 = calculate_score(
    evaluation.avaliacao_modelo_2
)


# ============================================================
# EXIBIR RESULTADO
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENTO 04 - EVALUATION")
print("=" * 70)


print(f"\nMODELO 1: {model_1['model_name']}")
print(f"Pontuação: {score_model_1}/30")

print("\nAvaliação:")
print(
    evaluation.avaliacao_modelo_1.model_dump_json(
        indent=4
    )
)


print("\n" + "-" * 70)


print(f"\nMODELO 2: {model_2['model_name']}")
print(f"Pontuação: {score_model_2}/30")

print("\nAvaliação:")
print(
    evaluation.avaliacao_modelo_2.model_dump_json(
        indent=4
    )
)


print("\n" + "=" * 70)
print("COMPARAÇÃO FINAL")
print("=" * 70)

print(
    f"\nMelhor modelo: {evaluation.melhor_modelo}"
)

print(
    "\nJustificativa:"
)

print(
    evaluation.justificativa_comparativa
)


# ============================================================
# SALVAR RESULTADO
# ============================================================

output_data = {
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
    "comparative_justification": evaluation.justificativa_comparativa,
}


output_path = "results/evaluation_04.json"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(
        output_data,
        file,
        ensure_ascii=False,
        indent=4,
    )


print("\n" + "=" * 70)
print("AVALIAÇÃO SALVA")
print("=" * 70)

print(f"\nArquivo: {output_path}")