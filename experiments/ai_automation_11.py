import csv
import json
import os

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel


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

MODEL = "gemini-3.8-flash"

JSON_OUTPUT_PATH = "results/ai_automation_11.json"
CSV_OUTPUT_PATH = "results/ai_automation_11.csv"


# ============================================================
# MODELO DE SAÍDA
# ============================================================

class AgriculturalReport(BaseModel):
    categoria: str
    prioridade: str
    resumo: str
    fatores_identificados: list[str]
    acao_recomendada: str
    requer_analise_humana: bool


# ============================================================
# DOCUMENTO DE ENTRADA
# ============================================================

agricultural_report = """
RELATÓRIO EXPERIMENTAL DE CAMPO

Uma propriedade agrícola apresentou queda de produtividade
na cultura da soja durante determinado período.

Os registros disponíveis indicam produtividade abaixo do
valor observado anteriormente.

Durante o período analisado foram registrados:

- precipitação irregular;
- períodos de déficit hídrico;
- temperatura elevada em alguns dias;
- possíveis alterações nas condições do solo.

Não existem, neste relatório, dados suficientes para confirmar
a causa da queda de produtividade.

É necessário analisar dados históricos, informações de solo,
clima, manejo e ocorrência de pragas antes de tomar uma
decisão operacional.
"""


# ============================================================
# PROMPT DE AUTOMAÇÃO
# ============================================================

prompt = f"""
Você é um sistema de automação para triagem de relatórios
agrícolas.

Sua tarefa é analisar o relatório recebido e transformar as
informações em dados estruturados para um fluxo automatizado.

Classifique o relatório considerando:

- categoria do problema;
- prioridade;
- resumo objetivo;
- fatores identificados;
- ação recomendada;
- necessidade de análise humana.

IMPORTANTE:

Os dados são experimentais e foram criados exclusivamente
para este laboratório.

Não trate as informações como dados reais de uma propriedade.

Não invente causas que não estejam presentes no relatório.

Se o relatório não tiver dados suficientes para determinar
uma causa, deixe isso explícito.

Relatório:

{agricultural_report}
"""


# ============================================================
# EXECUÇÃO DA IA
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENTO 11 - AI AUTOMATION")
print("=" * 70)

print("\nModelo:")
print(MODEL)

print("\nProcessando relatório...")


response = client.models.generate_content(
    model=MODEL,
    contents=prompt,
    config={
        "response_mime_type": "application/json",
        "response_schema": AgriculturalReport,
    },
)


# ============================================================
# VALIDAR SAÍDA ESTRUTURADA
# ============================================================

analysis = AgriculturalReport.model_validate_json(
    response.text
)


# ============================================================
# EXIBIR RESULTADO
# ============================================================

print("\n" + "-" * 70)
print("RESULTADO DA AUTOMAÇÃO")
print("-" * 70)

print(
    "\nCategoria:"
    f" {analysis.categoria}"
)

print(
    "Prioridade:"
    f" {analysis.prioridade}"
)

print(
    "\nResumo:"
    f"\n{analysis.resumo}"
)

print("\nFatores identificados:")

for factor in analysis.fatores_identificados:
    print(f"- {factor}")

print(
    "\nAção recomendada:"
    f"\n{analysis.acao_recomendada}"
)

print(
    "\nRequer análise humana:"
    f" {analysis.requer_analise_humana}"
)


# ============================================================
# PREPARAR DADOS PARA AUTOMAÇÃO
# ============================================================

automated_record = {
    "experiment": "11 - AI Automation",
    "model": MODEL,
    "input_type": "agricultural_report",
    "category": analysis.categoria,
    "priority": analysis.prioridade,
    "summary": analysis.resumo,
    "identified_factors": analysis.fatores_identificados,
    "recommended_action": analysis.acao_recomendada,
    "requires_human_review": analysis.requer_analise_humana,
    "data_note": (
        "Os dados utilizados neste experimento são "
        "experimentais e simulados exclusivamente para "
        "este laboratório."
    ),
}


# ============================================================
# SALVAR JSON
# ============================================================

with open(
    JSON_OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        automated_record,
        file,
        ensure_ascii=False,
        indent=4,
    )


# ============================================================
# SALVAR CSV
# ============================================================

with open(
    CSV_OUTPUT_PATH,
    "w",
    encoding="utf-8",
    newline="",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "category",
            "priority",
            "summary",
            "identified_factors",
            "recommended_action",
            "requires_human_review",
        ],
    )

    writer.writeheader()

    writer.writerow(
        {
            "category": analysis.categoria,
            "priority": analysis.prioridade,
            "summary": analysis.resumo,
            "identified_factors": " | ".join(
                analysis.fatores_identificados
            ),
            "recommended_action": analysis.acao_recomendada,
            "requires_human_review": (
                analysis.requer_analise_humana
            ),
        }
    )


# ============================================================
# FINALIZAÇÃO
# ============================================================

print("\n" + "=" * 70)
print("AUTOMAÇÃO CONCLUÍDA")
print("=" * 70)

print(
    f"\nJSON salvo em:"
    f"\n{JSON_OUTPUT_PATH}"
)

print(
    f"\nCSV salvo em:"
    f"\n{CSV_OUTPUT_PATH}"
)

print(
    "\nFluxo executado:"
    "\nRelatório → IA → Classificação → "
    "Dados estruturados → JSON + CSV"
)