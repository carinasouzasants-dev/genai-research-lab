import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel

from src.config import MODEL_FLASH, RESULTS_DIR
from src.gemini_client import generate_content
from src.results import save_json


MODEL = MODEL_FLASH


class AgriculturalReport(BaseModel):
    categoria: str
    prioridade: str
    resumo: str
    fatores_identificados: list[str]
    acao_recomendada: str
    requer_analise_humana: bool


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


def main():
    print("\n" + "=" * 70)
    print("EXPERIMENTO 11 - AI AUTOMATION")
    print("=" * 70)
    print("\nModelo:")
    print(MODEL)
    print("\nProcessando relatório...")

    response = generate_content(
        prompt,
        model=MODEL,
        config={
            "response_mime_type": "application/json",
            "response_schema": AgriculturalReport,
        },
    )
    analysis = AgriculturalReport.model_validate_json(response.text)

    print("\n" + "-" * 70)
    print("RESULTADO DA AUTOMAÇÃO")
    print("-" * 70)
    print(f"\nCategoria: {analysis.categoria}")
    print(f"Prioridade: {analysis.prioridade}")
    print(f"\nResumo:\n{analysis.resumo}")
    print("\nFatores identificados:")

    for factor in analysis.fatores_identificados:
        print(f"- {factor}")

    print(f"\nAção recomendada:\n{analysis.acao_recomendada}")
    print(f"\nRequer análise humana: {analysis.requer_analise_humana}")

    record = {
        "category": analysis.categoria,
        "priority": analysis.prioridade,
        "summary": analysis.resumo,
        "identified_factors": analysis.fatores_identificados,
        "recommended_action": analysis.acao_recomendada,
        "requires_human_review": analysis.requer_analise_humana,
    }

    json_output = save_json(
        RESULTS_DIR / "ai_automation_11.json",
        {
            "experiment": "11 - AI Automation",
            "model": MODEL,
            "input_type": "agricultural_report",
            **record,
            "data_note": (
                "Os dados utilizados neste experimento são "
                "experimentais e simulados exclusivamente para "
                "este laboratório."
            ),
        },
    )

    csv_output = RESULTS_DIR / "ai_automation_11.csv"
    csv_output.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_output, "w", encoding="utf-8", newline="") as file:
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
                **record,
                "identified_factors": " | ".join(
                    analysis.fatores_identificados
                ),
            }
        )

    print("\n" + "=" * 70)
    print("AUTOMAÇÃO CONCLUÍDA")
    print("=" * 70)
    print(f"\nJSON salvo em:\n{json_output}")
    print(f"\nCSV salvo em:\n{csv_output}")
    print(
        "\nFluxo executado:"
        "\nRelatório → IA → Classificação → "
        "Dados estruturados → JSON + CSV"
    )


if __name__ == "__main__":
    main()
