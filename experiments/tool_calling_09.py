import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google.genai import types

from src.agri_data import get_crop_productivity as lookup_crop
from src.agri_data import get_weather_data as lookup_weather
from src.config import MODEL_FLASH_36, RESULTS_DIR
from src.gemini_client import generate_content
from src.results import save_json


MODEL = MODEL_FLASH_36
tool_execution_log = []


def get_crop_productivity(crop: str, region: str) -> dict:
    """Produtividade inventada de uma cultura e uma região, como soja no sul."""
    result = lookup_crop(crop, region)
    tool_execution_log.append(
        {
            "tool": "get_crop_productivity",
            "arguments": {"crop": crop, "region": region},
            "result": result,
        }
    )

    print("\n[FERRAMENTA EXECUTADA]")
    print("get_crop_productivity")
    print(f"  crop: {crop}")
    print(f"  region: {region}")
    print(f"  result: {result}")
    return result


def get_weather_data(region: str) -> dict:
    """Chuva e temperatura inventadas de uma região, como sul ou centro-oeste."""
    result = lookup_weather(region)
    tool_execution_log.append(
        {
            "tool": "get_weather_data",
            "arguments": {"region": region},
            "result": result,
        }
    )

    print("\n[FERRAMENTA EXECUTADA]")
    print("get_weather_data")
    print(f"  region: {region}")
    print(f"  result: {result}")
    return result


tools = [get_crop_productivity, get_weather_data]
config = types.GenerateContentConfig(tools=tools)

question = """
Analise a situação da soja na região Sul.

Consulte os dados de produtividade da soja na região Sul
e também os dados climáticos da região Sul.

Depois explique se os dados disponíveis indicam algum fator
que merece investigação em relação à produtividade.

Deixe claro que os dados são experimentais e não representam
dados reais de uma propriedade agrícola.
"""


def main():
    print("\n" + "=" * 70)
    print("EXPERIMENTO 09 - TOOL CALLING")
    print("=" * 70)
    print("\nPergunta enviada ao modelo:")
    print(question)
    print("\nModelo:")
    print(MODEL)
    print("\nAguardando decisão do modelo sobre as ferramentas...")

    response = generate_content(
        question,
        model=MODEL,
        config=config,
    )
    answer = response.text

    print("\n" + "-" * 70)
    print("RESPOSTA FINAL DO MODELO")
    print("-" * 70)
    print(answer)
    print("\n" + "-" * 70)
    print("FERRAMENTAS EXECUTADAS")
    print("-" * 70)

    for index, execution in enumerate(tool_execution_log, start=1):
        print(f"\n[{index}] {execution['tool']}")
        print(f"Argumentos: {execution['arguments']}")
        print(f"Resultado: {execution['result']}")

    output_path = save_json(
        RESULTS_DIR / "tool_calling_09.json",
        {
            "experiment": "09 - Tool Calling",
            "model": MODEL,
            "question": question.strip(),
            "tools_available": [
                "get_crop_productivity",
                "get_weather_data",
            ],
            "tools_executed": tool_execution_log,
            "answer": answer,
            "data_note": (
                "Os dados utilizados pelas ferramentas são "
                "experimentais e simulados exclusivamente para "
                "este laboratório."
            ),
        },
    )

    print("\n" + "=" * 70)
    print("EXPERIMENTO 09 CONCLUÍDO")
    print("=" * 70)
    print(f"\nResultado salvo em: {output_path}")


if __name__ == "__main__":
    main()
