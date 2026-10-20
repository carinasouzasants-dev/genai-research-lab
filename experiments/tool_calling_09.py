import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


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
# ARQUIVO DE RESULTADO
# ============================================================

OUTPUT_PATH = "results/tool_calling_09.json"


# ============================================================
# MODELO
# ============================================================

MODEL = "gemini-3.6-flash"


# ============================================================
# LOG DAS FERRAMENTAS
# ============================================================

tool_execution_log = []


# ============================================================
# FERRAMENTA 1: PRODUTIVIDADE AGRÍCOLA
# ============================================================

def get_crop_productivity(
    crop: str,
    region: str,
) -> dict:
    """
    Returns an experimental agricultural productivity
    indicator for a crop and region.

    Args:
        crop: Crop name, such as soja, milho or algodao.
        region: Agricultural region.

    Returns:
        A dictionary containing the crop productivity data.
    """

    # Dados experimentais simulados para o laboratório.
    data = {
        "soja": {
            "sul": {
                "productivity": 3200,
                "unit": "kg/ha",
            },
            "centro-oeste": {
                "productivity": 3500,
                "unit": "kg/ha",
            },
        },
        "milho": {
            "sul": {
                "productivity": 6800,
                "unit": "kg/ha",
            },
            "centro-oeste": {
                "productivity": 7200,
                "unit": "kg/ha",
            },
        },
        "algodao": {
            "sul": {
                "productivity": 1500,
                "unit": "kg/ha",
            },
            "centro-oeste": {
                "productivity": 1700,
                "unit": "kg/ha",
            },
        },
    }

    crop_key = crop.lower().strip()
    region_key = region.lower().strip()

    result = data.get(crop_key, {}).get(region_key)

    if result is None:
        result = {
            "productivity": None,
            "unit": None,
            "message": (
                "Não existem dados experimentais "
                "para essa combinação."
            ),
        }

    tool_execution_log.append(
        {
            "tool": "get_crop_productivity",
            "arguments": {
                "crop": crop,
                "region": region,
            },
            "result": result,
        }
    )

    print("\n[TOOL EXECUTADA]")
    print("get_crop_productivity")
    print(f"  crop: {crop}")
    print(f"  region: {region}")
    print(f"  result: {result}")

    return result


# ============================================================
# FERRAMENTA 2: DADOS CLIMÁTICOS
# ============================================================

def get_weather_data(
    region: str,
) -> dict:
    """
    Returns experimental weather indicators for an
    agricultural region.

    Args:
        region: Agricultural region.

    Returns:
        A dictionary containing experimental weather data.
    """

    data = {
        "sul": {
            "rainfall_mm": 820,
            "average_temperature_c": 22.4,
            "water_stress": "moderado",
        },
        "centro-oeste": {
            "rainfall_mm": 760,
            "average_temperature_c": 24.8,
            "water_stress": "alto",
        },
    }

    region_key = region.lower().strip()

    result = data.get(region_key)

    if result is None:
        result = {
            "rainfall_mm": None,
            "average_temperature_c": None,
            "water_stress": "dados indisponíveis",
        }

    tool_execution_log.append(
        {
            "tool": "get_weather_data",
            "arguments": {
                "region": region,
            },
            "result": result,
        }
    )

    print("\n[TOOL EXECUTADA]")
    print("get_weather_data")
    print(f"  region: {region}")
    print(f"  result: {result}")

    return result


# ============================================================
# CONFIGURAÇÃO DAS TOOLS
# ============================================================

tools = [
    get_crop_productivity,
    get_weather_data,
]


config = types.GenerateContentConfig(
    tools=tools,
)


# ============================================================
# PERGUNTA
# ============================================================

question = """
Analise a situação da soja na região Sul.

Consulte os dados de produtividade da soja na região Sul
e também os dados climáticos da região Sul.

Depois explique se os dados disponíveis indicam algum fator
que merece investigação em relação à produtividade.

Deixe claro que os dados são experimentais e não representam
dados reais de uma propriedade agrícola.
"""


# ============================================================
# EXECUÇÃO
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENTO 09 - TOOL CALLING")
print("=" * 70)

print("\nPergunta enviada ao modelo:")
print(question)

print("\nModelo:")
print(MODEL)

print("\nAguardando decisão do modelo sobre as ferramentas...")


response = client.models.generate_content(
    model=MODEL,
    contents=question,
    config=config,
)


# ============================================================
# RESPOSTA FINAL
# ============================================================

answer = response.text


print("\n" + "-" * 70)
print("RESPOSTA FINAL DO MODELO")
print("-" * 70)

print(answer)


# ============================================================
# RESUMO DAS FERRAMENTAS
# ============================================================

print("\n" + "-" * 70)
print("FERRAMENTAS EXECUTADAS")
print("-" * 70)

for index, execution in enumerate(
    tool_execution_log,
    start=1,
):
    print(
        f"\n[{index}] "
        f"{execution['tool']}"
    )

    print(
        f"Argumentos: "
        f"{execution['arguments']}"
    )

    print(
        f"Resultado: "
        f"{execution['result']}"
    )


# ============================================================
# SALVAR RESULTADO
# ============================================================

output = {
    "experiment": "09 - Tool Calling",
    "model": MODEL,
    "question": question,
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
}


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        output,
        file,
        ensure_ascii=False,
        indent=4,
    )


# ============================================================
# FINALIZAÇÃO
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENTO 09 CONCLUÍDO")
print("=" * 70)

print(
    f"\nResultado salvo em: "
    f"{OUTPUT_PATH}"
)