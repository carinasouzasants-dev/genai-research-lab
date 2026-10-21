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

MODEL = "gemini-3.8-flash"

OUTPUT_PATH = "results/ai_agent_10.json"

MAX_STEPS = 4


# ============================================================
# ESTADO DO AGENTE
# ============================================================

agent_log = []

step_counter = 0


# ============================================================
# FERRAMENTA 1
# ============================================================

def get_crop_productivity(
    crop: str,
    region: str,
) -> dict:
    """
    Retorna um indicador experimental de produtividade
    agrícola para uma cultura e região.

    Os dados são simulados exclusivamente para este laboratório.
    """

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
            )
        }

    return result


# ============================================================
# FERRAMENTA 2
# ============================================================

def get_weather_data(
    region: str,
) -> dict:
    """
    Retorna indicadores climáticos experimentais
    para uma região agrícola.

    Os dados são simulados exclusivamente para este laboratório.
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

    return result


# ============================================================
# REGISTRO DAS FERRAMENTAS
# ============================================================

available_functions = {
    "get_crop_productivity": get_crop_productivity,
    "get_weather_data": get_weather_data,
}


# ============================================================
# DECLARAÇÕES DAS FERRAMENTAS
# ============================================================

crop_productivity_declaration = {
    "name": "get_crop_productivity",
    "description": (
        "Obtém um indicador experimental de produtividade "
        "agrícola para uma cultura e uma região."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "crop": {
                "type": "string",
                "description": (
                    "Nome da cultura, como soja, milho ou algodão."
                ),
            },
            "region": {
                "type": "string",
                "description": (
                    "Região agrícola, como Sul ou Centro-Oeste."
                ),
            },
        },
        "required": [
            "crop",
            "region",
        ],
    },
}


weather_declaration = {
    "name": "get_weather_data",
    "description": (
        "Obtém indicadores climáticos experimentais "
        "de uma região agrícola."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "region": {
                "type": "string",
                "description": (
                    "Região agrícola, como Sul ou Centro-Oeste."
                ),
            },
        },
        "required": [
            "region",
        ],
    },
}


tools = [
    types.Tool(
        function_declarations=[
            crop_productivity_declaration,
            weather_declaration,
        ]
    )
]


config = types.GenerateContentConfig(
    tools=tools,
)


# ============================================================
# OBJETIVO DO AGENTE
# ============================================================

user_goal = """
Você é um agente de análise agrícola.

Seu objetivo é analisar a situação da soja na região Sul
utilizando as ferramentas disponíveis.

Você deve:

1. Identificar quais informações são necessárias.
2. Utilizar as ferramentas disponíveis quando necessário.
3. Analisar os resultados obtidos.
4. Se perceber que outra ferramenta é necessária,
   faça uma nova chamada.
5. Ao final, apresente uma análise objetiva.

Investigue principalmente se os dados disponíveis indicam
algum fator que merece atenção em relação à produtividade.

IMPORTANTE:

Os dados disponíveis são experimentais e simulados
exclusivamente para este laboratório.

Não trate os dados como dados reais de uma propriedade.

Não invente informações que não estejam disponíveis.
"""


# ============================================================
# INÍCIO DO AGENTE
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENTO 10 - AI AGENT")
print("=" * 70)

print("\nObjetivo do agente:")
print(user_goal)

print("\nModelo:")
print(MODEL)

print("\nIniciando ciclo de decisão...")


contents = [
    types.Content(
        role="user",
        parts=[
            types.Part.from_text(
                text=user_goal
            )
        ],
    )
]


# ============================================================
# LOOP DO AGENTE
# ============================================================

final_answer = None

for step in range(1, MAX_STEPS + 1):

    step_counter = step

    print("\n" + "-" * 70)
    print(f"ETAPA DO AGENTE: {step}")
    print("-" * 70)

    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=config,
    )

    model_content = response.candidates[0].content

    function_calls = []

    for part in model_content.parts:

        if part.function_call:
            function_calls.append(part.function_call)

        if part.text:
            print("\nTexto produzido pelo modelo:")
            print(part.text)


    # ========================================================
    # CASO O MODELO TENHA CHEGADO À RESPOSTA FINAL
    # ========================================================

    if not function_calls:

        final_answer = response.text

        print("\n" + "-" * 70)
        print("AGENTE FINALIZOU A ANÁLISE")
        print("-" * 70)

        print("\nResposta final:")
        print(final_answer)

        break


    # ========================================================
    # REGISTRAR A DECISÃO DO MODELO
    # ========================================================

    print("\nDecisão do agente:")

    for function_call in function_calls:

        print(
            f"\nFerramenta escolhida: "
            f"{function_call.name}"
        )

        print(
            f"Argumentos: "
            f"{dict(function_call.args)}"
        )


    # ========================================================
    # ADICIONAR RESPOSTA DO MODELO AO HISTÓRICO
    # ========================================================

    contents.append(model_content)


    # ========================================================
    # EXECUTAR AS FERRAMENTAS
    # ========================================================

    function_response_parts = []

    for function_call in function_calls:

        function_name = function_call.name
        arguments = dict(function_call.args)

        function = available_functions.get(function_name)

        if function is None:

            result = {
                "error": (
                    f"Ferramenta '{function_name}' "
                    "não encontrada."
                )
            }

        else:

            print(
                f"\n[TOOL EXECUTADA] "
                f"{function_name}"
            )

            print(
                f"Argumentos: "
                f"{arguments}"
            )

            result = function(**arguments)

            print(
                f"Resultado: "
                f"{result}"
            )


        # ====================================================
        # REGISTRAR EXECUÇÃO
        # ====================================================

        execution = {
            "step": step,
            "tool": function_name,
            "arguments": arguments,
            "result": result,
        }

        agent_log.append(execution)


        # ====================================================
        # PREPARAR RESULTADO PARA O MODELO
        # ====================================================

        function_response_parts.append(
            types.Part.from_function_response(
                name=function_name,
                response={
                    "result": result,
                },
            )
        )


    # ========================================================
    # DEVOLVER RESULTADOS AO MODELO
    # ========================================================

    contents.append(
        types.Content(
            role="user",
            parts=function_response_parts,
        )
    )


# ============================================================
# LIMITE DE SEGURANÇA
# ============================================================

if final_answer is None:

    final_answer = (
        "O agente atingiu o limite máximo de etapas "
        "sem produzir uma resposta final."
    )

    print("\n" + "-" * 70)
    print("LIMITE DE ETAPAS ATINGIDO")
    print("-" * 70)

    print(final_answer)


# ============================================================
# SALVAR RESULTADO
# ============================================================

output = {
    "experiment": "10 - AI Agent",
    "model": MODEL,
    "goal": user_goal,
    "max_steps": MAX_STEPS,
    "steps_used": step_counter,
    "tools_available": [
        "get_crop_productivity",
        "get_weather_data",
    ],
    "agent_execution_log": agent_log,
    "final_answer": final_answer,
    "data_note": (
        "Os dados utilizados pelas ferramentas são "
        "experimentais e simulados exclusivamente para "
        "este laboratório. Eles não representam dados "
        "reais de uma propriedade agrícola."
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
print("EXPERIMENTO 10 CONCLUÍDO")
print("=" * 70)

print(
    f"\nEtapas utilizadas: "
    f"{step_counter}"
)

print(
    f"Ferramentas executadas: "
    f"{len(agent_log)}"
)

print(
    f"\nResultado salvo em: "
    f"{OUTPUT_PATH}"
)