import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google.genai import types

from src.agri_data import get_crop_productivity, get_weather_data
from src.config import MODEL_FLASH, RESULTS_DIR
from src.gemini_client import generate_content
from src.results import save_json


MODEL = MODEL_FLASH
# Para se o modelo continuar pedindo ferramenta e não responder.
MAX_STEPS = 4

agent_log = []
step_counter = 0

available_functions = {
    "get_crop_productivity": get_crop_productivity,
    "get_weather_data": get_weather_data,
}

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
        "required": ["crop", "region"],
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
        "required": ["region"],
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

config = types.GenerateContentConfig(tools=tools)

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


def main():
    global step_counter

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
            parts=[types.Part.from_text(text=user_goal)],
        )
    ]

    final_answer = None

    for step in range(1, MAX_STEPS + 1):
        step_counter = step

        print("\n" + "-" * 70)
        print(f"ETAPA DO AGENTE: {step}")
        print("-" * 70)

        response = generate_content(
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

        if not function_calls:
            final_answer = response.text
            print("\n" + "-" * 70)
            print("AGENTE FINALIZOU A ANÁLISE")
            print("-" * 70)
            print("\nResposta final:")
            print(final_answer)
            break

        print("\nDecisão do agente:")

        for function_call in function_calls:
            print(f"\nFerramenta escolhida: {function_call.name}")
            print(f"Argumentos: {dict(function_call.args)}")

        contents.append(model_content)
        function_response_parts = []

        for function_call in function_calls:
            function_name = function_call.name
            arguments = dict(function_call.args)
            function = available_functions.get(function_name)

            if function is None:
                result = {
                    "error": (
                        f"Ferramenta '{function_name}' não encontrada."
                    )
                }
            else:
                print(f"\n[FERRAMENTA EXECUTADA] {function_name}")
                print(f"Argumentos: {arguments}")
                result = function(**arguments)
                print(f"Resultado: {result}")

            agent_log.append(
                {
                    "step": step,
                    "tool": function_name,
                    "arguments": arguments,
                    "result": result,
                }
            )
            function_response_parts.append(
                types.Part.from_function_response(
                    name=function_name,
                    response={"result": result},
                )
            )

        # O resultado da ferramenta volta como fala do usuário, para o modelo ler.
        contents.append(
            types.Content(
                role="user",
                parts=function_response_parts,
            )
        )

    if final_answer is None:
        final_answer = (
            "O agente atingiu o limite máximo de etapas "
            "sem produzir uma resposta final."
        )
        print("\n" + "-" * 70)
        print("LIMITE DE ETAPAS ATINGIDO")
        print("-" * 70)
        print(final_answer)

    output_path = save_json(
        RESULTS_DIR / "ai_agent_10.json",
        {
            "experiment": "10 - AI Agent",
            "model": MODEL,
            "goal": user_goal.strip(),
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
        },
    )

    print("\n" + "=" * 70)
    print("EXPERIMENTO 10 CONCLUÍDO")
    print("=" * 70)
    print(f"\nEtapas utilizadas: {step_counter}")
    print(f"Ferramentas executadas: {len(agent_log)}")
    print(f"\nResultado salvo em: {output_path}")


if __name__ == "__main__":
    main()
