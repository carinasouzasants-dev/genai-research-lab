import json
from pathlib import Path

from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os


# ============================================================
# CONFIGURAÇÃO
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY não encontrada no arquivo .env.")

client = genai.Client(api_key=api_key)


# ============================================================
# ESTRUTURA DO RESULTADO
# ============================================================

class AgriculturalAnalysis(BaseModel):
    problema: str
    possiveis_causas: list[str]
    evidencias: list[str]
    acoes_recomendadas: list[str]


# ============================================================
# PROBLEMA ANALISADO
# ============================================================

prompt = """
Analise o seguinte problema agrícola:

Uma propriedade agrícola apresentou uma queda significativa
na produtividade durante determinado período.

Retorne a resposta seguindo exatamente a estrutura solicitada.
Não adicione informações fora da estrutura.
"""


# ============================================================
# GERAÇÃO DA RESPOSTA ESTRUTURADA
# ============================================================

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=prompt,
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=AgriculturalAnalysis,
    ),
)


# ============================================================
# RECUPERAR RESULTADO ESTRUTURADO
# ============================================================

result = response.parsed

if result is None:
    raise ValueError("O modelo não retornou um resultado estruturado.")


# ============================================================
# CONVERTER PARA DICIONÁRIO
# ============================================================

if hasattr(result, "model_dump"):
    result_dict = result.model_dump()
else:
    result_dict = result.dict()


# ============================================================
# SALVAR RESULTADO EM JSON
# ============================================================

output_path = Path(__file__).resolve().parent.parent / "results" / "structured_output_02.json"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(
        result_dict,
        file,
        ensure_ascii=False,
        indent=4,
    )


# ============================================================
# EXIBIR RESULTADO
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENTO 02: STRUCTURED OUTPUT")
print("=" * 60)

print("\nRESULTADO ESTRUTURADO:\n")

print(json.dumps(
    result_dict,
    ensure_ascii=False,
    indent=4
))

print("\n" + "=" * 60)
print(f"JSON salvo em:")
print(output_path)
print("=" * 60)