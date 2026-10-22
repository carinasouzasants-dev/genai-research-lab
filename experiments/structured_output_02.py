import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel

from src.config import MODEL_FLASH, RESULTS_DIR
from src.gemini_client import generate_content
from src.results import save_json


class AgriculturalAnalysis(BaseModel):
    problema: str
    possiveis_causas: list[str]
    evidencias: list[str]
    acoes_recomendadas: list[str]


prompt = """
Analise o seguinte problema agrícola:

Uma propriedade agrícola apresentou uma queda significativa
na produtividade durante determinado período.

Retorne a resposta seguindo exatamente a estrutura solicitada.
Não adicione informações fora da estrutura.
"""


def main():
    response = generate_content(
        prompt,
        model=MODEL_FLASH,
        config={
            "response_mime_type": "application/json",
            "response_schema": AgriculturalAnalysis,
        },
    )

    result = response.parsed

    if result is None:
        raise ValueError(
            "O modelo não retornou um resultado estruturado."
        )

    if hasattr(result, "model_dump"):
        result_dict = result.model_dump()
    else:
        result_dict = result.dict()

    output_path = save_json(
        RESULTS_DIR / "structured_output_02.json",
        result_dict,
    )

    print("\n" + "=" * 60)
    print("EXPERIMENTO 02: STRUCTURED OUTPUT")
    print("=" * 60)
    print("\nRESULTADO ESTRUTURADO:\n")
    print(json.dumps(result_dict, ensure_ascii=False, indent=4))
    print("\n" + "=" * 60)
    print("JSON salvo em:")
    print(output_path)
    print("=" * 60)


if __name__ == "__main__":
    main()
