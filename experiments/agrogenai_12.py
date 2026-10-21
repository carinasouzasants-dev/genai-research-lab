import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel


# ============================================================
# CONFIGURAÇÃO
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY não encontrada. "
        "Verifique o arquivo .env."
    )

client = genai.Client(api_key=API_KEY)

GENERATION_MODEL = "gemini-3.8-flash"
EMBEDDING_MODEL = "gemini-embedding-2"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "agricultural_knowledge.txt"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

JSON_OUTPUT_PATH = RESULTS_DIR / "agrogenai_12.json"


# ============================================================
# MODELO DE SAÍDA
# ============================================================

class AgroGenAIAnalysis(BaseModel):
    categoria: str
    prioridade: str
    resumo_executivo: str
    fatores_identificados: list[str]
    evidencias_da_base: list[str]
    recomendacoes: list[str]
    riscos_e_limitacoes: list[str]
    requer_analise_humana: bool


# ============================================================
# CARREGAMENTO DA BASE DE CONHECIMENTO
# ============================================================

def load_knowledge_base() -> list[str]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Base de conhecimento não encontrada: {DATA_PATH}"
        )

    text = DATA_PATH.read_text(encoding="utf-8")

    chunks = [
        chunk.strip()
        for chunk in re.split(r"\n\s*\n", text)
        if chunk.strip()
    ]

    return chunks


# ============================================================
# EMBEDDINGS
# ============================================================

def generate_embedding(text: str) -> list[float]:
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )

    return response.embeddings[0].values


# ============================================================
# SIMILARIDADE DE COSSENO
# ============================================================

def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:

    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    norm_a = sum(a * a for a in vector_a) ** 0.5
    norm_b = sum(b * b for b in vector_b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


# ============================================================
# BUSCA SEMÂNTICA
# ============================================================

def semantic_search(
    query: str,
    documents: list[str],
    top_k: int = 3,
):
    query_embedding = generate_embedding(query)

    scored_documents = []

    for document in documents:
        document_embedding = generate_embedding(document)

        similarity = cosine_similarity(
            query_embedding,
            document_embedding,
        )

        scored_documents.append(
            {
                "document": document,
                "similarity": similarity,
            }
        )

    scored_documents.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )

    return scored_documents[:top_k]


# ============================================================
# CENÁRIO AGRÍCOLA
# ============================================================

agricultural_report = """
Relatório experimental de uma área de produção de soja.

Foi observada redução da produtividade estimada em relação ao histórico
da área. O período também apresentou irregularidade na distribuição
das chuvas e sinais de possível déficit hídrico.

As temperaturas médias ficaram acima do padrão observado anteriormente.

Também existem indícios de alterações nas condições do solo, porém os
dados disponíveis ainda não são suficientes para determinar a causa
principal da queda de produtividade.

Para uma análise mais confiável, devem ser considerados dados históricos
de produtividade, umidade e características do solo, precipitação,
temperatura, manejo da cultura e ocorrência de pragas ou doenças.

Este relatório é experimental e não representa um diagnóstico agrícola.
"""


# ============================================================
# BUSCA DE CONTEXTO
# ============================================================

print("\n" + "=" * 70)
print("AGROGENAI | ANÁLISE INTELIGENTE DE CENÁRIO AGRÍCOLA")
print("=" * 70)

print("\nCarregando base de conhecimento...")

knowledge_base = load_knowledge_base()

print(f"Documentos encontrados: {len(knowledge_base)}")

print("\nExecutando busca semântica...")

retrieved_documents = semantic_search(
    agricultural_report,
    knowledge_base,
    top_k=3,
)

print("\nDocumentos mais relevantes:")

for index, item in enumerate(retrieved_documents, start=1):
    print(
        f"\n[{index}] Similaridade: "
        f"{item['similarity']:.4f}"
    )

    print(item["document"][:300])


# ============================================================
# CONSTRUÇÃO DO CONTEXTO
# ============================================================

context = "\n\n".join(
    [
        (
            f"EVIDÊNCIA {index} "
            f"(similaridade {item['similarity']:.4f}):\n"
            f"{item['document']}"
        )
        for index, item in enumerate(
            retrieved_documents,
            start=1,
        )
    ]
)


# ============================================================
# PROMPT PRINCIPAL
# ============================================================

prompt = f"""
Você é um assistente experimental de inteligência artificial
voltado para análise de informações agrícolas.

Seu objetivo é analisar um relatório agrícola utilizando somente
as evidências fornecidas e produzir uma análise estruturada.

IMPORTANTE:

- Não invente dados.
- Não apresente hipóteses como fatos.
- Diferencie evidência de hipótese.
- Não produza diagnóstico definitivo.
- Recomendações devem ser consideradas sugestões para investigação.
- Sempre indique limitações dos dados.
- Determine se a análise humana é necessária.
- O resultado será usado apenas como protótipo experimental.

RELATÓRIO AGRÍCOLA:

{agricultural_report}

BASE DE CONHECIMENTO RECUPERADA POR BUSCA SEMÂNTICA:

{context}

Produza uma análise contendo:

1. Categoria do problema.
2. Prioridade.
3. Resumo executivo.
4. Fatores identificados.
5. Evidências utilizadas da base.
6. Recomendações para investigação.
7. Riscos e limitações.
8. Indicação explícita se análise humana é necessária.

A resposta deve seguir exatamente o schema fornecido.
"""


# ============================================================
# GERAÇÃO ESTRUTURADA
# ============================================================

print("\nGerando análise com IA...")

response = client.models.generate_content(
    model=GENERATION_MODEL,
    contents=prompt,
    config={
        "response_mime_type": "application/json",
        "response_schema": AgroGenAIAnalysis,
    },
)

analysis = AgroGenAIAnalysis.model_validate_json(
    response.text
)


# ============================================================
# RESULTADO
# ============================================================

print("\n" + "=" * 70)
print("RESULTADO AGROGENAI")
print("=" * 70)

print(f"\nCategoria:")
print(analysis.categoria)

print(f"\nPrioridade:")
print(analysis.prioridade)

print(f"\nResumo executivo:")
print(analysis.resumo_executivo)

print("\nFatores identificados:")

for factor in analysis.fatores_identificados:
    print(f"- {factor}")

print("\nRecomendações:")

for recommendation in analysis.recomendacoes:
    print(f"- {recommendation}")

print("\nRiscos e limitações:")

for risk in analysis.riscos_e_limitacoes:
    print(f"- {risk}")

print(
    "\nRequer análise humana: "
    f"{analysis.requer_analise_humana}"
)


# ============================================================
# SALVAMENTO
# ============================================================

result = {
    "project": "AgroGenAI",
    "experiment": 12,
    "generation_model": GENERATION_MODEL,
    "embedding_model": EMBEDDING_MODEL,
    "knowledge_documents": len(knowledge_base),
    "retrieved_documents": [
        {
            "similarity": item["similarity"],
            "document": item["document"],
        }
        for item in retrieved_documents
    ],
    "agricultural_report": agricultural_report.strip(),
    "analysis": analysis.model_dump(),
}


with open(
    JSON_OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        result,
        file,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# FINALIZAÇÃO
# ============================================================

print("\n" + "=" * 70)
print("AGROGENAI CONCLUÍDO")
print("=" * 70)

print(f"\nResultado salvo em:")
print(JSON_OUTPUT_PATH)

print("\nPipeline executado:")
print(
    "Relatório agrícola"
    " → Embeddings"
    " → Busca semântica"
    " → Recuperação de contexto"
    " → LLM"
    " → Structured Output"
    " → Análise"
)

print("\n" + "=" * 70)