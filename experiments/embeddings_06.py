import json
import math
import os
import re

from dotenv import load_dotenv
from google import genai


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
# ARQUIVOS
# ============================================================

KNOWLEDGE_PATH = "data/agricultural_knowledge.txt"
OUTPUT_PATH = "results/embeddings_06.json"


# ============================================================
# MODELO DE EMBEDDING
# ============================================================

EMBEDDING_MODEL = "gemini-embedding-2"


# ============================================================
# CARREGAR DOCUMENTO
# ============================================================

with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as file:
    knowledge = file.read()


# ============================================================
# DIVIDIR DOCUMENTO EM TRECHOS
# ============================================================

chunks = [
    chunk.strip()
    for chunk in re.split(r"\n\s*\n", knowledge)
    if chunk.strip()
]


# ============================================================
# PERGUNTA
# ============================================================

question = """
O que pode explicar uma redução no rendimento de uma plantação?
"""


# ============================================================
# GERAR EMBEDDING
# ============================================================

def generate_embedding(text):
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )

    return response.embeddings[0].values


# ============================================================
# SIMILARIDADE DE COSSENO
# ============================================================

def cosine_similarity(vector_a, vector_b):

    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (
        magnitude_a * magnitude_b
    )


# ============================================================
# EMBEDDING DA PERGUNTA
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENTO 06 - EMBEDDINGS + BUSCA SEMÂNTICA")
print("=" * 70)

print("\nGerando embedding da pergunta...")

question_embedding = generate_embedding(question)


print(
    f"Dimensão do embedding: "
    f"{len(question_embedding)}"
)


# ============================================================
# EMBEDDINGS DOS TRECHOS
# ============================================================

print("\nGerando embeddings dos documentos...")

embedded_chunks = []

for index, chunk in enumerate(chunks, start=1):

    print(f"Processando trecho {index}/{len(chunks)}...")

    embedding = generate_embedding(chunk)

    embedded_chunks.append(
        {
            "chunk": chunk,
            "embedding": embedding,
        }
    )


# ============================================================
# CALCULAR SIMILARIDADE
# ============================================================

results = []

for item in embedded_chunks:

    similarity = cosine_similarity(
        question_embedding,
        item["embedding"],
    )

    results.append(
        {
            "chunk": item["chunk"],
            "similarity": round(
                similarity,
                6,
            ),
        }
    )


# ============================================================
# ORDENAR POR RELEVÂNCIA
# ============================================================

results.sort(
    key=lambda item: item["similarity"],
    reverse=True,
)


# ============================================================
# TOP 3
# ============================================================

top_results = results[:3]


# ============================================================
# EXIBIR RESULTADOS
# ============================================================

print("\n" + "-" * 70)

print("PERGUNTA:")
print(question)

print("\n" + "-" * 70)

print("TRECHOS MAIS RELEVANTES:")

for index, result in enumerate(
    top_results,
    start=1,
):

    print(
        f"\n[{index}] "
        f"Similaridade: {result['similarity']}"
    )

    print(result["chunk"])


# ============================================================
# SALVAR RESULTADO
# ============================================================

output = {
    "experiment": "06 - Embeddings + Semantic Search",
    "embedding_model": EMBEDDING_MODEL,
    "question": question,
    "embedding_dimension": len(question_embedding),
    "top_results": top_results,
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
print("EXPERIMENTO 06 CONCLUÍDO")
print("=" * 70)

print(
    f"\nResultado salvo em: {OUTPUT_PATH}"
)