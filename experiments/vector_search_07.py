import json
import math
import os
import re

import numpy as np
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
OUTPUT_PATH = "results/vector_search_07.json"


# ============================================================
# MODELO DE EMBEDDING
# ============================================================

EMBEDDING_MODEL = "gemini-embedding-2"


# ============================================================
# CONFIGURAÇÃO DA BUSCA
# ============================================================

TOP_K = 3


# ============================================================
# CARREGAR DOCUMENTO
# ============================================================

with open(
    KNOWLEDGE_PATH,
    "r",
    encoding="utf-8",
) as file:
    knowledge = file.read()


# ============================================================
# DIVIDIR DOCUMENTO EM CHUNKS
# ============================================================

chunks = [
    chunk.strip()
    for chunk in re.split(
        r"\n\s*\n",
        knowledge,
    )
    if chunk.strip()
]


# ============================================================
# PERGUNTA
# ============================================================

question = """
Quais fatores podem estar relacionados a uma queda na
produtividade agrícola?
"""


# ============================================================
# GERAR EMBEDDING
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
        a * b
        for a, b in zip(
            vector_a,
            vector_b,
        )
    )

    magnitude_a = math.sqrt(
        sum(
            value * value
            for value in vector_a
        )
    )

    magnitude_b = math.sqrt(
        sum(
            value * value
            for value in vector_b
        )
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (
        magnitude_a * magnitude_b
    )


# ============================================================
# CONSTRUIR ÍNDICE VETORIAL
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENTO 07 - VECTOR SEARCH")
print("=" * 70)

print("\nConstruindo índice vetorial...")


document_embeddings = []

for index, chunk in enumerate(
    chunks,
    start=1,
):

    print(
        f"Gerando embedding "
        f"{index}/{len(chunks)}..."
    )

    embedding = generate_embedding(chunk)

    document_embeddings.append(
        embedding
    )


# ============================================================
# CONVERTER PARA MATRIZ NUMPY
# ============================================================

embedding_matrix = np.array(
    document_embeddings
)

print(
    "\nÍndice vetorial criado."
)

print(
    f"Quantidade de documentos: "
    f"{len(chunks)}"
)

print(
    f"Dimensão dos vetores: "
    f"{embedding_matrix.shape[1]}"
)


# ============================================================
# EMBEDDING DA PERGUNTA
# ============================================================

print(
    "\nGerando embedding da pergunta..."
)

question_embedding = generate_embedding(
    question
)


# ============================================================
# BUSCA VETORIAL
# ============================================================

print(
    "\nExecutando busca vetorial..."
)

search_results = []

for index, document_vector in enumerate(
    embedding_matrix
):

    similarity = cosine_similarity(
        question_embedding,
        document_vector.tolist(),
    )

    search_results.append(
        {
            "chunk_index": index + 1,
            "chunk": chunks[index],
            "similarity": round(
                similarity,
                6,
            ),
        }
    )


# ============================================================
# ORDENAR RESULTADOS
# ============================================================

search_results.sort(
    key=lambda item: item["similarity"],
    reverse=True,
)


# ============================================================
# TOP-K
# ============================================================

top_results = search_results[:TOP_K]


# ============================================================
# EXIBIR RESULTADOS
# ============================================================

print("\n" + "-" * 70)

print("PERGUNTA:")

print(question)

print("\n" + "-" * 70)

print("TOP RESULTADOS:")

for rank, result in enumerate(
    top_results,
    start=1,
):

    print(
        f"\n[{rank}] "
        f"Chunk {result['chunk_index']}"
    )

    print(
        f"Similaridade: "
        f"{result['similarity']}"
    )

    print(
        f"\n{result['chunk']}"
    )


# ============================================================
# SALVAR RESULTADOS
# ============================================================

output = {
    "experiment": "07 - Vector Search",
    "embedding_model": EMBEDDING_MODEL,
    "question": question,
    "document_count": len(chunks),
    "embedding_dimension": int(
        embedding_matrix.shape[1]
    ),
    "top_k": TOP_K,
    "results": top_results,
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
print("EXPERIMENTO 07 CONCLUÍDO")
print("=" * 70)

print(
    f"\nResultado salvo em: "
    f"{OUTPUT_PATH}"
)