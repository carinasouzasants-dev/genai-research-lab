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
OUTPUT_PATH = "results/rag_embeddings_08.json"


# ============================================================
# MODELOS
# ============================================================

# Modelo utilizado para transformar textos em vetores.
EMBEDDING_MODEL = "gemini-embedding-2"

# Modelo utilizado para gerar a resposta final.
# Alterado para evitar o erro 503 de alta demanda.
GENERATION_MODEL = "gemini-3.6-flash"


# ============================================================
# CONFIGURAÇÃO DA BUSCA
# ============================================================

TOP_K = 3


# ============================================================
# CARREGAR CONHECIMENTO
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
Uma propriedade agrícola apresentou uma queda significativa
na produtividade. Quais fatores deveriam ser investigados
para entender as possíveis causas?
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
print("EXPERIMENTO 08 - RAG COM EMBEDDINGS")
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


embedding_matrix = np.array(
    document_embeddings
)


print(
    "\nÍndice vetorial criado."
)

print(
    f"Documentos indexados: "
    f"{len(chunks)}"
)

print(
    f"Dimensão dos embeddings: "
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
    "\nExecutando busca semântica..."
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
# RANKING
# ============================================================

search_results.sort(
    key=lambda item: item["similarity"],
    reverse=True,
)


# ============================================================
# SELECIONAR TOP-K
# ============================================================

top_results = search_results[:TOP_K]


# ============================================================
# CONSTRUIR CONTEXTO
# ============================================================

context = "\n\n".join(
    result["chunk"]
    for result in top_results
)


# ============================================================
# PROMPT RAG
# ============================================================

prompt = f"""
Você é um assistente de análise agrícola.

Sua tarefa é responder à pergunta utilizando exclusivamente
as informações presentes no contexto fornecido.

REGRAS:

1. Não invente dados.
2. Não apresente informações que não estejam no contexto
   como fatos.
3. Se o contexto não for suficiente para responder alguma
   parte da pergunta, informe essa limitação.
4. Diferencie fatores conhecidos de hipóteses que ainda
   precisam ser investigadas.
5. Produza uma resposta objetiva e organizada.

CONTEXTO:

{context}

PERGUNTA:

{question}
"""


# ============================================================
# GERAÇÃO DA RESPOSTA
# ============================================================

print(
    "\nGerando resposta com o modelo..."
)

print(
    f"Modelo de geração: {GENERATION_MODEL}"
)

response = client.models.generate_content(
    model=GENERATION_MODEL,
    contents=prompt,
)

answer = response.text


# ============================================================
# EXIBIR RESULTADOS
# ============================================================

print("\n" + "-" * 70)

print("PERGUNTA:")
print(question)

print("\n" + "-" * 70)

print("TRECHOS RECUPERADOS:")

for rank, result in enumerate(
    top_results,
    start=1,
):

    print(
        f"\n[{rank}] "
        f"Similaridade: "
        f"{result['similarity']}"
    )

    print(result["chunk"])


print("\n" + "-" * 70)

print("RESPOSTA GERADA PELO RAG:")

print(answer)


# ============================================================
# SALVAR RESULTADOS
# ============================================================

output = {
    "experiment": "08 - RAG with Embeddings",
    "embedding_model": EMBEDDING_MODEL,
    "generation_model": GENERATION_MODEL,
    "question": question,
    "document_count": len(chunks),
    "embedding_dimension": int(
        embedding_matrix.shape[1]
    ),
    "top_k": TOP_K,
    "retrieved_chunks": top_results,
    "context": context,
    "answer": answer,
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
print("EXPERIMENTO 08 CONCLUÍDO")
print("=" * 70)

print(
    f"\nResultado salvo em: "
    f"{OUTPUT_PATH}"
)