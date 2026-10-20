import json
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
OUTPUT_PATH = "results/rag_05.json"


# ============================================================
# CARREGAR CONHECIMENTO
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
# PERGUNTA EXPERIMENTAL
# ============================================================

question = """
Quais fatores devem ser investigados quando uma propriedade
agrícola apresenta uma queda significativa de produtividade?
"""


# ============================================================
# BUSCA SIMPLES POR PALAVRAS-CHAVE
# ============================================================

def retrieve_relevant_chunks(question, chunks, top_k=3):

    question_words = set(
        re.findall(
            r"\b[a-zá-úA-ZÁ-Ú]+\b",
            question.lower(),
        )
    )

    scored_chunks = []

    for chunk in chunks:

        chunk_words = set(
            re.findall(
                r"\b[a-zá-úA-ZÁ-Ú]+\b",
                chunk.lower(),
            )
        )

        score = len(
            question_words.intersection(chunk_words)
        )

        scored_chunks.append(
            {
                "chunk": chunk,
                "score": score,
            }
        )

    scored_chunks.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return scored_chunks[:top_k]


# ============================================================
# RECUPERAR CONTEXTO
# ============================================================

retrieved_chunks = retrieve_relevant_chunks(
    question,
    chunks,
)

context = "\n\n".join(
    item["chunk"]
    for item in retrieved_chunks
)


# ============================================================
# PROMPT RAG
# ============================================================

prompt = f"""
Você é um assistente de análise agrícola.

Responda à pergunta utilizando SOMENTE as informações
presentes no contexto fornecido.

Se o contexto não possuir informação suficiente para responder
alguma parte da pergunta, diga explicitamente que a informação
não está disponível no contexto.

Não invente dados.

CONTEXTO:

{context}


PERGUNTA:

{question}

Forneça uma resposta objetiva e organizada.
"""


# ============================================================
# GERAR RESPOSTA
# ============================================================

response = client.models.generate_content(
    model="gemini-3.7-flash",
    contents=prompt,
)

answer = response.text


# ============================================================
# EXIBIR RESULTADO
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENTO 05 - RAG")
print("=" * 70)

print("\nPERGUNTA:")
print(question)

print("\n" + "-" * 70)

print("TRECHOS RECUPERADOS:")

for index, item in enumerate(
    retrieved_chunks,
    start=1,
):
    print(f"\n[{index}] Score: {item['score']}")
    print(item["chunk"])

print("\n" + "-" * 70)

print("RESPOSTA DO MODELO:")
print(answer)


# ============================================================
# SALVAR RESULTADO
# ============================================================

result = {
    "experiment": "05 - RAG",
    "model": "gemini-3.7-flash",
    "question": question,
    "retrieved_chunks": retrieved_chunks,
    "context": context,
    "answer": answer,
}


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        result,
        file,
        ensure_ascii=False,
        indent=4,
    )


print("\n" + "=" * 70)
print("EXPERIMENTO CONCLUÍDO")
print("=" * 70)

print(f"\nResultado salvo em: {OUTPUT_PATH}")