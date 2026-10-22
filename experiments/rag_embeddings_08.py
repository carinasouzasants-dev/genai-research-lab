import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import (
    EMBEDDING_MODEL,
    MODEL_FLASH_36,
    RESULTS_DIR,
    RETRIEVAL_QUESTION,
)
from src.gemini_client import embed_text, generate_response
from src.results import save_json
from src.retrieval import load_chunks, similarities


TOP_K = 3
# O 3.8 respondeu 503, então esta execução usa o 3.6.
GENERATION_MODEL = MODEL_FLASH_36


def main():
    chunks = load_chunks()
    question = RETRIEVAL_QUESTION

    print("\n" + "=" * 70)
    print("EXPERIMENTO 08 - RAG COM EMBEDDINGS")
    print("=" * 70)
    print("\nConstruindo índice vetorial...")

    document_embeddings = []

    for index, chunk in enumerate(chunks, start=1):
        print(f"Gerando embedding {index}/{len(chunks)}...")
        document_embeddings.append(embed_text(chunk))

    embedding_matrix = np.array(document_embeddings)

    print("\nÍndice vetorial criado.")
    print(f"Documentos indexados: {len(chunks)}")
    print(f"Dimensão dos embeddings: {embedding_matrix.shape[1]}")
    print("\nGerando embedding da pergunta...")

    question_embedding = embed_text(question)
    scores = similarities(question_embedding, embedding_matrix)
    search_results = [
        {
            "chunk_index": index + 1,
            "chunk": chunks[index],
            "similarity": score,
        }
        for index, score in enumerate(scores)
    ]
    search_results.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )
    top_results = search_results[:TOP_K]
    context = "\n\n".join(result["chunk"] for result in top_results)

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

    print("\nGerando resposta com o modelo...")
    print(f"Modelo de geração: {GENERATION_MODEL}")

    answer = generate_response(prompt, model=GENERATION_MODEL)

    print("\n" + "-" * 70)
    print("PERGUNTA:")
    print(question)
    print("\n" + "-" * 70)
    print("TRECHOS RECUPERADOS:")

    for rank, result in enumerate(top_results, start=1):
        print(f"\n[{rank}] Similaridade: {result['similarity']}")
        print(result["chunk"])

    print("\n" + "-" * 70)
    print("RESPOSTA GERADA PELO RAG:")
    print(answer)

    output_path = save_json(
        RESULTS_DIR / "rag_embeddings_08.json",
        {
            "experiment": "08 - RAG with Embeddings",
            "question_group": "retrieval-05-08",
            "retrieval": "embedding",
            "embedding_model": EMBEDDING_MODEL,
            "generation_model": GENERATION_MODEL,
            "question": question.strip(),
            "document_count": len(chunks),
            "embedding_dimension": int(embedding_matrix.shape[1]),
            "top_k": TOP_K,
            "retrieved_chunks": top_results,
            "context": context,
            "answer": answer,
        },
    )

    print("\n" + "=" * 70)
    print("EXPERIMENTO 08 CONCLUÍDO")
    print("=" * 70)
    print(f"\nResultado salvo em: {output_path}")


if __name__ == "__main__":
    main()
