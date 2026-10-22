import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import EMBEDDING_MODEL, RESULTS_DIR, RETRIEVAL_QUESTION
from src.gemini_client import embed_text
from src.results import save_json
from src.retrieval import load_chunks, similarities


TOP_K = 3


def main():
    chunks = load_chunks()
    question = RETRIEVAL_QUESTION

    print("\n" + "=" * 70)
    print("EXPERIMENTO 07 - VECTOR SEARCH")
    print("=" * 70)
    print("\nConstruindo índice vetorial...")

    document_embeddings = []

    for index, chunk in enumerate(chunks, start=1):
        print(f"Gerando embedding {index}/{len(chunks)}...")
        document_embeddings.append(embed_text(chunk))

    embedding_matrix = np.array(document_embeddings)

    print("\nÍndice vetorial criado.")
    print(f"Quantidade de documentos: {len(chunks)}")
    print(f"Dimensão dos vetores: {embedding_matrix.shape[1]}")
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

    print("\nExecutando busca vetorial...")
    print("\n" + "-" * 70)
    print("PERGUNTA:")
    print(question)
    print("\n" + "-" * 70)
    print("TOP RESULTADOS:")

    for rank, result in enumerate(top_results, start=1):
        print(f"\n[{rank}] Chunk {result['chunk_index']}")
        print(f"Similaridade: {result['similarity']}")
        print(f"\n{result['chunk']}")

    output_path = save_json(
        RESULTS_DIR / "vector_search_07.json",
        {
            "experiment": "07 - Vector Search",
            "question_group": "retrieval-05-08",
            "retrieval": "embedding-index",
            "embedding_model": EMBEDDING_MODEL,
            "question": question.strip(),
            "document_count": len(chunks),
            "embedding_dimension": int(embedding_matrix.shape[1]),
            "top_k": TOP_K,
            "results": top_results,
        },
    )

    print("\n" + "=" * 70)
    print("EXPERIMENTO 07 CONCLUÍDO")
    print("=" * 70)
    print(f"\nResultado salvo em: {output_path}")


if __name__ == "__main__":
    main()
