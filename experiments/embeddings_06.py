import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import EMBEDDING_MODEL, RESULTS_DIR, RETRIEVAL_QUESTION
from src.gemini_client import embed_text
from src.results import save_json
from src.retrieval import load_chunks, similarities


def main():
    chunks = load_chunks()
    question = RETRIEVAL_QUESTION

    print("\n" + "=" * 70)
    print("EXPERIMENTO 06 - EMBEDDINGS + BUSCA SEMÂNTICA")
    print("=" * 70)
    print("\nGerando embedding da pergunta...")

    question_embedding = embed_text(question)
    print(f"Dimensão do embedding: {len(question_embedding)}")
    print("\nGerando embeddings dos documentos...")

    document_embeddings = []

    for index, chunk in enumerate(chunks, start=1):
        print(f"Processando trecho {index}/{len(chunks)}...")
        document_embeddings.append(embed_text(chunk))

    scores = similarities(question_embedding, document_embeddings)
    results = [
        {"chunk": chunk, "similarity": score}
        for chunk, score in zip(chunks, scores)
    ]
    results.sort(key=lambda item: item["similarity"], reverse=True)
    top_results = results[:3]

    print("\n" + "-" * 70)
    print("PERGUNTA:")
    print(question)
    print("\n" + "-" * 70)
    print("TRECHOS MAIS RELEVANTES:")

    for index, result in enumerate(top_results, start=1):
        print(f"\n[{index}] Similaridade: {result['similarity']}")
        print(result["chunk"])

    output_path = save_json(
        RESULTS_DIR / "embeddings_06.json",
        {
            "experiment": "06 - Embeddings + Semantic Search",
            "question_group": "retrieval-05-08",
            "retrieval": "embedding",
            "embedding_model": EMBEDDING_MODEL,
            "question": question.strip(),
            "embedding_dimension": len(question_embedding),
            "top_results": top_results,
        },
    )

    print("\n" + "=" * 70)
    print("EXPERIMENTO 06 CONCLUÍDO")
    print("=" * 70)
    print(f"\nResultado salvo em: {output_path}")


if __name__ == "__main__":
    main()
