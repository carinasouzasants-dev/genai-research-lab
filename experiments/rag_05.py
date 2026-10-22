import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import MODEL_FLASH_37, RESULTS_DIR, RETRIEVAL_QUESTION
from src.gemini_client import generate_response
from src.results import save_json
from src.retrieval import keyword_search, load_chunks


def main():
    chunks = load_chunks()
    question = RETRIEVAL_QUESTION
    retrieved_chunks = keyword_search(question, chunks, top_k=3)
    context = "\n\n".join(item["chunk"] for item in retrieved_chunks)

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

    answer = generate_response(prompt, model=MODEL_FLASH_37)

    print("\n" + "=" * 70)
    print("EXPERIMENTO 05 - RAG")
    print("=" * 70)
    print("\nPERGUNTA:")
    print(question)
    print("\n" + "-" * 70)
    print("TRECHOS RECUPERADOS:")

    for index, item in enumerate(retrieved_chunks, start=1):
        print(f"\n[{index}] Score: {item['score']}")
        print(item["chunk"])

    print("\n" + "-" * 70)
    print("RESPOSTA DO MODELO:")
    print(answer)

    output_path = save_json(
        RESULTS_DIR / "rag_05.json",
        {
            "experiment": "05 - RAG",
            "question_group": "retrieval-05-08",
            "retrieval": "keyword",
            "model": MODEL_FLASH_37,
            "question": question.strip(),
            "retrieved_chunks": retrieved_chunks,
            "context": context,
            "answer": answer,
        },
    )

    print("\n" + "=" * 70)
    print("EXPERIMENTO CONCLUÍDO")
    print("=" * 70)
    print(f"\nResultado salvo em: {output_path}")


if __name__ == "__main__":
    main()
