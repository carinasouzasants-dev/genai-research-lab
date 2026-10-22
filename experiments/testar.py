import argparse
import sys
from pathlib import Path

# Rodar este arquivo direto não coloca a raiz do projeto no caminho.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import MODEL_FLASH, RESULTS_DIR, RETRIEVAL_QUESTION
from src.results import save_json
from src.retrieval import keyword_search, load_chunks, similarities


TOP_K = 3
HISTORY_PATH = RESULTS_DIR / "teste_usuario.json"


def preview(text, limit=220):
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 3].rstrip() + "..."


def show_hits(title, hits, score_key):
    print("\n" + title)
    print("-" * 70)

    if not hits:
        print("Nenhum trecho ficou acima de zero.\n")
        return

    for index, item in enumerate(hits, start=1):
        print(f"[{index}] {score_key}: {item[score_key]}")
        print(preview(item["chunk"]))
        print()


def keyword_hits(question, chunks):
    hits = keyword_search(question, chunks, top_k=TOP_K)
    # Nota 0 significa que não há palavra em comum. Esses parágrafos ficam de fora.
    return [item for item in hits if item["score"] > 0]


def embedding_hits(question, chunks, document_embeddings):
    # Importa aqui para o script abrir antes de alguém chamar a API.
    from src.gemini_client import embed_text

    scores = similarities(embed_text(question), document_embeddings)
    ranked = [
        {"chunk": chunk, "similaridade": score}
        for chunk, score in zip(chunks, scores)
    ]
    ranked.sort(key=lambda item: item["similaridade"], reverse=True)
    return ranked[:TOP_K]


def build_embeddings(chunks):
    from src.gemini_client import embed_text

    # Os vetores dos parágrafos ficam para a próxima pergunta. Só a pergunta é refeita.
    print("\nGerando embeddings da base. Na próxima pergunta esta etapa não repete.")
    vectors = []

    for index, chunk in enumerate(chunks, start=1):
        print(f"Trecho {index}/{len(chunks)}")
        vectors.append(embed_text(chunk))

    return vectors


def answer_with(question, hits, model):
    from src.gemini_client import generate_response

    context = "\n\n".join(item["chunk"] for item in hits)
    prompt = f"""
Você é um assistente de análise agrícola.

Responda à pergunta utilizando somente o contexto.
Se o contexto não bastar, diga que a informação não está disponível.
Não invente dados.

CONTEXTO:

{context}

PERGUNTA:

{question}
"""
    return generate_response(prompt, model=model)


def remember(entry):
    history = []

    if HISTORY_PATH.exists():
        import json

        history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))

    history.append(entry)
    save_json(HISTORY_PATH, history)


def run_question(question, chunks, document_embeddings, with_api):
    question = " ".join(question.split())
    by_keyword = keyword_hits(question, chunks)

    print("\n" + "=" * 70)
    print("PERGUNTA")
    print(question)
    show_hits("BUSCA POR PALAVRA", by_keyword, "score")

    by_embedding = None
    keyword_answer = None
    embedding_answer = None

    if with_api:
        if document_embeddings is None:
            document_embeddings = build_embeddings(chunks)

        by_embedding = embedding_hits(
            question,
            chunks,
            document_embeddings,
        )
        show_hits("BUSCA POR EMBEDDING", by_embedding, "similaridade")

        print("Gerando as duas respostas...\n")
        keyword_answer = answer_with(question, by_keyword, MODEL_FLASH)
        embedding_answer = answer_with(question, by_embedding, MODEL_FLASH)

        print("RESPOSTA COM O CONTEXTO DA PALAVRA")
        print("-" * 70)
        print(keyword_answer)
        print("\nRESPOSTA COM O CONTEXTO DO EMBEDDING")
        print("-" * 70)
        print(embedding_answer)

    remember(
        {
            "question": question,
            "keyword": [
                {"score": item["score"], "chunk": item["chunk"]}
                for item in by_keyword
            ],
            "embedding": None
            if by_embedding is None
            else [
                {
                    "similarity": item["similaridade"],
                    "chunk": item["chunk"],
                }
                for item in by_embedding
            ],
            "keyword_answer": keyword_answer,
            "embedding_answer": embedding_answer,
        }
    )
    print(f"\nRegistro salvo em: {HISTORY_PATH}")
    return document_embeddings


def interactive(chunks):
    document_embeddings = None

    print("\n" + "=" * 70)
    print("TESTE DO USUÁRIO")
    print("=" * 70)
    print(
        "\nEscreva uma pergunta sobre o texto agrícola."
        "\nEnter repete a pergunta do laboratório."
        "\nDigite sair para encerrar."
        "\nA busca por palavra roda na hora."
        "\nO embedding e as respostas usam a API do Gemini."
    )

    while True:
        typed = input("\nPergunta: ").strip()

        if typed.lower() in {"sair", "q"}:
            print("Encerrado.")
            return

        question = typed or RETRIEVAL_QUESTION.strip()
        use_api = input("Ver embedding e respostas do modelo? (s/n): ").strip().lower()
        document_embeddings = run_question(
            question,
            chunks,
            document_embeddings,
            with_api=use_api.startswith("s"),
        )


def main():
    parser = argparse.ArgumentParser(
        description="Testa uma pergunta na base agrícola."
    )
    parser.add_argument(
        "--pergunta",
        help="Roda uma pergunta e encerra.",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Também consulta embedding e o modelo.",
    )
    args = parser.parse_args()
    chunks = load_chunks()

    if args.pergunta:
        run_question(args.pergunta, chunks, None, with_api=args.api)
        return

    interactive(chunks)


if __name__ == "__main__":
    main()
