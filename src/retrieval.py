import re
from pathlib import Path

import numpy as np

from src.config import KNOWLEDGE_PATH


# "de" e "que" aparecem em quase todo parágrafo, então não entram na conta.
STOPWORDS = {
    "como",
    "quando",
    "para",
    "pela",
    "pelo",
    "pelas",
    "pelos",
    "uma",
    "uns",
    "umas",
    "que",
    "com",
    "sem",
    "por",
    "das",
    "dos",
    "nas",
    "nos",
    "numa",
    "num",
    "este",
    "esta",
    "esse",
    "essa",
    "isso",
    "sobre",
    "entre",
    "mais",
    "menos",
    "sao",
    "são",
    "ser",
    "foi",
    "nao",
    "não",
    "sua",
    "seu",
    "pode",
    "podem",
    "deve",
    "devem",
}


def split_paragraphs(text):
    return [
        chunk.strip()
        for chunk in re.split(r"\n\s*\n", text)
        if chunk.strip()
    ]


def load_chunks(path=None):
    source = Path(path) if path is not None else KNOWLEDGE_PATH
    return split_paragraphs(source.read_text(encoding="utf-8"))


def tokenize(text):
    words = re.findall(
        r"\b[a-zá-úA-ZÁ-Ú]+\b",
        text.lower(),
    )
    return {
        word
        for word in words
        if word not in STOPWORDS and len(word) > 2
    }


def keyword_search(question, chunks, top_k=3):
    # A nota é quantas palavras da pergunta também estão no parágrafo.
    question_words = tokenize(question)
    scored_chunks = []

    for chunk in chunks:
        score = len(question_words.intersection(tokenize(chunk)))
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


def cosine_similarity(vector_a, vector_b):
    array_a = np.asarray(vector_a, dtype=float)
    array_b = np.asarray(vector_b, dtype=float)
    norm_a = np.linalg.norm(array_a)
    norm_b = np.linalg.norm(array_b)

    if norm_a == 0 or norm_b == 0:
        # Vetor zerado não tem direção, então não há o que comparar.
        return 0.0

    return float(np.dot(array_a, array_b) / (norm_a * norm_b))


def similarities(query_vector, document_vectors):
    # Cosseno da pergunta contra todos os parágrafos de uma vez.
    matrix = np.asarray(document_vectors, dtype=float)
    query = np.asarray(query_vector, dtype=float)
    query_norm = np.linalg.norm(query)
    document_norms = np.linalg.norm(matrix, axis=1)
    dots = matrix @ query
    denominators = document_norms * query_norm
    scores = np.divide(
        dots,
        denominators,
        out=np.zeros(len(matrix), dtype=float),
        where=denominators != 0,
    )
    return [round(float(score), 6) for score in scores]
