import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel

from src.config import EMBEDDING_MODEL, MODEL_FLASH, RESULTS_DIR
from src.gemini_client import embed_text, generate_content
from src.results import save_json
from src.retrieval import cosine_similarity, load_chunks


class AgroGenAIAnalysis(BaseModel):
    categoria: str
    prioridade: str
    resumo_executivo: str
    fatores_identificados: list[str]
    evidencias_da_base: list[str]
    recomendacoes: list[str]
    riscos_e_limitacoes: list[str]
    requer_analise_humana: bool


agricultural_report = """
Relatório experimental de uma área de produção de soja.

Foi observada redução da produtividade estimada em relação ao histórico
da área. O período também apresentou irregularidade na distribuição
das chuvas e sinais de possível déficit hídrico.

As temperaturas médias ficaram acima do padrão observado anteriormente.

Também existem indícios de alterações nas condições do solo, porém os
dados disponíveis ainda não são suficientes para determinar a causa
principal da queda de produtividade.

Para uma análise mais confiável, devem ser considerados dados históricos
de produtividade, umidade e características do solo, precipitação,
temperatura, manejo da cultura e ocorrência de pragas ou doenças.

Este relatório é experimental e não representa um diagnóstico agrícola.
"""


def semantic_search(query, documents, top_k=3):
    # Transforma o relatório e cada parágrafo em vetor e fica com os mais próximos.
    query_embedding = embed_text(query)
    scored_documents = []

    for document in documents:
        similarity = cosine_similarity(
            query_embedding,
            embed_text(document),
        )
        scored_documents.append(
            {
                "document": document,
                "similarity": similarity,
            }
        )

    scored_documents.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )
    return scored_documents[:top_k]


def main():
    print("\n" + "=" * 70)
    print("AGROGENAI | ANÁLISE INTELIGENTE DE CENÁRIO AGRÍCOLA")
    print("=" * 70)
    print("\nCarregando base de conhecimento...")

    knowledge_base = load_chunks()
    print(f"Documentos encontrados: {len(knowledge_base)}")
    print("\nExecutando busca semântica...")

    retrieved_documents = semantic_search(
        agricultural_report,
        knowledge_base,
        top_k=3,
    )

    print("\nDocumentos mais relevantes:")

    for index, item in enumerate(retrieved_documents, start=1):
        print(f"\n[{index}] Similaridade: {item['similarity']:.4f}")
        print(item["document"][:300])

    context = "\n\n".join(
        (
            f"EVIDÊNCIA {index} "
            f"(similaridade {item['similarity']:.4f}):\n"
            f"{item['document']}"
        )
        for index, item in enumerate(retrieved_documents, start=1)
    )

    prompt = f"""
Você é um assistente experimental de inteligência artificial
voltado para análise de informações agrícolas.

Seu objetivo é analisar um relatório agrícola utilizando somente
as evidências fornecidas e produzir uma análise estruturada.

IMPORTANTE:

- Não invente dados.
- Não apresente hipóteses como fatos.
- Diferencie evidência de hipótese.
- Não produza diagnóstico definitivo.
- Recomendações devem ser consideradas sugestões para investigação.
- Sempre indique limitações dos dados.
- Determine se a análise humana é necessária.
- O resultado será usado apenas como protótipo experimental.

RELATÓRIO AGRÍCOLA:

{agricultural_report}

BASE DE CONHECIMENTO RECUPERADA POR BUSCA SEMÂNTICA:

{context}

Produza uma análise contendo:

1. Categoria do problema.
2. Prioridade.
3. Resumo executivo.
4. Fatores identificados.
5. Evidências utilizadas da base.
6. Recomendações para investigação.
7. Riscos e limitações.
8. Indicação explícita se análise humana é necessária.

A resposta deve seguir exatamente o schema fornecido.
"""

    print("\nGerando análise com IA...")

    response = generate_content(
        prompt,
        model=MODEL_FLASH,
        config={
            "response_mime_type": "application/json",
            "response_schema": AgroGenAIAnalysis,
        },
    )
    analysis = AgroGenAIAnalysis.model_validate_json(response.text)

    print("\n" + "=" * 70)
    print("RESULTADO AGROGENAI")
    print("=" * 70)
    print(f"\nCategoria:\n{analysis.categoria}")
    print(f"\nPrioridade:\n{analysis.prioridade}")
    print(f"\nResumo executivo:\n{analysis.resumo_executivo}")
    print("\nFatores identificados:")

    for factor in analysis.fatores_identificados:
        print(f"- {factor}")

    print("\nRecomendações:")

    for recommendation in analysis.recomendacoes:
        print(f"- {recommendation}")

    print("\nRiscos e limitações:")

    for risk in analysis.riscos_e_limitacoes:
        print(f"- {risk}")

    print(f"\nRequer análise humana: {analysis.requer_analise_humana}")

    output_path = save_json(
        RESULTS_DIR / "agrogenai_12.json",
        {
            "project": "AgroGenAI",
            "experiment": 12,
            "generation_model": MODEL_FLASH,
            "embedding_model": EMBEDDING_MODEL,
            "knowledge_documents": len(knowledge_base),
            "retrieved_documents": [
                {
                    "similarity": item["similarity"],
                    "document": item["document"],
                }
                for item in retrieved_documents
            ],
            "agricultural_report": agricultural_report.strip(),
            "analysis": analysis.model_dump(),
        },
    )

    print("\n" + "=" * 70)
    print("AGROGENAI CONCLUÍDO")
    print("=" * 70)
    print(f"\nResultado salvo em:\n{output_path}")
    print("\nPipeline executado:")
    print(
        "Relatório agrícola"
        " → Embeddings"
        " → Busca semântica"
        " → Recuperação de contexto"
        " → LLM"
        " → Structured Output"
        " → Análise"
    )
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
