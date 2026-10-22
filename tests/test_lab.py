import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agri_data import get_crop_productivity, get_weather_data
from src.config import KNOWLEDGE_PATH, RETRIEVAL_QUESTION
from src.retrieval import keyword_search, load_chunks, similarities


class RetrievalComparisonTest(unittest.TestCase):
    def test_keyword_search_ranks_shared_words_first(self):
        # O parágrafo do escritório divide mais palavras com a pergunta do laboratório.
        chunks = load_chunks()
        top = keyword_search(RETRIEVAL_QUESTION, chunks, top_k=1)[0]

        self.assertIn("escritório", top["chunk"])
        self.assertGreater(top["score"], 0)

    def test_cosine_prefers_the_closer_meaning(self):
        # O segundo vetor aponta para o mesmo lado da pergunta.
        query = [1.0, 0.0, 0.0]
        agronomic = [0.9, 0.1, 0.0]
        office = [0.1, 0.9, 0.0]
        scores = similarities(query, [office, agronomic])

        self.assertGreater(scores[1], scores[0])

    def test_knowledge_file_contains_the_contrast(self):
        text = KNOWLEDGE_PATH.read_text(encoding="utf-8")

        self.assertIn("Déficit hídrico e colheita", text)
        self.assertIn("Rotina do escritório da fazenda", text)

    def test_tool_names_accept_accents(self):
        crop = get_crop_productivity("algodão", "Centro Oeste")
        weather = get_weather_data("Sul")

        self.assertEqual(crop["productivity"], 1700)
        self.assertEqual(weather["water_stress"], "moderado")


if __name__ == "__main__":
    unittest.main()
