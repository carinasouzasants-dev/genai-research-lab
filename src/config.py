from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
KNOWLEDGE_PATH = DATA_DIR / "agricultural_knowledge.txt"

MODEL_FLASH = "gemini-3.8-flash"
MODEL_FLASH_37 = "gemini-3.7-flash"
MODEL_FLASH_36 = "gemini-3.6-flash"
EMBEDDING_MODEL = "gemini-embedding-2"

# Mesma pergunta nos experimentos 05 a 08.
# "falta" e "produz" batem no parágrafo do escritório.
# O da lavoura fala em "secos", "graos" e "colheita".
RETRIEVAL_QUESTION = """
Como a falta de chuva reduz o que a planta produz?
"""
