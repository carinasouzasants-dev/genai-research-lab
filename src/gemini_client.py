import os
import time

from dotenv import load_dotenv
from google import genai

from src.config import EMBEDDING_MODEL, ENV_FILE, MODEL_FLASH


# O .env fica na raiz do projeto, mesmo se o script rodar de outra pasta.
load_dotenv(dotenv_path=ENV_FILE)

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY não encontrada. "
        "Verifique o arquivo .env."
    )

client = genai.Client(api_key=api_key)

RETRY_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 2


def _is_retryable(error):
    # 503 e 429 querem dizer que o modelo está ocupado. O resto interrompe.
    message = str(error).lower()
    return any(
        token in message
        for token in (
            "503",
            "429",
            "unavailable",
            "resource exhausted",
            "high demand",
        )
    )


def _call_with_retry(operation):
    last_error = None

    for attempt in range(RETRY_ATTEMPTS):
        try:
            return operation()
        except Exception as error:
            last_error = error
            if not _is_retryable(error) or attempt == RETRY_ATTEMPTS - 1:
                raise
            time.sleep(RETRY_WAIT_SECONDS * (attempt + 1))

    raise last_error


def generate_content(
    prompt=None,
    model=MODEL_FLASH,
    config=None,
    contents=None,
):
    payload = contents if contents is not None else prompt
    kwargs = {
        "model": model,
        "contents": payload,
    }

    if config is not None:
        kwargs["config"] = config

    return _call_with_retry(
        lambda: client.models.generate_content(**kwargs)
    )


def generate_response(
    prompt,
    model=MODEL_FLASH,
    config=None,
):
    response = generate_content(
        prompt,
        model=model,
        config=config,
    )
    return response.text


def embed_text(text, model=EMBEDDING_MODEL):
    response = _call_with_retry(
        lambda: client.models.embed_content(
            model=model,
            contents=text,
        )
    )
    return response.embeddings[0].values
