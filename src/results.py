import json
from pathlib import Path


def save_json(path, data, indent=4):
    # Cria results/ se esta for a primeira execução.
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(data, ensure_ascii=False, indent=indent) + "\n",
        encoding="utf-8",
    )
    return output
