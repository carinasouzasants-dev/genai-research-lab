import unicodedata


# Produtividade inventada. Não são números de fazenda real.
CROP_PRODUCTIVITY = {
    "soja": {
        "sul": {"productivity": 3200, "unit": "kg/ha"},
        "centro-oeste": {"productivity": 3500, "unit": "kg/ha"},
    },
    "milho": {
        "sul": {"productivity": 6800, "unit": "kg/ha"},
        "centro-oeste": {"productivity": 7200, "unit": "kg/ha"},
    },
    "algodao": {
        "sul": {"productivity": 1500, "unit": "kg/ha"},
        "centro-oeste": {"productivity": 1700, "unit": "kg/ha"},
    },
}

WEATHER = {
    "sul": {
        "rainfall_mm": 820,
        "average_temperature_c": 22.4,
        "water_stress": "moderado",
    },
    "centro-oeste": {
        "rainfall_mm": 760,
        "average_temperature_c": 24.8,
        "water_stress": "alto",
    },
}


def _key(value):
    # "Algodão" e "Centro Oeste" precisam achar "algodao" e "centro-oeste".
    folded = unicodedata.normalize("NFKD", value)
    without_marks = "".join(
        char for char in folded if not unicodedata.combining(char)
    )
    return without_marks.lower().strip().replace(" ", "-")


def get_crop_productivity(crop, region):
    result = (
        CROP_PRODUCTIVITY
        .get(_key(crop), {})
        .get(_key(region))
    )

    if result is None:
        return {
            "productivity": None,
            "unit": None,
            "message": (
                "Não existem dados experimentais "
                "para essa combinação."
            ),
        }

    return dict(result)


def get_weather_data(region):
    result = WEATHER.get(_key(region))

    if result is None:
        return {
            "rainfall_mm": None,
            "average_temperature_c": None,
            "water_stress": "dados indisponíveis",
        }

    return dict(result)
