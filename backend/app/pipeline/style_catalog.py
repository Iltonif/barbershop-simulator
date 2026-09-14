"""
Etapa 5 del pipeline: catálogo de cortes.

Cada corte se describe de forma paramétrica (no solo como una imagen de
referencia) para poder condicionar la generación: longitud por zona,
tipo de degradado, y para qué tipos de pelo funciona bien.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from app.config import STYLES_CATALOG_PATH


@dataclass
class HaircutStyle:
    id: str
    name: str
    description: str
    length_top_mm: int
    length_sides_mm: int
    length_back_mm: int
    fade_type: str  # "ninguno" | "bajo" | "medio" | "alto" | "skin"
    suitable_hair_types: list[str]
    reference_image: str | None = None
    # Añadidos al importar el ranking de 100 cortes de Esquire España (ver
    # scripts/import_esquire_styles.py): clasificación directa por longitud
    # (independiente de los mm exactos, que son una estimación orientativa
    # a partir de esa categoría) y de dónde sale la descripción del corte,
    # para poder auditar/corregir clasificaciones concretas más adelante.
    length_category: str | None = None  # "corto" | "medio" | "largo" | "extra_largo"
    source: str | None = None
    # Familia visual usada para agrupar cortes que comparten una foto de
    # referencia de stock (ver scripts/assign_photos.py): varios cortes del
    # catálogo comparten familia y por tanto la misma reference_image.
    style_family: str | None = None


def load_catalog(path: Path = STYLES_CATALOG_PATH) -> list[HaircutStyle]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [HaircutStyle(**item) for item in raw]


def get_style_by_id(style_id: str, path: Path = STYLES_CATALOG_PATH) -> HaircutStyle | None:
    for style in load_catalog(path):
        if style.id == style_id:
            return style
    return None
