"""
Etapa 5 del pipeline: catálogo de cortes.

Cada corte se describe de forma paramétrica (no solo como una imagen de
referencia) para poder condicionar la generación: longitud por zona,
tipo de degradado, y para qué tipos de pelo funciona bien.
"""

import json
from dataclasses import dataclass, fields
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


_CAMPOS_VALIDOS = {f.name for f in fields(HaircutStyle)}


def load_catalog(path: Path = STYLES_CATALOG_PATH) -> list[HaircutStyle]:
    """Ignora deliberadamente cualquier clave del JSON que no sea un campo
    conocido de `HaircutStyle`, en vez de dejar que `TypeError` reviente toda
    la llamada por una única entrada con una clave inesperada (ver el
    incidente documentado en `STYLES_CATALOG_PATH`, `app/config.py`): un
    catálogo con algún campo de más o desactualizado degrada mejor devolviendo
    ese estilo sin ese dato de más, no tirando abajo `/recommendations`
    entero para todos los clientes."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [
        HaircutStyle(**{k: v for k, v in item.items() if k in _CAMPOS_VALIDOS})
        for item in raw
    ]


def get_style_by_id(style_id: str, path: Path = STYLES_CATALOG_PATH) -> HaircutStyle | None:
    for style in load_catalog(path):
        if style.id == style_id:
            return style
    return None
