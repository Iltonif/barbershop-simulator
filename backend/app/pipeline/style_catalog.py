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
    # Formas de cara para las que ESTE corte concreto viene recomendado por
    # su propia ficha de origen (sept 2026: documento aportado por Pedro con
    # 5 cortes con nombre, cada uno con su recomendación explícita de forma
    # de cara -- ver recommender._efecto_forma_cara_especifica). Es distinto
    # de las reglas generales de `recommender._efecto_forma_cara` (que
    # razonan por atributos del corte -- largo, fade, familia -- para
    # cualquier corte del catálogo): esto es la recomendación tal cual la
    # trae la fuente de ese corte en concreto, sin inferir nada. Valores
    # esperados: los mismos que `ClientProfile.face_shape_override`
    # ("ovalada"|"redonda"|"cuadrada"|"alargada"|"diamante"|"triangular"|
    # "triangular_invertida"). `None`/lista vacía si el corte no trae esa
    # recomendación (la inmensa mayoría del catálogo, importado sin ese
    # dato).
    recommended_face_shapes: list[str] | None = None


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


def load_custom_styles() -> list[HaircutStyle]:
    """Cortes que el peluquero ha añadido desde el propio selector cuando no
    encontraba el que buscaba (ver `frontend/assets/style-picker.js`, sept
    2026). Se guardan en la base de datos, no en `STYLES_CATALOG_PATH` --
    ver la nota de ese mismo nombre en `app/config.py` sobre por qué ese
    JSON no puede recibir escrituras en producción sin perderlas en el
    siguiente despliegue.

    Import diferido de `app.db.repository`: así `load_catalog`/
    `get_style_by_id` (usados por tests e import scripts con un `path`
    propio) siguen sin depender de la base de datos, y solo paga ese coste
    quien de verdad necesita el catálogo completo (`load_full_catalog`)."""
    from app.db import repository

    return [
        HaircutStyle(**row, source="peluquero", style_family=None, length_category=None, reference_image=None)
        for row in repository.list_custom_styles()
    ]


def load_full_catalog(path: Path = STYLES_CATALOG_PATH) -> list[HaircutStyle]:
    """El catálogo base (`load_catalog`, versionado en git) más los cortes
    propios del peluquero (`load_custom_styles`, en la base de datos). Es lo
    que deben usar el catálogo, el simulador, las recomendaciones y el
    historial -- cualquier sitio donde un corte "cualquiera" del negocio
    tiene que poder aparecer. `load_catalog`/`get_style_by_id` a secas se
    dejan para los tests y scripts que trabajan solo con el JSON base."""
    return load_catalog(path) + load_custom_styles()


def get_style_by_id_anywhere(style_id: str, path: Path = STYLES_CATALOG_PATH) -> HaircutStyle | None:
    for style in load_full_catalog(path):
        if style.id == style_id:
            return style
    return None
