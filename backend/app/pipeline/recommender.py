"""
Recomendador de cortes de pelo a partir del perfil del cliente.

Cruza el tipo de cabello (fijado a mano por el barbero, ver
`ClientProfile.hair_texture_override`) con el catálogo de cortes
(`style_catalog.load_catalog`), y tiene en cuenta dos señales adicionales
del perfil, cada una opcional:

- El mapa de crecimiento/remolinos dibujado en `frontend/growth-map.html`:
  avisa cuando un corte muy corto y uniforme (buzz) puede no ser buena idea
  si hay remolinos marcados, porque con tan poco largo arriba se notan
  mucho más y son más difíciles de disimular.
- La forma de cara (`ClientProfile.face_shape_override`, fijada a mano por
  el barbero — ver la nota en `clients_routes.get_recommendations` sobre
  por qué no se usa la detección automática sin confirmar): avisa cuando un
  corte no encaja bien con las recomendaciones habituales de barbería para
  esa forma de cara.

Ninguna de las dos señales descarta cortes: los avisa y los deja más abajo
en la lista, con una nota explicando el motivo, para que decida el
barbero — es una heurística orientativa, no una regla estricta.

Las reglas de forma de cara están basadas en varias fuentes de peluquería/
barbería consultadas en septiembre de 2026 (Llongueras, fabricbarberia.com,
manofmany.com, entre otras — ver la respuesta al usuario donde se citan).
Solo se aplican reglas de "evitar" cuando varias fuentes independientes
coinciden; por eso "ovalada" y "cuadrada" no tienen ninguna restricción
aquí (las fuentes las describen como versátiles y a veces se contradicen
entre sí en los detalles), mientras que "redonda" y "alargada" sí, porque
ahí todas las fuentes consultadas coinciden en la misma dirección.
"""

from dataclasses import dataclass

from app.pipeline.style_catalog import HaircutStyle, load_catalog

# Un remolino es más difícil de disimular cuanto más corto es el largo
# arriba: por eso el umbral mira solo `length_top_mm` y no el fade_type de
# los laterales — en este catálogo un fade "alto"/"skin" en los laterales
# convive con largo normal arriba (p.ej. undercuts), así que no es un buen
# indicador por sí solo. Lo realmente problemático es un corte tipo buzz,
# muy corto y uniforme en toda la cabeza.
_UMBRAL_LARGO_PROBLEMATICO_MM = 10

# Familias del catálogo (ver style_catalog.py / el script que asignó las
# fotos) que refuerzan visualmente una cara redonda: el tazón por su
# silueta redondeada, y el afro/rizado voluminoso por su forma esférica.
_FAMILIAS_REDONDEADAS = {
    "corte_tazon_bowl_liso",
    "corte_tazon_bowl_rizado",
    "afro_rizado_voluminoso",
}
_FAMILIA_TUPE_POMPADOUR = "tupe_pompadour_clasico"


@dataclass
class StyleRecommendation:
    style: HaircutStyle
    note: str | None


def _es_potencialmente_problematico_con_remolinos(style: HaircutStyle) -> bool:
    return style.length_top_mm <= _UMBRAL_LARGO_PROBLEMATICO_MM


def _nota_remolinos(style: HaircutStyle, whorls: list[dict]) -> str | None:
    if not whorls or not _es_potencialmente_problematico_con_remolinos(style):
        return None
    if len(whorls) >= 2:
        return (
            "Este cliente tiene varios remolinos marcados: un corte tan corto y "
            "uniforme puede dejarlos muy a la vista. Puede valer la pena dejar "
            "algo más de largo arriba para disimularlos."
        )
    return (
        "Hay un remolino marcado en el mapa de crecimiento: con tan poco largo "
        "arriba puede notarse. Valorar dejar algo más de longitud para "
        "disimularlo."
    )


def _nota_forma_cara(style: HaircutStyle, face_shape: str | None) -> str | None:
    """Aviso (no descarte) cuando el corte va a contracorriente de lo que
    recomiendan varias fuentes de barbería para esta forma de cara. Solo
    cubre "redonda" y "alargada" — ver el docstring del módulo."""
    if not face_shape:
        return None

    is_bowl_or_voluminous = style.style_family in _FAMILIAS_REDONDEADAS
    is_buzz_muy_corto = style.fade_type == "ninguno" and style.length_top_mm <= 10

    if face_shape == "redonda" and (is_bowl_or_voluminous or is_buzz_muy_corto):
        return (
            "Con cara redonda, varias guías de barbería recomiendan dar volumen "
            "arriba en vez de una silueta redondeada o muy uniforme — puede "
            "acentuar la redondez de la cara."
        )

    is_tupe_alto = style.style_family == _FAMILIA_TUPE_POMPADOUR
    is_fade_muy_alto_con_top_largo = style.fade_type in ("alto", "skin") and style.length_top_mm >= 90

    if face_shape == "alargada" and (is_tupe_alto or is_fade_muy_alto_con_top_largo):
        return (
            "Con cara alargada, sumar más altura arriba (tipo tupé/pompadour o "
            "un fade muy marcado con mucho largo arriba) tiende a alargarla "
            "más — suele funcionar mejor un corte que sume anchura, como uno "
            "con flequillo o raya lateral."
        )

    return None


def recommend_styles(
    hair_texture: str,
    whorls: list[dict] | None = None,
    face_shape: str | None = None,
) -> list[StyleRecommendation]:
    """Devuelve los cortes del catálogo compatibles con `hair_texture`,
    ordenados de más a menos recomendados. `whorls` es la lista tal cual se
    guarda en `ClientProfile.custom_growth_map["whorls"]`, y `face_shape` es
    `ClientProfile.face_shape_override` — ambos opcionales (`None`/vacío si
    el barbero no los ha rellenado; en ese caso simplemente no se aplica esa
    señal, no es un error)."""

    whorls = whorls or []

    compatibles = [
        style for style in load_catalog() if hair_texture in style.suitable_hair_types
    ]

    recomendaciones = []
    for style in compatibles:
        notas = [
            n
            for n in (_nota_remolinos(style, whorls), _nota_forma_cara(style, face_shape))
            if n
        ]
        note = " ".join(notas) if notas else None
        recomendaciones.append(StyleRecommendation(style=style, note=note))

    # Los que no tienen ningún aviso van primero; dentro de cada grupo, se
    # mantiene el orden del catálogo.
    recomendaciones.sort(key=lambda r: r.note is not None)
    return recomendaciones
