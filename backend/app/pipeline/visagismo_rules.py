"""
Reglas de visagismo (morfología craneal/facial + estilo de vida) sobre el
recomendador de cortes.

Este módulo traduce el `visagismo_profile` guardado en el perfil del
cliente (ver `ClientProfile.visagismo_profile` / `VisagismoProfileIn` en
`app/api/schemas.py`) en ajustes sobre `recommend_styles()`
(`app/pipeline/recommender.py`). El esquema de ese perfil usa conceptos
de visagismo bastante abstractos (morfología craneal, geometría facial,
dirección de flequillo...) que el usuario definió en un JSON aparte; este
módulo es el puente entre ese vocabulario abstracto y el vocabulario
concreto de `data/styles/styles.json` (`style_family`, `fade_type`,
`length_top_mm`), que es mucho más limitado (12 familias, 5 tipos de
fade). Por eso cada regla de abajo documenta explícitamente a qué
concepto concreto del catálogo se mapea cada término abstracto del
schema, y cuándo un término no tiene ningún equivalente razonable en este
catálogo (en cuyo caso esa regla se implementa solo a medias, o no se
implementa, en vez de forzar un mapeo arbitrario).

Sigue la misma filosofía que el resto de `recommender.py` (ver su
docstring): NINGUNA regla descarta un corte. Cada una suma o resta puntos
a un corte compatible y, si corresponde, añade una nota explicando por
qué aparece más arriba o más abajo en la lista — la decisión final es
siempre del barbero. Esto es así incluso para las reglas del schema
original que hablan de "bypass"/"avoid" (lenguaje de exclusión dura): se
traducen a una penalización fuerte (que en la práctica casi siempre las
manda al final de la lista) en vez de a una exclusión real, para no
arriesgarse a dejar una combinación de pelo/perfil sin ninguna
recomendación visible por un fallo de mapeo en este puente.

Un profile parcialmente relleno (la mayoría de sus campos son opcionales,
igual que `hair_texture_override`/`face_shape_override`) simplemente no
activa las reglas que dependen de los campos que faltan.
"""

from dataclasses import dataclass, field

from app.pipeline.rule_effects import Effect
from app.pipeline.style_catalog import HaircutStyle

# Puntuaciones: negativo = prioriza (sube en la lista), positivo = matiza
# o desaconseja (baja en la lista). Nunca se usan para descartar un corte,
# solo para ordenar los que ya son compatibles por tipo de pelo (ver
# `recommend_styles`). Dos magnitudes: floja (recomendación suave, p.ej.
# "encaja bien con...") y fuerte (para las reglas que el schema original
# expresa como "bypass"/"avoid", lenguaje de exclusión).
_BOOST_SUAVE = -1
_BOOST_FUERTE = -2
_AVISO_SUAVE = 1
_AVISO_FUERTE = 2

# style_family que requieren peinado activo con producto/secador para
# verse bien a diario (tupé/pompadour necesita volumen levantado a mano;
# melena larga y recogido necesitan mantenimiento para no verse
# desordenados; raya lateral clásica se despeina sin repasar la raya).
# Usado por la regla de `daily_maintenance_commitment`.
_FAMILIAS_REQUIEREN_PEINADO = {
    "tupe_pompadour_clasico",
    "melena_larga",
    "recogido_mono_coleta",
    "clasico_raya_lateral",
}

# Proxy de "corte de bajo mantenimiento / textura natural, sin peinar":
# uniforme (buzz) o con top corto y laterales muy degradados, que ya
# quedan bien recién salidos de la ducha sin peine ni producto.
_FAMILIAS_BAJO_MANTENIMIENTO = {"buzz_corto_uniforme"}


@dataclass
class VisagismoAdjustment:
    score: int
    note: str | None
    # Etiqueta corta para la tarjeta ("Ideal sin peinarte"); `note` es la
    # explicación completa. En `evaluate_profile`, `effects` lleva cada
    # regla aplicada por separado (ver rule_effects.py).
    label: str | None = None
    effects: list[Effect] = field(default_factory=list)


def _get(profile: dict, *path, default=None):
    """Navega un `visagismo_profile` (dict anidado, cualquier nivel puede
    faltar si el barbero no lo ha rellenado todavía) sin lanzar KeyError."""
    node = profile
    for key in path:
        if not isinstance(node, dict):
            return default
        node = node.get(key)
        if node is None:
            return default
    return node


def _regla_morfologia_craneal(style: HaircutStyle, profile: dict) -> VisagismoAdjustment | None:
    """braquicéfalo (cabeza más ancha/redondeada) -> el schema pide
    "constrain lateral_volume a bajo/compacto" y "enforce upper_volume a
    medio/alto". Se traduce como: fade alto/skin en laterales (compacto) +
    largo arriba >=30mm (volumen medio/alto). Solo se puntúa el caso en el
    que el corte SÍ cumple ambas condiciones a la vez (regla de tipo
    "enforce", no de tipo "avoid": no hay una combinación contraria clara
    que penalizar sin inventarse un umbral que el schema no da)."""
    if _get(profile, "anatomical_metrics", "cranial_morphology") != "brachycephalic":
        return None
    laterales_compactos = style.fade_type in ("alto", "skin")
    volumen_arriba = style.length_top_mm >= 30
    if laterales_compactos and volumen_arriba:
        return VisagismoAdjustment(
            _BOOST_SUAVE,
            "Cabeza de morfología braquicéfala: este corte combina laterales "
            "compactos con volumen arriba, que suele quedar bien en este caso.",
            label="Encaja con tu forma de cabeza",
        )
    return None


def _regla_geometria_facial(style: HaircutStyle, profile: dict) -> VisagismoAdjustment | None:
    """cara redonda -> el schema pide "restrict fringe_straight" (flequillo
    recto de lado a lado) y "prioritize angular_pompadour or high_fade".
    Este catálogo no tiene ninguna familia que represente un "flequillo
    recto" específicamente (la más cercana, `clasico_raya_lateral`, es
    justo lo contrario: pelo peinado hacia un lado, no un flequillo
    recto), así que la mitad de "restrict" no se implementa aquí para no
    inventar un mapeo sin base real -- solo se implementa la mitad de
    "prioritize"."""
    if _get(profile, "anatomical_metrics", "facial_geometry") != "round":
        return None
    es_tupe_angular = style.style_family == "tupe_pompadour_clasico"
    es_fade_alto = style.fade_type in ("alto", "skin")
    if es_tupe_angular or es_fade_alto:
        return VisagismoAdjustment(
            _BOOST_SUAVE,
            "Cara redonda: un pompadour con volumen definido o un fade alto "
            "suelen estilizar más que una silueta redondeada.",
            label="Estiliza la cara redonda",
        )
    return None


def _regla_nacimiento_pelo(style: HaircutStyle, profile: dict) -> VisagismoAdjustment | None:
    """Entradas en M -> el schema pide priorizar crop_texture/french_crop/
    side_part_weight_distribution y evitar slick_back. Mapeo: los cortes
    texturizados cortos (`corte_texturizado_general`, `fade_undercut_textura`)
    y la raya lateral clásica (`clasico_raya_lateral`, que reparte peso
    visual hacia un lado en vez de dejar todo el flequillo centrado sobre
    la línea de entradas) se priorizan; el tupé/pompadour peinado hacia
    atrás con mucho largo (`tupe_pompadour_clasico` + top largo) es el
    equivalente más cercano a "slick back" en este catálogo y se
    desaconseja, porque deja toda la línea de nacimiento a la vista."""
    if _get(profile, "hair_physical_metrics", "frontal_hairline_shape") != "m_shaped_receding":
        return None
    familias_priorizadas = {"corte_texturizado_general", "fade_undercut_textura", "clasico_raya_lateral"}
    if style.style_family in familias_priorizadas:
        return VisagismoAdjustment(
            _BOOST_SUAVE,
            "Entradas en M: un corte texturizado o con raya lateral disimula "
            "mejor la línea de nacimiento que dejarla muy expuesta.",
            label="Disimula las entradas",
        )
    if style.style_family == "tupe_pompadour_clasico" and style.length_top_mm >= 70:
        return VisagismoAdjustment(
            _AVISO_SUAVE,
            "Entradas en M: un tupé/pompadour peinado hacia atrás con mucho "
            "largo deja la línea de nacimiento muy a la vista.",
            label="Deja las entradas a la vista",
        )
    return None


def _regla_mantenimiento_diario(style: HaircutStyle, profile: dict) -> VisagismoAdjustment | None:
    """`daily_maintenance_commitment == zero_minutes` -> el schema dice
    literalmente "bypass" (excluir) los cortes que necesiten producto o
    secador, y recomendar buzz/crew/textura natural. Aquí se traduce el
    "bypass" en la penalización fuerte (`_AVISO_FUERTE`) en vez de una
    exclusión real -- ver el porqué en el docstring del módulo -- y el
    "recomendar" en el boost fuerte simétrico."""
    if _get(profile, "lifestyle_and_preferences", "daily_maintenance_commitment") != "zero_minutes":
        return None
    if style.style_family in _FAMILIAS_BAJO_MANTENIMIENTO or (
        style.fade_type in ("alto", "skin") and style.length_top_mm <= 15
    ):
        return VisagismoAdjustment(
            _BOOST_FUERTE,
            "Cliente sin tiempo para peinarse a diario: este corte queda bien "
            "sin producto ni secador.",
            label="Sin peinarse cada día",
        )
    if style.style_family in _FAMILIAS_REQUIEREN_PEINADO or style.length_top_mm >= 60:
        return VisagismoAdjustment(
            _AVISO_FUERTE,
            "Cliente sin tiempo para peinarse a diario: este corte necesita "
            "producto y/o secador para verse bien, puede no encajar con su "
            "rutina.",
            label="Pide peinarse cada día",
        )
    return None


def _regla_frecuencia_visitas(style: HaircutStyle, profile: dict) -> VisagismoAdjustment | None:
    """>20 días entre visitas -> el schema pide desaconsejar el fade
    "0" (piel) y recomendar taper fade (fade progresivo, sin línea neta)
    o corte clásico a tijera. Mapeo: `fade_type == "skin"` es justo el
    "0_skin_fade" del schema; `fade_type` bajo/medio es el equivalente de
    "taper_fade" en este catálogo (degradado progresivo, no a piel); y
    `clasico_raya_lateral` (sin fade marcado, corte de tijera tradicional)
    cubre el "classic_scissor_cut"."""
    dias = _get(profile, "lifestyle_and_preferences", "barbershop_visit_frequency_days")
    if dias is None or dias <= 20:
        return None
    if style.fade_type == "skin":
        return VisagismoAdjustment(
            _AVISO_SUAVE,
            f"Este cliente pasa más de 20 días entre visitas ({dias}): un fade "
            "a piel se nota crecido mucho antes que uno progresivo.",
            label="Se nota crecido pronto",
        )
    if style.fade_type in ("bajo", "medio") or style.style_family == "clasico_raya_lateral":
        return VisagismoAdjustment(
            _BOOST_SUAVE,
            f"Este cliente pasa más de 20 días entre visitas ({dias}): un "
            "degradado progresivo o un corte clásico a tijera aguantan mejor "
            "sin retoque.",
            label="Aguanta entre visitas",
        )
    return None


# Todas las reglas de arriba, en el orden en que se documentaron. Añadir
# una regla nueva aquí basta para que `evaluate_profile` la aplique.
_REGLAS = [
    _regla_morfologia_craneal,
    _regla_geometria_facial,
    _regla_nacimiento_pelo,
    _regla_mantenimiento_diario,
    _regla_frecuencia_visitas,
]


def evaluate_profile(style: HaircutStyle, profile: dict | None) -> VisagismoAdjustment:
    """Aplica todas las reglas de visagismo a un corte concreto y devuelve
    el ajuste combinado (puntuación acumulada + notas concatenadas).
    `profile` puede ser `None` (cliente sin perfil de visagismo todavía) o
    tener solo algunos campos rellenos -- ambos casos son válidos, las
    reglas que dependen de un campo ausente simplemente no se activan."""
    if not profile:
        return VisagismoAdjustment(0, None)

    total_score = 0
    notas: list[str] = []
    effects: list[Effect] = []
    for regla in _REGLAS:
        resultado = regla(style, profile)
        if resultado is None:
            continue
        total_score += resultado.score
        if resultado.note:
            notas.append(resultado.note)
            effects.append(Effect(resultado.score, resultado.label or resultado.note, resultado.note))

    return VisagismoAdjustment(total_score, " ".join(notas) if notas else None, effects=effects)
