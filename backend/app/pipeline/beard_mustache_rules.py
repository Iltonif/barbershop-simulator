"""
Recomendación profesional de barba y bigote (sept 2026), a partir de la
forma de rostro (`face_shape_override`) y de los rasgos faciales de la
ficha (`visagismo_profile.anatomical_metrics.facial_features_profile` /
`hair_physical_metrics`). Sustituye a la antigua `trait_rules.beard_advice`
(que solo cubría mentón retraído y mandíbula poco definida) por un motor
mucho más completo, pero sigue el mismo contrato: `list[dict]` con
`label`/`detail`, sin depender del corte ni afectar a su orden.

Fuentes (indicadas por Pedro, peluquero dueño del proyecto, a partir de
vídeos y texto que él mismo transcribió/resumió en sept 2026):
- Vídeo de tipos de bigote (13 estilos: chevron, Dalí, inglés, húngaro,
  fu manchu, horizontal/lápiz, herradura, imperial, piramidal, morsa/walrus,
  mosquetero, revolucionario, corto).
- Vídeo de estilos básicos de barba (5 bases: en collar, completa/clásica,
  perilla, chiva/chivita, de varios días/media sombra -- cualquier barba
  moderna es una combinación de estas 5).
- Vídeo "adaptación del tipo de barba y bigote al tipo de rostro": qué
  estilo de barba+bigote favorece a cada forma de rostro, y correcciones
  por frente, mentón/cuello, nariz, mandíbula y labios.
- PDF "VISAGISMO MASCULINO, tipos de rostros" (transcripción de un vídeo
  sobre cómo dibujar/clasificar la forma del rostro por proporciones: 5
  líneas verticales -> "dos unidades y media", 5 líneas horizontales ->
  "tres unidades y media"; un rostro equilibrado en ambas es "ovalado").
  Esa técnica es de dibujo/clasificación manual, no una medición
  automática -- aquí solo se documenta como referencia (ver
  `frontend/guia-barba-bigote.html`), igual que ya se decidió para el
  resto de rasgos de visagismo (ver `face_analysis.py` y el historial de
  `wip-perfil-automatico`): no se reimplementa como CV.

Correlación forma de rostro -> barba y bigote (fuente: el vídeo de
adaptación al tipo de rostro). Usa las 6 formas de `face_shape_override`
más recientes (ver `schemas.FaceShapeOverrideIn`); "cuadrada" se deja
FUERA a propósito: ninguna fuente proporcionada la cubre, y el proyecto
prefiere un hueco honesto a una regla inventada (mismo criterio que ya se
sigue en `recommender.py`/`combined_rules.py` para no cubrir combinaciones
sin fuente clara).

Correcciones (mismo vídeo) por: frente (`intellectual_zone_forehead`,
mismo campo ya existente en `FacialHorizontalZonesRatioIn` pero sin
ninguna regla que lo leyera hasta ahora), entradas/alopecia
(`frontal_hairline_shape == "m_shaped_receding"`, reutilizado de
`hair_physical_metrics`), papada (`has_double_chin`, campo nuevo), nariz
(`nose_size`, campo nuevo), mandíbula/mentón (`chin_projection`, ya
existente -- fusionado con la regla previa de mentón retraído en vez de
duplicarla) y labios (`lip_thickness`, campo nuevo). El aviso de "poca
diferencia entre mandíbula y cuello" del vídeo se fusiona con la regla ya
existente de `jawline_definition == "soft"` por el mismo motivo.
"""

from __future__ import annotations


def _features(profile: dict | None) -> dict:
    if not isinstance(profile, dict):
        return {}
    anat = profile.get("anatomical_metrics") or {}
    return anat.get("facial_features_profile") or {}


def _forehead(profile: dict | None) -> str | None:
    if not isinstance(profile, dict):
        return None
    anat = profile.get("anatomical_metrics") or {}
    zonas = anat.get("facial_horizontal_zones_ratio") or {}
    return zonas.get("intellectual_zone_forehead")


def _hairline(profile: dict | None) -> str | None:
    if not isinstance(profile, dict):
        return None
    hpm = profile.get("hair_physical_metrics") or {}
    return hpm.get("frontal_hairline_shape")


def _shaved_prefix(profile: dict | None) -> str:
    lifestyle = (profile or {}).get("lifestyle_and_preferences") or {} if isinstance(profile, dict) else {}
    return "Si quiere probar barba y bigote: " if lifestyle.get("beard_preference") == "clean_shaven" else ""


# Forma de rostro -> (título, recomendación). "cuadrada" no tiene entrada:
# ver el docstring del módulo.
_POR_FORMA = {
    "ovalada": (
        "Rostro ovalado: casi cualquier estilo vale",
        "Es la forma más equilibrada: admite prácticamente cualquier combinación de barba y bigote (de la "
        "barba de varios días al bigote imperial o al fu manchu) sin que ninguna desequilibre la cara. Elige "
        "según el resto de rasgos (mentón, nariz, labios) más que por la forma del rostro en sí.",
    ),
    "alargada": (
        "Rostro alargado: barba rebajada y bigote que acorte",
        "Una barba corta y rebajada (poco volumen en la barbilla, para no alargar más la cara) combinada con "
        "un bigote horizontal o de lápiz -- que marca una línea horizontal y acorta visualmente la cara -- es "
        "la combinación más favorecedora.",
    ),
    "redonda": (
        "Rostro redondo: barba angulosa y bigote grande",
        "Una barba de candado con líneas angulosas, o una chivita/perilla alargada hacia abajo, estiliza y "
        "alarga la cara. Combinada con un bigote grande (imperial o morsa/walrus) que aporte volumen y "
        "líneas marcadas, contrarresta la redondez.",
    ),
    "diamante": (
        "Rostro en diamante: barba de candado y bigote con pelo bajo el labio",
        "Una barba de candado, que rellena la línea de la mandíbula (la zona más estrecha en este rostro), "
        "combinada con un bigote que se prolonga con algo de pelo bajo el labio (estilo mosquetero) equilibra "
        "los pómulos marcados típicos de esta forma.",
    ),
    "triangular_invertida": (
        "Rostro triangular invertido: barba completa y densa en mentón y mejillas",
        "Con la frente/pómulos más anchos que la mandíbula, una barba completa y densa, marcada sobre todo en "
        "el mentón y las mejillas, rellena visualmente la parte baja de la cara y equilibra la proporción.",
    ),
    "triangular": (
        "Rostro triangular: perilla o chivita muy corta y pulida",
        "Con la mandíbula más ancha que la frente, una perilla o chivita muy corta y bien pulida (sin dejar "
        "que la barba gane anchura en la mandíbula) es lo más favorecedor. Evitar formas rectas o cuadradas "
        "en el contorno de la barba, que acentúan la anchura de la mandíbula.",
    ),
}


def _por_forma(face_shape: str | None, prefix: str) -> list[dict]:
    entry = _POR_FORMA.get(face_shape or "")
    if not entry:
        return []
    label, detail = entry
    return [{"label": label, "detail": prefix + detail}]


def _frente(profile: dict | None) -> list[dict]:
    out: list[dict] = []
    frente = _forehead(profile)
    if frente == "narrow":
        out.append({
            "label": "Barba y frente pequeña",
            "detail": "Con la frente pequeña, evita una barba muy larga o poblada: abruma la proporción "
                      "superior de la cara. Mejor un volumen de barba moderado.",
        })
    elif frente == "prominent":
        out.append({
            "label": "Barba y frente ancha",
            "detail": "Con la frente ancha, puedes llevar una barba más larga y poblada sin que se vea "
                      "desproporcionada: ayuda a equilibrar la parte de arriba con la de abajo.",
        })
    if _hairline(profile) == "m_shaped_receding":
        out.append({
            "label": "Barba con entradas",
            "detail": "Con entradas o alopecia frontal, ten cuidado de no concentrar todo el volumen visual "
                      "en la barba mientras la parte de arriba se ve más rala: busca un equilibrio entre "
                      "ambas zonas en vez de compensar una con la otra.",
        })
    return out


def _menton_y_mandibula(f: dict) -> list[dict]:
    """Fusiona la corrección de mandíbula/mentón del vídeo de barba y
    bigote con la regla previa de `trait_rules.beard_advice` para
    `chin_projection` (mismo campo, no se duplica)."""
    chin = f.get("chin_projection")
    if chin == "retruded":
        return [{
            "label": "Barba para el mentón retraído",
            "detail": "Una barba tupida y con cuerpo en la barbilla (barba completa o perilla cerrada, "
                      "dejando 2-3 cm) da volumen y compensa la retrusión del mentón. Evita la barba de pocos "
                      "días muy clara, que lo hace parecer más débil. Si llevas bigote, que sea muy recortado "
                      "para no restarle protagonismo a la barbilla.",
        }]
    if chin == "prominent":
        return [{
            "label": "Barba y bigote con mentón/mandíbula prominente",
            "detail": "Con mentón o mandíbula prominente (prognatismo), un bigote de línea recta y sin "
                      "adornos (tipo chevron o inglés) es lo más favorecedor. Si llevas barba, que sea muy "
                      "rebajada, para no acentuar más la proyección.",
        }]
    return []


def _cuello_mandibula(f: dict) -> list[dict]:
    """Fusiona "poca diferencia entre mandíbula y cuello" (vídeo de barba y
    bigote) con la regla previa de `jawline_definition == "soft"`."""
    if f.get("jawline_definition") == "soft":
        return [{
            "label": "Barba para marcar la mandíbula",
            "detail": "Con poca diferencia entre la mandíbula y el cuello, deja la barba con algo de largo y "
                      "líneas rectas en la línea de la mandíbula, y perfila el cuello por debajo dibujando una "
                      "línea que se incline algo hacia atrás (no más de un dedo por encima de la nuez): así "
                      "se alarga visualmente el cuello y se define la mandíbula en vez de dejar a la vista la "
                      "zona blanda de debajo.",
        }]
    return []


def _papada(f: dict) -> list[dict]:
    if f.get("has_double_chin"):
        return [{
            "label": "Barba con papada",
            "detail": "Con papada o doble mentón, una barba completa que cubra bien esa zona ayuda a "
                      "disimularla. Evita las barbas muy rebajadas o el afeitado muy apurado por debajo de la "
                      "mandíbula, que dejan la papada más a la vista.",
        }]
    return []


def _nariz(f: dict) -> list[dict]:
    tamano = f.get("nose_size")
    if tamano == "large":
        return [{
            "label": "Bigote y nariz grande",
            "detail": "Con la nariz grande o larga, un bigote tupido equilibra la parte central de la cara. "
                      "Se puede combinar sin problema con barba.",
        }]
    if tamano == "small":
        return [{
            "label": "Barba y bigote con nariz pequeña",
            "detail": "Con la nariz pequeña, usa una barba y un bigote muy rebajados y discretos, para no "
                      "desequilibrar las proporciones de la cara.",
        }]
    return []


def _labios(f: dict) -> list[dict]:
    labios = f.get("lip_thickness")
    if labios == "prominent":
        return [{
            "label": "Bigote y labios prominentes",
            "detail": "Con los labios prominentes, un bigote tupido ayuda a disimularlos.",
        }]
    if labios == "thin":
        return [{
            "label": "Bigote y labios finos",
            "detail": "Con los labios finos, un bigote muy recortado no acentúa la delgadez del labio.",
        }]
    return []


def advice(profile: dict | None, face_shape: str | None) -> list[dict]:
    """Consejo de barba y bigote (no depende del corte). Vacío si no hay
    forma de rostro ni rasgos que lo justifiquen. Si el cliente ha dicho
    que va afeitado, se da igual pero como sugerencia (ver
    `_shaved_prefix`)."""
    f = _features(profile)
    prefix = _shaved_prefix(profile)
    out: list[dict] = []
    out += _por_forma(face_shape, prefix)
    out += _menton_y_mandibula(f)
    out += _cuello_mandibula(f)
    out += _papada(f)
    out += _frente(profile)
    out += _nariz(f)
    out += _labios(f)
    return out
