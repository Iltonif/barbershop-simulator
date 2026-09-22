"""
Parámetros del maniquí personalizado de cada cliente (growth-map.html y
la vista de la ficha): tipo, largo, color y densidad del pelo, línea del
pelo, y qué rasgos de la cara se aplican y cuánto.

Pedro: "que el maniquí vaya cambiando en función de las características
del cliente: si tiene el pelo afro, que tenga pelo afro y sus físicas, así
con todos los tipos de cabello, la longitud y las características de
visajismo (mandíbula, ojos, simetría, orejas, entradas...)".

Aquí solo se traduce la ficha a números; la forma de cada rasgo son
"morph targets" de MakeHuman (CC0) ya calculados sobre el maniquí
(`frontend/assets/rasgos/*.bin`, los genera
`tools/construir_cabeza_masculina.py`) y el pelo lo dibuja
`frontend/assets/avatar.js`.

Largo actual: si el peluquero lo puso a mano, eso más lo que ha crecido
desde entonces; si no, el del último corte registrado más lo crecido; si
tampoco hay, uno corto por defecto. El pelo crece de media ~1,25 cm al mes
(~0,4 mm/día); se limita a 6 meses de crecimiento para no inventar melenas.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from app.db.models import ClientProfile

GROWTH_MM_PER_DAY = 0.41
MAX_GROWTH_DAYS = 180
DEFAULT_LENGTH = {"top": 40, "sides": 15, "back": 15}

HAIR_COLORS = ("negro", "castano_oscuro", "castano", "castano_claro", "rubio", "pelirrojo", "canoso")

_FACE_OVERRIDE = {"ovalada": "face-oval", "redonda": "face-round", "cuadrada": "face-square", "alargada": "face-rectangular"}
_FACE_GEOMETRY = {"oval": "face-oval", "round": "face-round", "square": "face-square",
                  "rectangular_elongated": "face-rectangular", "diamond": "face-diamond",
                  "triangle": "face-triangle", "heart": "face-heart"}
_PATTERN_TO_TEXTURE = {"straight": "liso", "wavy": "ondulado", "curly": "rizado", "coily": "afro"}


def _get(d: dict | None, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _parse(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts.replace(" ", "T"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def morph_weights(client: ClientProfile) -> dict[str, float]:
    """Rasgos de la ficha -> {nombre del morph: peso 0-1}."""
    vp = client.visagismo_profile or {}
    am = vp.get("anatomical_metrics") or {}
    ff = am.get("facial_features_profile") or {}
    w: dict[str, float] = {}

    face = _FACE_OVERRIDE.get(client.face_shape_override or "") or _FACE_GEOMETRY.get(am.get("facial_geometry") or "")
    if face:
        w[face] = 0.8
    skull = {"brachycephalic": "skull-short", "dolichocephalic": "skull-long"}.get(am.get("cranial_morphology"))
    if skull:
        w[skull] = 0.8
    chin = {"retruded": "chin-retruded", "prominent": "chin-prominent"}.get(ff.get("chin_projection"))
    if chin:
        w[chin] = 0.9
    jaw = {"soft": "jaw-soft", "defined": "jaw-defined"}.get(ff.get("jawline_definition"))
    if jaw:
        w[jaw] = 0.8
    if ff.get("ears_projection") == "prominent_protruding":
        w["ears-prominent"] = 1.0
    eyes = {"close_set": "eyes-close", "wide_set": "eyes-wide"}.get(ff.get("eye_spacing"))
    if eyes:
        w[eyes] = 0.8
    if ff.get("eye_symmetry") == "asymmetric":
        # El análisis automático deja escrito cuál está más abierto; el
        # pequeño es el otro. Sin esa nota no se sabe el lado: no se aplica.
        m = re.search(r"ojo (derecho|izquierdo) está", ff.get("detected_anomalies_notes") or "")
        if m:
            pct = float(ff.get("eye_symmetry_percent") or 25)
            w["eye-l-small" if m.group(1) == "derecho" else "eye-r-small"] = min(1.0, pct / 40)
    profile = ff.get("profile_type")
    if profile == "convex_prominent_nose":
        w["nose-convex"] = 0.8
    elif profile == "concave":
        w["nose-concave"] = 0.8
        w["chin-prominent"] = max(w.get("chin-prominent", 0), 0.3)
    neck = {"short_thick": ("neck-short", 0.7), "long_thin": ("neck-long", 0.5)}.get(ff.get("neck_proportions"))
    if neck:
        w[neck[0]] = neck[1]
    if ff.get("eyebrow_type") == "prominent_ridge":
        w["brow-ridge"] = 0.6
    forehead = {"prominent": "forehead-tall", "narrow": "forehead-short"}.get(
        _get(am, "facial_horizontal_zones_ratio", "intellectual_zone_forehead"))
    if forehead:
        w[forehead] = 0.7
    return w


def current_length(client: ClientProfile, last_cut: dict | None, now: datetime | None = None) -> dict:
    """Largo de hoy en mm por zona + de dónde sale.

    `last_cut`: {"at": iso, "top", "sides", "back", "fade"} del último corte
    registrado con un corte del catálogo, o None."""
    now = now or datetime.now(timezone.utc)

    def grown(since: str) -> float:
        days = max(0.0, (now - _parse(since)).total_seconds() / 86400)
        return round(min(days, MAX_GROWTH_DAYS) * GROWTH_MM_PER_DAY, 1)

    if client.current_length and client.current_length_at:
        g = grown(client.current_length_at)
        base, source, fade = client.current_length, "peluquero", client.current_length.get("fade") or "ninguno"
    elif last_cut:
        g = grown(last_cut["at"])
        base, source, fade = last_cut, "corte", last_cut.get("fade") or "ninguno"
    else:
        g, base, source, fade = 0.0, DEFAULT_LENGTH, "defecto", "ninguno"
    return {
        "top": round(base["top"] + g, 1), "sides": round(base["sides"] + g, 1), "back": round(base["back"] + g, 1),
        "fade": fade, "fade_mm": round(g, 1), "grown_mm": g, "source": source,
    }


def avatar_params(client: ClientProfile, last_cut: dict | None, hair_texture: str | None) -> dict:
    hp = _get(client.visagismo_profile, "hair_physical_metrics") or {}
    texture = hair_texture or _PATTERN_TO_TEXTURE.get(hp.get("hair_pattern_shape") or "") or "liso"
    return {
        "hair_texture": texture,
        "hair_texture_known": bool(hair_texture or hp.get("hair_pattern_shape")),
        "hair_color": client.hair_color if client.hair_color in HAIR_COLORS else None,
        "hair_density": hp.get("hair_density") or "medium",
        "hair_thickness": hp.get("hair_texture_thickness"),
        "hairline": hp.get("frontal_hairline_shape") or "linear_straight",
        "length": current_length(client, last_cut),
        "morphs": morph_weights(client),
    }
