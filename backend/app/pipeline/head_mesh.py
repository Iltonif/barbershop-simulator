"""
Etapa 4 del pipeline: geometría de la cabeza y dirección de crecimiento.

En vez de depender de un maniquí físico marcado a mano por cada cliente
(no escala), este módulo empieza con un patrón de crecimiento ESTÁNDAR
sobre zonas de la cabeza (corona, flequillo, laterales, nuca), derivado de
los landmarks faciales. El barbero puede sustituirlo por completo dibujando
a mano el suyo sobre la cabeza del cliente — ver `apply_custom_growth_map`
y la herramienta visual en `frontend/growth-map.html`.

⚠️ PLACEHOLDER de geometría: no reconstruye un mesh 3D real todavía. Hay
DOS espacios de coordenadas distintos conviviendo aquí, y es importante no
confundirlos:

1. El mapa POR DEFECTO (`build_default_growth_map`) se calcula a partir de
   los landmarks 2D de la foto (`face_analysis.py`) — es un mapa plano
   sobre la imagen, con la profundidad (z) fijada a 0.0 porque no hay
   ninguna estimación real de profundidad.
2. El mapa DIBUJADO A MANO (`apply_custom_growth_map`) viene de
   `frontend/growth-map.html`, donde el barbero dibuja sobre una cabeza 3D
   genérica (Three.js, geometría propia, sin depender de ningún modelo
   externo por temas de licencia). Ahí sí hay x, y, z reales — son las
   coordenadas de un punto sobre la superficie de ESA cabeza genérica, no
   sobre la foto del cliente.

Estos dos espacios NO están todavía conciliados entre sí (por eso ambos
usan la misma estructura `GrowthStroke`/`Whorl` de 3 coordenadas, con z=0
en el caso 1): hacerlo de verdad requeriría una reconstrucción 3D real de
la cabeza del cliente a partir de su foto (modelos de "single-image 3D
face reconstruction" / 3DMM), que es justo el trabajo pendiente de verdad
aquí — mientras no exista, el mapa dibujado a mano queda guardado como la
intención del barbero, para cuando exista esa reconstrucción.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GrowthStroke:
    """Un trazo que indica la dirección de crecimiento del pelo en una
    zona: de `start` a `end`. Ver nota de coordenadas arriba — z=0.0 en el
    mapa por defecto (2D sobre la foto), z real en el mapa dibujado a mano
    (3D sobre la cabeza genérica).

    `zone`: nombre de una de las zonas fijas de HEAD_ZONES (ver más abajo),
    o None para un trazo "libre" antiguo, de antes de que
    `frontend/growth-map.html` pasara del dibujo a mano alzada al sistema
    de una única dirección dominante por zona — se mantiene el campo
    opcional en vez de migrar los datos ya guardados, así que un perfil de
    cliente guardado con la versión anterior de la herramienta se sigue
    leyendo sin romperse (simplemente no se puede editar zona a zona hasta
    que el barbero vuelva a dibujar su mapa)."""

    start: tuple[float, float, float]
    end: tuple[float, float, float]
    zone: str | None = None


@dataclass
class Whorl:
    """Un remolino: un punto de la cabeza donde el pelo gira en espiral
    en vez de crecer en una dirección fija."""

    center: tuple[float, float, float]
    rotation: str  # "horario" | "antihorario"


@dataclass
class HeadGrowthMap:
    strokes: list[GrowthStroke] = field(default_factory=list)
    whorls: list[Whorl] = field(default_factory=list)
    has_manual_override: bool = False


# Patrón de crecimiento por defecto: la mayoría de cabezas comparten un
# remolino en la coronilla que gira en sentido horario (hay una minoría
# antihorario) y el pelo de la nuca crece hacia abajo/afuera. Expresado
# como trazos cortos alrededor de un centro de cabeza aproximado, en
# coordenadas normalizadas relativas a ese mismo centro y al ancho de cara.
_DEFAULT_STROKE_TEMPLATE = [
    # (nombre solo para comentario, dx_centro, dy_centro, direction_deg, longitud_ratio)
    ("corona", 0.0, -0.05, 0, 0.10),
    ("flequillo", 0.0, 0.05, 90, 0.14),
    ("laterales_izq", -0.18, 0.0, 180, 0.16),
    ("laterales_der", 0.18, 0.0, 0, 0.16),
    ("nuca", 0.0, 0.20, 270, 0.14),
]

# Zonas fijas para el sistema de "una dirección dominante por zona" de
# frontend/growth-map.html (sustituye al dibujo de flechas libres a mano
# alzada de antes). Los nombres coinciden a propósito con
# _DEFAULT_STROKE_TEMPLATE de arriba, para que el mapa por defecto y el
# dibujado a mano por el barbero hablen de las mismas zonas.
#
# `direction` es un vector (no normalizado) desde el centro de la cabeza
# 3D genérica hacia fuera, en el mismo sistema de ejes que
# frontend/assets/head.glb (x: izquierda/derecha, y: abajo/arriba, z:
# detrás/delante). growth-map.html lo usa para lanzar un rayo desde fuera
# de la cabeza hacia el centro y así colocar el marcador de cada zona
# sobre la superficie real del modelo cargado, sea cual sea su forma
# exacta (incluida la escalada por cliente de head_shape.py) -- en vez de
# fijar una coordenada 3D absoluta, que se desencuadraría en cuanto la
# cabeza cambia de proporciones.
#
# IMPORTANTE: estos mismos nombres y direcciones están duplicados en
# frontend/growth-map.html (HEAD_ZONES en el <script>) porque no hay
# ningún mecanismo en este proyecto para compartir una constante entre
# Python y JS puro sin build step -- si se cambia aquí, hay que cambiarlo
# también allí.
HEAD_ZONES = [
    # (nombre, etiqueta para el barbero, direction_xyz)
    #
    # Estos vectores se ajustaron a mano (viendo capturas del maniquí desde
    # varios angulos) porque los originales, aunque simetricos, caian en
    # zonas equivocadas de la piel: "flequillo" apuntaba entre las cejas y
    # "laterales" apuntaba a la mejilla/mandibula en vez de al cuero
    # cabelludo por encima de la oreja. Si se cambia el maniquí (otro
    # base_mesh o otra normalizacion de altura), hay que volver a revisar
    # estos valores a ojo, no son una formula general.
    ("corona", "Coronilla", (0.0, 1.0, -0.1)),
    ("flequillo", "Flequillo / frente", (0.0, 0.9, 0.75)),
    ("laterales_izq", "Lateral izquierdo", (-0.9, 0.4, 0.15)),
    ("laterales_der", "Lateral derecho", (0.9, 0.4, 0.15)),
    ("nuca", "Nuca", (0.0, -0.05, -1.0)),
]


def _point_from_angle(origin: np.ndarray, direction_deg: float, length: float) -> tuple[float, float]:
    """0° = hacia arriba de la imagen (y decreciente), sentido horario."""
    rad = np.radians(direction_deg)
    dx = length * np.sin(rad)
    dy = -length * np.cos(rad)
    return (float(origin[0] + dx), float(origin[1] + dy))


def _as_3d(point_2d: tuple[float, float]) -> tuple[float, float, float]:
    """El mapa por defecto es 2D (sobre la foto): se completa con z=0.0
    para encajar en la misma estructura que el mapa dibujado a mano."""
    return (point_2d[0], point_2d[1], 0.0)


def build_default_growth_map(landmarks: np.ndarray, image_shape: tuple[int, int]) -> HeadGrowthMap:
    """Construye un mapa de crecimiento estándar posicionado sobre la
    cabeza detectada, usando los landmarks de `face_analysis.py` como
    referencia de escala y posición, ya normalizado a 0.0-1.0.

    `landmarks` usa el esquema de 68 puntos (dlib/iBUG) de
    `face_analysis.py` — jaw: 0-16, cejas: 17-26, entrecejo: 27, barbilla: 8.
    Ese esquema no tiene puntos de frente, así que `top_of_head` es una
    extrapolación hacia arriba desde las cejas (misma idea que
    `_estimate_hairline` en face_analysis.py).
    """
    h, w = image_shape[:2]
    eyebrow_center = landmarks[17:27].mean(axis=0)
    face_height = float(np.linalg.norm(landmarks[27] - landmarks[8]))  # entrecejo -> barbilla
    top_of_head = eyebrow_center + np.array([0, -face_height * 0.9])
    face_width = float(np.linalg.norm(landmarks[0] - landmarks[16]))  # jaw izq/der

    strokes = []
    for name, dx_ratio, dy_ratio, direction_deg, length_ratio in _DEFAULT_STROKE_TEMPLATE:
        origin = top_of_head + np.array([dx_ratio * face_width, dy_ratio * face_width])
        end = _point_from_angle(origin, direction_deg, length_ratio * face_width)
        strokes.append(
            GrowthStroke(
                start=_as_3d((float(origin[0] / w), float(origin[1] / h))),
                end=_as_3d((end[0] / w, end[1] / h)),
                zone=name,
            )
        )

    whorl_center = top_of_head / np.array([w, h])
    whorls = [
        Whorl(center=_as_3d((float(whorl_center[0]), float(whorl_center[1]))), rotation="horario")
    ]

    return HeadGrowthMap(strokes=strokes, whorls=whorls, has_manual_override=False)


def apply_custom_growth_map(
    growth_map: HeadGrowthMap,
    strokes: list[dict],
    whorls: list[dict],
) -> HeadGrowthMap:
    """Sustituye el mapa de crecimiento por uno dibujado a mano por el
    barbero (ver `frontend/growth-map.html` y
    `PATCH /api/clients/{id}/growth-map`).

    `strokes` y `whorls` llegan como listas de dicts ya deserializadas del
    JSON guardado en el perfil del cliente (`ClientProfile.custom_growth_map`),
    con las claves `x1,y1,z1,x2,y2,z2` para cada trazo y `x,y,z,rotation`
    para cada remolino — coordenadas 3D reales sobre la cabeza genérica de
    `frontend/growth-map.html`, no normalizadas 0.0-1.0 como el mapa por
    defecto (ver nota de coordenadas arriba).

    `z1`/`z2`/`z` se leen con `.get(..., 0.0)` en vez de indexado directo:
    perfiles guardados con la versión anterior de `growth-map.html` (canvas
    2D, antes de pasar a Three.js) solo tenían `x1,y1,x2,y2`/`x,y`, sin
    profundidad. Sin este `.get()` esos registros antiguos revientan aquí
    con `KeyError` en cuanto se vuelve a generar una simulación para ese
    cliente. No hace falta migrar esos datos: simplemente se interpretan
    como si estuvieran sobre el plano z=0.0 de la cabeza 3D actual.
    """
    new_strokes = [
        GrowthStroke(
            start=(s["x1"], s["y1"], s.get("z1", 0.0)),
            end=(s["x2"], s["y2"], s.get("z2", 0.0)),
            # .get(): perfiles guardados antes del sistema de zonas no
            # traen esta clave -- se leen igual, simplemente sin zona
            # asociada (ver docstring de GrowthStroke).
            zone=s.get("zone"),
        )
        for s in strokes
    ]
    new_whorls = [
        Whorl(center=(w["x"], w["y"], w.get("z", 0.0)), rotation=w["rotation"]) for w in whorls
    ]

    growth_map.strokes = new_strokes
    growth_map.whorls = new_whorls
    growth_map.has_manual_override = True
    return growth_map
