"""
Interpreta el mapa de crecimiento dibujado en `frontend/growth-map.html`
(flechas trazadas con el dedo sobre el maniquí + remolinos) y lo convierte
en datos que entiende el recomendador: en qué zona está cada remolino,
hacia dónde crece el pelo en la frente, la coronilla, los laterales y la
nuca, y de qué lado cae la raya de forma natural.

Coordenadas: las que guarda growth-map.html son de MUNDO en la escena de
Three.js, con el maniquí (`frontend/assets/head.glb`) escalado x2 y bajado
0,3 (`headGroup.scale = 2`, `position.y = -0.3`). Aquí se pasan al espacio
del .glb y se clasifican por la DIRECCIÓN desde el centro de la cabeza, no
por la posición exacta, para que sigan valiendo si el peluquero ajustó el
maniquí al ancho/alto de la cara del cliente (`head_shape.py`), que estira
la cabeza hasta un ±15 %.

Ejes del maniquí: +x = lado IZQUIERDO del cliente (su oreja izquierda),
+y = arriba, +z = hacia la cara.

Las constantes de centro y límites de zona se ajustaron a ojo pintando las
zonas sobre el maniquí v4 (ver CLAUDE.md, "Maniquí v4"). Si se cambia el
maniquí, hay que revisarlas.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

WORLD_SCALE = 2.0
WORLD_OFFSET = np.array([0.0, -0.3, 0.0])
HEAD_CENTER = np.array([0.0, 0.14, -0.02])  # espacio del .glb

ZONE_LABELS = {
    "frente": "Frente",
    "arriba": "Parte de arriba",
    "coronilla": "Coronilla",
    "lateral_izq": "Lateral izquierdo",
    "lateral_dcha": "Lateral derecho",
    "nuca": "Nuca",
}

DIRECTION_LABELS = {
    "delante": "hacia delante",
    "atras": "hacia atrás",
    "arriba": "hacia arriba",
    "abajo": "hacia abajo",
    "izquierda": "hacia su izquierda",
    "derecha": "hacia su derecha",
}


def to_local(point) -> np.ndarray:
    return (np.asarray(point, dtype=float) - WORLD_OFFSET) / WORLD_SCALE


def zone_of(point_world) -> str:
    """Zona de la cabeza de un punto (coordenadas de mundo)."""
    u = to_local(point_world) - HEAD_CENTER
    n = np.linalg.norm(u)
    if n < 1e-9:
        return "arriba"
    x, y, z = u / n
    if x > 0.62 and y < 0.72:
        return "lateral_izq"
    if x < -0.62 and y < 0.72:
        return "lateral_dcha"
    if z > 0.42:
        return "frente"
    if y > 0.8 and z > -0.12:
        return "arriba"
    if y > 0.12:
        return "coronilla"
    return "nuca"


def _surface_normal(point_world) -> np.ndarray:
    u = to_local(point_world) - HEAD_CENTER
    return u / max(np.linalg.norm(u), 1e-9)


def direction_of(start_world, end_world) -> str:
    """Dirección dominante de un tramo, quitando la componente que se
    separa de la cabeza (la normal aproximada desde el centro)."""
    v = np.asarray(end_world, float) - np.asarray(start_world, float)
    nrm = _surface_normal((np.asarray(start_world, float) + np.asarray(end_world, float)) / 2)
    v = v - nrm * v.dot(nrm)
    ax = int(np.argmax(np.abs(v)))
    sign = v[ax] >= 0
    return [("izquierda", "derecha"), ("arriba", "abajo"), ("delante", "atras")][ax][0 if sign else 1]


def _stroke_points(stroke: dict) -> list[np.ndarray]:
    pts = stroke.get("points")
    if pts and len(pts) >= 2:
        return [np.asarray(p, float) for p in pts]
    return [np.array([stroke["x1"], stroke["y1"], stroke.get("z1", 0.0)]),
            np.array([stroke["x2"], stroke["y2"], stroke.get("z2", 0.0)])]


@dataclass
class StrokeInfo:
    zone: str
    direction: str


@dataclass
class WhorlInfo:
    zone: str
    rotation: str  # "horario" | "antihorario"


@dataclass
class GrowthSummary:
    strokes: list[StrokeInfo] = field(default_factory=list)
    whorls: list[WhorlInfo] = field(default_factory=list)
    # Dirección dominante por zona (solo zonas con flechas).
    zone_direction: dict[str, str] = field(default_factory=dict)
    natural_part: str | None = None      # "izquierda" | "derecha"
    natural_part_source: str | None = None

    @property
    def whorl_zones(self) -> list[str]:
        return [w.zone for w in self.whorls]

    @property
    def crown_whorls(self) -> int:
        return sum(z in ("coronilla", "arriba") for z in self.whorl_zones)

    @property
    def front_whorl(self) -> bool:
        return "frente" in self.whorl_zones

    @property
    def nape_whorl(self) -> bool:
        return "nuca" in self.whorl_zones

    @property
    def front_growth(self) -> str | None:
        """Cómo crece el pelo de la frente: "delante" (cae sobre la frente),
        "atras" (hacia arriba/atrás), "lado_izquierda"/"lado_derecha"."""
        d = self.zone_direction.get("frente")
        if d in ("delante", "abajo"):
            return "delante"
        if d in ("atras", "arriba"):
            return "atras"
        if d in ("izquierda", "derecha"):
            return "lado_" + d
        return None

    @property
    def nape_growth_irregular(self) -> bool:
        """En la nuca el pelo crece normalmente hacia abajo; hacia un lado o
        hacia arriba es lo que obliga a adaptar la línea de la nuca."""
        return self.zone_direction.get("nuca") in ("izquierda", "derecha", "arriba")

    def lines(self) -> list[str]:
        """Resumen legible, una línea por dato (para la web)."""
        out = []
        for w in self.whorls:
            out.append(f"Remolino {w.rotation} en {ZONE_LABELS[w.zone].lower()}")
        for zone, d in self.zone_direction.items():
            if zone == "frente" and d in ("delante", "abajo"):
                text = "cae hacia la frente"
            elif zone == "frente" and d in ("atras", "arriba"):
                text = "hacia atrás"
            else:
                text = DIRECTION_LABELS[d]
            out.append(f"{ZONE_LABELS[zone]}: {text}")
        if self.natural_part:
            out.append(f"Raya natural: a su {self.natural_part}")
        return out


def summarize(growth_map: dict | None) -> GrowthSummary:
    """`growth_map` es `ClientProfile.custom_growth_map` tal cual
    ({"strokes": [...], "whorls": [...]}), o el payload de
    `PATCH .../growth-map`."""
    summary = GrowthSummary()
    if not growth_map:
        return summary
    votes: dict[str, dict[str, float]] = {}
    for stroke in growth_map.get("strokes") or []:
        pts = _stroke_points(stroke)
        first_dir = None
        for a, b in zip(pts, pts[1:]):
            seg = float(np.linalg.norm(b - a))
            if seg < 1e-6:
                continue
            zone, d = zone_of((a + b) / 2), direction_of(a, b)
            votes.setdefault(zone, {}).setdefault(d, 0.0)
            votes[zone][d] += seg
            first_dir = first_dir or (zone_of(pts[0]), direction_of(pts[0], pts[-1]))
        if first_dir:
            summary.strokes.append(StrokeInfo(zone=first_dir[0], direction=first_dir[1]))
    order = list(ZONE_LABELS)
    for zone in sorted(votes, key=order.index):
        summary.zone_direction[zone] = max(votes[zone].items(), key=lambda kv: kv[1])[0]

    for w in growth_map.get("whorls") or []:
        summary.whorls.append(WhorlInfo(zone=zone_of((w["x"], w["y"], w.get("z", 0.0))),
                                        rotation=w.get("rotation", "horario")))

    # Raya natural. Regla habitual de barbería: remolino de la coronilla en
    # sentido horario -> raya a la izquierda; antihorario -> a la derecha
    # (Fellow Barber, Man For Himself). Si no hay remolino marcado ahí, se
    # mira hacia qué lado va el pelo de la frente: la raya queda en el lado
    # contrario.
    crown = [w for w in summary.whorls if w.zone in ("coronilla", "arriba")]
    if len(crown) == 1:
        summary.natural_part = "izquierda" if crown[0].rotation == "horario" else "derecha"
        summary.natural_part_source = "remolino"
    elif summary.front_growth in ("lado_izquierda", "lado_derecha"):
        summary.natural_part = "derecha" if summary.front_growth == "lado_izquierda" else "izquierda"
        summary.natural_part_source = "frente"
    return summary
