"""Modelos Pydantic de request/response de la API."""

from pydantic import BaseModel


class StyleOut(BaseModel):
    id: str
    name: str
    description: str
    length_top_mm: int
    length_sides_mm: int
    length_back_mm: int
    fade_type: str
    suitable_hair_types: list[str]
    reference_image: str | None = None
    length_category: str | None = None
    source: str | None = None
    style_family: str | None = None


class StyleRecommendationOut(BaseModel):
    """Un corte recomendado, con una nota opcional (p. ej. aviso de
    remolinos) explicando por qué aparece más abajo en la lista."""

    style: StyleOut
    note: str | None = None


class RecommendationsOut(BaseModel):
    client_id: str
    hair_texture: str
    whorl_count: int
    face_shape: str | None = None
    recommendations: list[StyleRecommendationOut]


class SimulationResponse(BaseModel):
    style_id: str
    detected_hair_texture: str
    detected_face_shape: str
    warnings: list[str] = []
    # La imagen generada se devuelve como base64 para simplificar el MVP.
    # TODO: para producción, considerar devolver una URL firmada de corta
    # duración en vez de embeber la imagen en el JSON.
    image_base64: str | None = None


class ClientCreateIn(BaseModel):
    """Alta de un perfil de cliente. `consent_history` es obligatorio: sin
    él, la API rechaza la creación (ver `clients_routes.create_client`).
    `consent_model_improvement` y `consent_save_photo` son opcionales y
    van por separado a propósito — cada finalidad de tratamiento necesita
    su propio consentimiento, no vale uno genérico para todo."""

    display_name: str | None = None
    consent_history: bool
    consent_model_improvement: bool = False
    consent_save_photo: bool = False
    notes: str | None = None


class ClientOut(BaseModel):
    id: str
    created_at: str
    display_name: str | None = None
    consent_history: bool
    consent_model_improvement: bool
    consent_save_photo: bool
    hair_texture_override: str | None = None
    face_shape_override: str | None = None
    custom_growth_map: dict | None = None
    notes: str | None = None


class VisitOut(BaseModel):
    id: str
    client_id: str
    created_at: str
    style_id: str | None = None
    detected_hair_texture: str | None = None
    detected_face_shape: str | None = None
    used_hair_texture: str | None = None
    warnings: list[str] = []
    photo_path: str | None = None


class ClientWithHistoryOut(ClientOut):
    visits: list[VisitOut] = []


class HairTypeOverrideIn(BaseModel):
    texture: str  # "liso" | "ondulado" | "rizado" | "afro"


class FaceShapeOverrideIn(BaseModel):
    face_shape: str  # "ovalada" | "redonda" | "cuadrada" | "alargada"


class GrowthStrokeIn(BaseModel):
    """Un trazo dibujado a mano sobre la cabeza 3D de
    `frontend/growth-map.html`: de (x1,y1,z1) a (x2,y2,z2), coordenadas
    reales sobre la superficie de esa cabeza genérica (no normalizadas
    0.0-1.0 — ver la nota de coordenadas en `head_mesh.py`)."""

    x1: float
    y1: float
    z1: float
    x2: float
    y2: float
    z2: float


class WhorlIn(BaseModel):
    x: float
    y: float
    z: float
    rotation: str  # "horario" | "antihorario"


class CustomGrowthMapIn(BaseModel):
    """Payload de `PATCH /api/clients/{id}/growth-map`: el estado COMPLETO
    del lienzo de `frontend/growth-map.html`, no un delta — sustituye
    cualquier mapa guardado anteriormente para ese cliente."""

    strokes: list[GrowthStrokeIn] = []
    whorls: list[WhorlIn] = []
