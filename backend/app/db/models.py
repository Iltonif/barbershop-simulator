"""
Dataclasses del dominio de cliente/historial.

No confundir con los modelos Pydantic de `app/api/schemas.py`: aquellos
son el contrato HTTP (lo que entra/sale de la API), estos son el modelo
interno que usa `repository.py`.
"""

from dataclasses import dataclass, field


@dataclass
class ClientProfile:
    id: str
    created_at: str
    display_name: str | None
    consent_history: bool
    consent_model_improvement: bool
    consent_save_photo: bool
    hair_texture_override: str | None = None
    face_shape_override: str | None = None
    # {"strokes": [{"x1","y1","z1","x2","y2","z2"}, ...], "whorls": [{"x","y","z","rotation"}, ...]},
    # coordenadas normalizadas 0.0-1.0. None si el barbero no ha dibujado
    # nada todavía para este cliente (se usa el patrón por defecto).
    custom_growth_map: dict | None = None
    # Perfil extendido de visagismo (morfología craneal/facial, métricas
    # físicas del pelo, estilo de vida) -- ver `VisagismoProfileIn` en
    # `app/api/schemas.py` para la forma exacta y `app/pipeline/
    # visagismo_rules.py` para cómo se usa. None si el barbero no lo ha
    # rellenado todavía (ninguna regla de visagismo se activa en ese caso).
    visagismo_profile: dict | None = None
    notes: str | None = None


@dataclass
class Visit:
    id: str
    client_id: str
    created_at: str
    style_id: str | None
    detected_hair_texture: str | None
    detected_face_shape: str | None
    used_hair_texture: str | None
    warnings: list[str] = field(default_factory=list)
    photo_path: str | None = None
