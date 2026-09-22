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
    # Consentimiento SEPARADO para enviar el perfil de visagismo a un
    # servicio externo de pago (API de Claude, ver
    # `app/pipeline/visagismo_ai_advisor.py`) -- distinto de
    # `consent_history`/`consent_model_improvement`/`consent_save_photo`
    # porque implica una finalidad de tratamiento distinta (transferencia
    # a un tercero), no solo guardar el dato en el propio servidor.
    consent_ai_analysis: bool
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
    phone: str | None = None
    consent_simulation: bool = False
    simulation_photo_path: str | None = None
    liked_styles: list[str] = field(default_factory=list)
    hair_color: str | None = None
    current_length: dict | None = None      # {"top", "sides", "back"} en mm, puesto a mano
    current_length_at: str | None = None


@dataclass
class WaitingEntry:
    id: str
    client_id: str
    created_at: str
    updated_at: str
    status: str
    requested_history_id: str | None = None


@dataclass
class HaircutRecord:
    id: str
    client_id: str
    created_at: str
    style_id: str | None
    style_name: str | None
    notes: str | None
    photo_path: str | None


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
