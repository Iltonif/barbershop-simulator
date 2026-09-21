"""Modelos Pydantic de request/response de la API."""

from pydantic import BaseModel, Field, computed_field


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


class ReasonOut(BaseModel):
    """Por qué encaja o no un corte: etiqueta corta para la tarjeta y la
    explicación completa (ver app/pipeline/rule_effects.py)."""

    label: str
    detail: str


class StyleRecommendationOut(BaseModel):
    """Un corte recomendado. `reasons` = por qué encaja (sube en la lista);
    `warnings` = qué no encaja (baja). `note` junta los avisos en un texto,
    como antes."""

    style: StyleOut
    note: str | None = None
    reasons: list[ReasonOut] = []
    warnings: list[ReasonOut] = []


class RecommendationsOut(BaseModel):
    client_id: str
    # None = nadie lo ha dicho todavía (no se filtra por tipo de pelo).
    hair_texture: str | None = None
    hair_texture_source: str | None = None  # "peluquero" | "cliente"
    whorl_count: int
    face_shape: str | None = None
    recommendations: list[StyleRecommendationOut]
    # Consejo de barba según mentón/mandíbula (no depende del corte).
    beard_advice: list[ReasonOut] = []
    liked_styles: list[str] = []


class SimulationResponse(BaseModel):
    style_id: str
    detected_hair_texture: str
    detected_face_shape: str
    warnings: list[str] = []
    # La imagen generada se devuelve como base64 para simplificar el MVP.
    # TODO: para producción, considerar devolver una URL firmada de corta
    # duración en vez de embeber la imagen en el JSON.
    image_base64: str | None = None
    # Proveedor que generó la imagen (ver haircut_editor.py); None = sin
    # simulación real (se devuelve la foto original).
    provider: str | None = None


class SimulationProviderOut(BaseModel):
    id: str
    label: str
    company: str


class ClientCreateIn(BaseModel):
    """Alta de un perfil de cliente. `consent_history` es obligatorio: sin
    él, la API rechaza la creación (ver `clients_routes.create_client`).
    `consent_model_improvement`, `consent_save_photo` y `consent_ai_analysis`
    son opcionales y van por separado a propósito — cada finalidad de
    tratamiento necesita su propio consentimiento, no vale uno genérico
    para todo. `consent_ai_analysis` es el más delicado de los tres: sin
    él no se puede generar el informe de visagismo por IA (ver
    `app/pipeline/visagismo_ai_advisor.py`), porque esa finalidad implica
    enviar el perfil de visagismo a un servicio externo (API de Claude),
    no solo guardarlo en el propio servidor."""

    display_name: str | None = None
    consent_history: bool
    consent_model_improvement: bool = False
    consent_save_photo: bool = False
    consent_ai_analysis: bool = False
    notes: str | None = None


class FacialHorizontalZonesRatioIn(BaseModel):
    """Proporción de cada tercio horizontal de la cara, a ojo del barbero
    (no hay ninguna medición automática de esto todavía)."""

    intellectual_zone_forehead: str | None = None  # "proportional" | "prominent" | "narrow"
    affective_zone_mid_face: str | None = None  # "proportional" | "prominent" | "narrow"
    sensitive_zone_jaw_chin: str | None = None  # "proportional" | "prominent" | "narrow"


class FacialFeaturesProfileIn(BaseModel):
    profile_type: str | None = None  # "straight" | "convex_prominent_nose" | "concave"
    ears_projection: str | None = None  # "flat" | "prominent_protruding"
    neck_proportions: str | None = None  # "short_thick" | "long_thin" | "proportional"
    eye_spacing: str | None = None  # "close_set" | "proportional" | "wide_set"
    eyebrow_type: str | None = None  # "straight_low" | "arched" | "prominent_ridge"
    # Campos añadidos para el análisis automático de rasgos faciales (ver
    # `app/pipeline/facial_traits_analysis.py`), pero también rellenables
    # a mano igual que el resto de este esquema.
    eye_symmetry: str | None = None  # "symmetric" | "asymmetric"
    eye_symmetry_percent: float | None = None  # diferencia de apertura entre ojos, 0-100
    has_glasses: bool | None = None  # solo informativo (no afecta a las reglas de recomendación)
    # Rasgos de perfil (a mano, con la guía de visagismo; ver trait_rules.py).
    chin_projection: str | None = None  # "retruded" | "balanced" | "prominent"
    jawline_definition: str | None = None  # "defined" | "soft"
    # Cualquier irregularidad que no encaje en un campo estructurado de
    # arriba (p.ej. una cicatriz, una asimetría de nariz/orejas puntual):
    # texto libre en vez de intentar catalogar cada caso posible.
    detected_anomalies_notes: str | None = None


class AnatomicalMetricsIn(BaseModel):
    cranial_morphology: str | None = None  # "mesocephalic" | "brachycephalic" | "dolichocephalic"
    facial_geometry: str | None = None  # "oval" | "square" | "round" | "rectangular_elongated" | "diamond" | "triangle" | "heart"
    facial_horizontal_zones_ratio: FacialHorizontalZonesRatioIn = FacialHorizontalZonesRatioIn()
    facial_features_profile: FacialFeaturesProfileIn = FacialFeaturesProfileIn()


class GrowthDirectionsCowlicksIn(BaseModel):
    """No confundir con los remolinos reales dibujados a mano en
    `frontend/growth-map.html` (`ClientProfile.custom_growth_map`, con
    coordenadas 3D exactas sobre la cabeza) — esto es una descripción
    rápida en la ficha del cliente, sin coordenadas, pensada para rellenar
    aunque todavía no se haya hecho el mapa de crecimiento detallado."""

    crown_cowlick: str | None = None  # "clockwise" | "counter_clockwise" | "double" | "strong_rebellion"
    fringe_direction: str | None = None  # "forward" | "lateral_left" | "lateral_right" | "cowlick_present"


class HairPhysicalMetricsIn(BaseModel):
    hair_density: str | None = None  # "low_thinning" | "medium" | "high_dense"
    hair_texture_thickness: str | None = None  # "fine" | "medium" | "coarse_thick" | "afro"
    hair_pattern_shape: str | None = None  # "straight" | "wavy" | "curly" | "coily"
    growth_directions_cowlicks: GrowthDirectionsCowlicksIn = GrowthDirectionsCowlicksIn()
    frontal_hairline_shape: str | None = None  # "linear_straight" | "m_shaped_receding" | "widows_peak" | "high_forehead"


class StylingProductsUsageIn(BaseModel):
    uses_product: bool = False
    preferred_finish: str | None = None  # "matte_natural" | "shiny_wet" | "none"


class LifestyleAndPreferencesIn(BaseModel):
    daily_maintenance_commitment: str | None = None  # "zero_minutes" | "low_1_5_mins" | "medium_5_15_mins" | "high_requires_blowdryer"
    styling_products_usage: StylingProductsUsageIn = StylingProductsUsageIn()
    barbershop_visit_frequency_days: int | None = None
    professional_social_environment: str | None = None  # "corporate_formal" | "creative_artistic" | "casual_sporty"
    beard_preference: str | None = None  # "clean_shaven" | "stubble_short" | "full_long_volume" | "sharp_lined"


class VisagismoProfileIn(BaseModel):
    """Perfil extendido de visagismo: morfología craneal/facial, métricas
    físicas del pelo y estilo de vida. Todo opcional -- se puede rellenar
    poco a poco, igual que `hair_texture_override`/`face_shape_override`.
    Se guarda tal cual (como dict) en `ClientProfile.visagismo_profile` y
    lo consume `app/pipeline/visagismo_rules.py` para matizar (nunca
    descartar, ver el docstring de ese módulo) las recomendaciones de
    `recommend_styles`.

    Nota RGPD: esto es un desglose bastante más fino de datos biométricos
    (geometría facial, morfología craneal...) que `face_shape_override` /
    `hair_texture_override` -- se guarda en la misma fila de `clients` y
    por tanto bajo el mismo `consent_history` que el resto del perfil (ver
    sección RGPD de CLAUDE.md), no hace falta un consentimiento aparte,
    pero conviene tenerlo en cuenta al decidir qué tan detallado rellenar
    esto para un cliente real."""

    anatomical_metrics: AnatomicalMetricsIn = AnatomicalMetricsIn()
    hair_physical_metrics: HairPhysicalMetricsIn = HairPhysicalMetricsIn()
    lifestyle_and_preferences: LifestyleAndPreferencesIn = LifestyleAndPreferencesIn()


class ClientOut(BaseModel):
    id: str
    created_at: str
    display_name: str | None = None
    consent_history: bool
    consent_model_improvement: bool
    consent_save_photo: bool
    consent_ai_analysis: bool
    hair_texture_override: str | None = None
    face_shape_override: str | None = None
    custom_growth_map: dict | None = None
    visagismo_profile: dict | None = None
    notes: str | None = None
    phone: str | None = None
    consent_simulation: bool = False
    liked_styles: list[str] = []
    # La ruta en disco no sale nunca en la API, solo si hay foto.
    simulation_photo_path: str | None = Field(default=None, exclude=True)

    @computed_field
    @property
    def has_simulation_photo(self) -> bool:
        return bool(self.simulation_photo_path)


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
    """Un trazo sobre la cabeza 3D de `frontend/growth-map.html`: de
    (x1,y1,z1) a (x2,y2,z2), coordenadas reales sobre la superficie de esa
    cabeza genérica (no normalizadas 0.0-1.0 — ver la nota de coordenadas
    en `head_mesh.py`).

    `zone`: nombre de zona de `head_mesh.HEAD_ZONES` (p.ej. "corona"),
    o None para un trazo libre de antes del sistema de zonas — ver
    `head_mesh.GrowthStroke`."""

    x1: float
    y1: float
    z1: float
    x2: float
    y2: float
    z2: float
    zone: str | None = None


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


class HeadShapeOut(BaseModel):
    """Respuesta de `POST /api/growth-map/head-shape`: factores de
    escala por eje para ajustar el maniquí 3D de growth-map.html al
    ancho/alto de cara reales del cliente (ver `pipeline/head_shape.py`
    para el porqué no se ajusta también la profundidad)."""

    scale_x: float
    scale_y: float
    width_to_height: float


class VisagismoAIReportOut(BaseModel):
    """Respuesta de `POST /api/clients/{id}/visagismo-ai-report`. No hay un
    `AIReportIn` porque el endpoint no recibe payload -- toma los datos ya
    guardados del cliente (`visagismo_profile`, overrides, remolinos), ver
    `app/pipeline/visagismo_ai_advisor.build_user_message`.

    Deliberadamente NO se persiste en BD (a diferencia de `ClientOut`):
    cada llamada genera un informe nuevo bajo demanda. Ver nota "pendiente"
    en CLAUDE.md sobre guardar historial de informes en el futuro."""

    client_id: str
    model: str
    report: str


class VisagismoAutoAnalysisOut(BaseModel):
    """Respuesta de `POST /api/clients/{id}/visagismo-auto-analysis`: el
    resultado de analizar las 3 fotos guiadas (frontal + perfil izq. +
    perfil dcha.) con `app/pipeline/facial_traits_analysis.py`, ya
    fusionado y guardado en el `visagismo_profile` del cliente (por eso
    se devuelve el `ClientOut` completo, igual que el resto de endpoints
    `.../visagismo-profile` y `.../growth-map`).

    `warnings` recoge lo que NO se ha podido detectar (p.ej. una foto sin
    cara reconocible) para que el barbero sepa qué campos conviene
    revisar/rellenar a mano. `detected_anomalies_notes` es el resumen en
    texto de asimetrías detectadas (con su porcentaje), pensado para
    mostrarse tal cual en la ficha del cliente."""

    client: ClientOut
    warnings: list[str] = []
    detected_anomalies_notes: str | None = None


# --- Flujo "cliente esperando en el sillón" (session_routes.py) -----------

class BarberLoginIn(BaseModel):
    pin: str


class ClientRegisterIn(BaseModel):
    """Alta hecha por el propio cliente en la tablet o en su móvil. Da él
    mismo los consentimientos (antes los marcaba el peluquero)."""

    display_name: str
    phone: str
    consent_history: bool
    consent_save_photo: bool = False
    consent_simulation: bool = False


class ClientLoginIn(BaseModel):
    phone: str


class ConsentsIn(BaseModel):
    consent_save_photo: bool | None = None
    consent_simulation: bool | None = None


class LikesIn(BaseModel):
    style_ids: list[str]


class StoredPhotoSimulationIn(BaseModel):
    style_id: str
    provider: str | None = None


class QuestionnaireIn(BaseModel):
    """Respuestas de "Mi perfil" (frontend/cuestionario.html). Todas
    opcionales: "No lo sé" no manda nada y no borra lo que hubiera."""

    hair_pattern_shape: str | None = None  # straight | wavy | curly | coily
    face_shape: str | None = None  # ovalada | redonda | cuadrada | alargada
    frontal_hairline_shape: str | None = None
    daily_maintenance_commitment: str | None = None
    barbershop_visit_frequency_days: int | None = None
    beard_preference: str | None = None


class HaircutOut(BaseModel):
    """Un corte del historial del cliente. La foto se pide aparte
    (`.../history/{id}/photo`); aquí solo si hay."""

    id: str
    created_at: str
    style_id: str | None = None
    style_name: str | None = None
    notes: str | None = None
    reference_image: str | None = None  # foto del catálogo, si es un corte del catálogo
    photo_path: str | None = Field(default=None, exclude=True)

    @computed_field
    @property
    def has_photo(self) -> bool:
        return bool(self.photo_path)


class HaircutRequestIn(BaseModel):
    history_id: str | None = None  # None = quitar la petición


class WaitingOut(BaseModel):
    id: str
    status: str
    created_at: str
    client: ClientOut
    is_new: bool  # primera visita (sin datos del peluquero todavía)
    questionnaire_done: bool
    has_hair_texture: bool
    # Corte del historial que el cliente ha pedido repetir hoy.
    requested: HaircutOut | None = None


class WaitingStatusIn(BaseModel):
    status: str  # waiting | in_service | done
