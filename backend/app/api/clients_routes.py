"""
Endpoints de perfil de cliente e historial de visitas.

Guardar cualquier dato a través de este router implica tratar datos
biométricos de una persona identificable (tipo de pelo, forma de cara,
remolinos) de forma persistente — algo que el resto del pipeline evita a
propósito por defecto (ver sección RGPD de CLAUDE.md). No activar este
flujo con clientes reales sin haber resuelto antes cómo se pide y registra
el consentimiento en el mostrador (de momento aquí solo se exige el flag
`consent_history`, pero cómo se recoge ese consentimiento en la práctica
—verbal, checkbox en tablet, papel firmado— es una decisión de producto
pendiente, no solo técnica).
"""

from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    ClientCreateIn,
    ClientOut,
    ClientWithHistoryOut,
    CustomGrowthMapIn,
    FaceShapeOverrideIn,
    HairTypeOverrideIn,
    RecommendationsOut,
    StyleOut,
    StyleRecommendationOut,
    VisagismoAIReportOut,
    VisagismoProfileIn,
    VisitOut,
)
from app.config import ANTHROPIC_MODEL
from app.db import repository
from app.pipeline import visagismo_ai_advisor
from app.pipeline.recommender import recommend_styles

router = APIRouter()


@router.post("/clients", response_model=ClientOut)
def create_client(payload: ClientCreateIn):
    if not payload.consent_history:
        raise HTTPException(
            status_code=422,
            detail=(
                "No se puede crear un perfil sin consent_history=true "
                "(consentimiento explícito del cliente para guardar su historial)."
            ),
        )
    client = repository.create_client(
        display_name=payload.display_name,
        consent_history=payload.consent_history,
        consent_model_improvement=payload.consent_model_improvement,
        consent_save_photo=payload.consent_save_photo,
        consent_ai_analysis=payload.consent_ai_analysis,
        notes=payload.notes,
    )
    return ClientOut(**client.__dict__)


@router.get("/clients", response_model=list[ClientOut])
def list_clients():
    return [ClientOut(**client.__dict__) for client in repository.list_clients()]


@router.get("/clients/{client_id}", response_model=ClientWithHistoryOut)
def get_client(client_id: str):
    client = repository.get_client(client_id)
    if client is None:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    visits = repository.list_visits(client_id)
    return ClientWithHistoryOut(
        **client.__dict__,
        visits=[VisitOut(**visit.__dict__) for visit in visits],
    )


@router.patch("/clients/{client_id}/hair-type", response_model=ClientOut)
def override_hair_type(client_id: str, payload: HairTypeOverrideIn):
    client = repository.update_hair_type_override(client_id, payload.texture)
    if client is None:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    return ClientOut(**client.__dict__)


@router.patch("/clients/{client_id}/face-shape", response_model=ClientOut)
def override_face_shape(client_id: str, payload: FaceShapeOverrideIn):
    client = repository.update_face_shape_override(client_id, payload.face_shape)
    if client is None:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    return ClientOut(**client.__dict__)


@router.patch("/clients/{client_id}/growth-map", response_model=ClientOut)
def override_growth_map(client_id: str, payload: CustomGrowthMapIn):
    """Guarda el mapa de crecimiento dibujado a mano en
    `frontend/growth-map.html`. Sustituye por completo lo que hubiera
    guardado antes para este cliente (ver `repository.update_custom_growth_map`)."""
    client = repository.update_custom_growth_map(
        client_id,
        strokes=[s.model_dump() for s in payload.strokes],
        whorls=[w.model_dump() for w in payload.whorls],
    )
    if client is None:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    return ClientOut(**client.__dict__)


@router.patch("/clients/{client_id}/visagismo-profile", response_model=ClientOut)
def override_visagismo_profile(client_id: str, payload: VisagismoProfileIn):
    """Guarda el perfil extendido de visagismo (morfología craneal/facial,
    métricas físicas del pelo, estilo de vida) que usa `recommend_styles`
    a través de `app/pipeline/visagismo_rules.py`. Igual que `.../growth-map`,
    sustituye por completo lo que hubiera guardado antes -- el barbero puede
    ir rellenando la ficha poco a poco a lo largo de varias visitas, cada
    `PATCH` manda el estado completo del formulario en ese momento, no solo
    los campos que cambiaron."""
    client = repository.update_visagismo_profile(client_id, payload.model_dump())
    if client is None:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    return ClientOut(**client.__dict__)


@router.post("/clients/{client_id}/visagismo-ai-report", response_model=VisagismoAIReportOut)
def generate_visagismo_ai_report(client_id: str):
    """Genera un informe de visagismo con IA (API de Claude) a partir del
    perfil de visagismo ya guardado del cliente -- ver
    `app/pipeline/visagismo_ai_advisor.py` para qué se envía exactamente
    (nunca la foto ni datos identificables) y por qué es una finalidad de
    tratamiento distinta al resto del perfil.

    Requiere `consent_ai_analysis=true` en el perfil del cliente (422 si
    no), igual que `create_client` exige `consent_history=true` -- pero
    aquí el motivo es más fuerte todavía: esto envía datos a un tercero
    (Anthropic), no solo los guarda en el propio servidor.

    Cada llamada tiene coste real de tokens y NO se persiste (no se guarda
    historial de informes) -- ver nota "pendiente" en CLAUDE.md."""
    client = repository.get_client(client_id)
    if client is None:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    if not client.consent_ai_analysis:
        raise HTTPException(
            status_code=422,
            detail=(
                "Este cliente no tiene consent_ai_analysis=true: no se puede "
                "enviar su perfil de visagismo a un servicio externo (API de "
                "Claude) sin ese consentimiento explícito y separado "
                "(ver sección RGPD de CLAUDE.md)."
            ),
        )
    try:
        report = visagismo_ai_advisor.generate_ai_report(client)
    except visagismo_ai_advisor.AIAdvisorNotConfigured as exc:
        # 503: el servicio no está disponible EN ESTE DESPLIEGUE (falta la
        # API key), no un error del cliente ni del propio informe.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except visagismo_ai_advisor.AIAdvisorError as exc:
        # 502: la llamada al servicio externo falló (red, cuota, respuesta
        # inválida...) -- el problema está "río arriba", no en esta API.
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return VisagismoAIReportOut(client_id=client_id, model=ANTHROPIC_MODEL, report=report)


@router.get("/clients/{client_id}/recommendations", response_model=RecommendationsOut)
def get_recommendations(client_id: str):
    """Cortes recomendados para este cliente según su tipo de cabello
    (`hair_texture_override`) y su mapa de crecimiento/remolinos
    (`custom_growth_map`). Requiere que el barbero ya haya rellenado el
    tipo de cabello — si no, devuelve 422 en vez de adivinar."""
    client = repository.get_client(client_id)
    if client is None:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    if not client.hair_texture_override:
        raise HTTPException(
            status_code=422,
            detail=(
                "Este cliente todavía no tiene un tipo de cabello guardado. "
                "Rellena la encuesta de tipo de cabello antes de ver recomendaciones."
            ),
        )
    whorls = (client.custom_growth_map or {}).get("whorls", [])
    # face_shape_override es una corrección a mano del barbero, igual que
    # hair_texture_override — no se usa aquí la "forma de cara detectada"
    # de una simulación individual (SimulationResponse.detected_face_shape)
    # porque esa detección automática no se confirma ni se guarda en el
    # perfil por defecto (ver la Etapa 9 de routes.simulate): sin que el
    # barbero la confirme, no es lo bastante fiable como para condicionar
    # recomendaciones.
    recs = recommend_styles(
        client.hair_texture_override,
        whorls=whorls,
        face_shape=client.face_shape_override,
        visagismo_profile=client.visagismo_profile,
    )
    return RecommendationsOut(
        client_id=client_id,
        hair_texture=client.hair_texture_override,
        whorl_count=len(whorls),
        face_shape=client.face_shape_override,
        recommendations=[
            StyleRecommendationOut(style=StyleOut(**r.style.__dict__), note=r.note)
            for r in recs
        ],
    )
