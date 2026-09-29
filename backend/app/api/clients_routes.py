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

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.api.schemas import (
    AppearanceIn,
    Avatar3DOut,
    BarberSheetOut,
    ClientCreateIn,
    ClientOut,
    ClientWithHistoryOut,
    CustomGrowthMapIn,
    FaceShapeOverrideIn,
    GrowthSummaryOut,
    HairTypeOverrideIn,
    ReasonOut,
    RecommendationsOut,
    SimulationResponse,
    StyleOut,
    StyleRecommendationOut,
    VisagismoAIReportOut,
    VisagismoAutoAnalysisOut,
    VisagismoProfileIn,
    VisitOut,
)
from app import config
from app.config import ANTHROPIC_MODEL
from app.db import repository
from app.api import client_service
from app.pipeline import (
    avatar,
    avatar3d,
    facial_traits_analysis,
    growth_analysis,
    mesh_metrics,
    visagismo_ai_advisor,
    visagismo_vision_analysis,
)
from app.pipeline.recommender import recommend_styles

router = APIRouter()


def _client_or_404(client_id: str):
    client = repository.get_client(client_id)
    if client is None:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    return client


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


@router.get("/clients/{client_id}/avatar")
def client_avatar(client_id: str):
    """Cómo tiene que verse su maniquí: pelo (tipo, largo de hoy, color,
    densidad, línea del pelo) y rasgos de la cara (ver pipeline/avatar.py)."""
    client = repository.get_client(client_id)
    if client is None:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    return client_service.avatar_for(client)


@router.patch("/clients/{client_id}/appearance")
def set_client_appearance(client_id: str, payload: AppearanceIn):
    if payload.hair_color is not None and payload.hair_color not in avatar.HAIR_COLORS:
        raise HTTPException(status_code=422, detail="Color de pelo no válido")
    client = repository.set_appearance(client_id, payload.hair_color,
                                       payload.current_length.model_dump() if payload.current_length else None)
    if client is None:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    return client_service.avatar_for(client)


@router.post("/growth-map/summary", response_model=GrowthSummaryOut)
def growth_map_summary(payload: CustomGrowthMapIn):
    """Interpretación del mapa que se está dibujando (zonas, direcciones,
    raya natural), para enseñarla en growth-map.html mientras se dibuja.
    No guarda nada."""
    s = growth_analysis.summarize(payload.model_dump())
    return GrowthSummaryOut(lines=s.lines(), zone_direction=s.zone_direction,
                            whorl_zones=s.whorl_zones, natural_part=s.natural_part)


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


def _read_upload_as_bgr(upload: UploadFile, label: str) -> np.ndarray:
    contents = upload.file.read()
    image_array = np.frombuffer(contents, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail=f"No se pudo leer la foto de {label}")
    return image_bgr


@router.post("/clients/{client_id}/visagismo-auto-analysis", response_model=VisagismoAutoAnalysisOut)
def override_visagismo_auto_analysis(
    client_id: str,
    photo_frontal: UploadFile = File(...),
    photo_perfil_izquierdo: UploadFile = File(...),
    photo_perfil_derecho: UploadFile = File(...),
):
    """Recibe 3 fotos guiadas del cliente (frontal, perfil izquierdo y
    perfil derecho) y rellena parte de `facial_features_profile` dentro de
    `visagismo_profile`.

    La foto frontal siempre se analiza localmente (simetría ocular,
    separación de ojos, gafas -- ver `facial_traits_analysis.py`): eso es
    gratis, 100% local, y no depende de ningún consentimiento adicional.

    Perfil, cejas, orejas, mentón, mandíbula, cuello, frente, tamaño de
    nariz, grosor de labios y papada NO se pueden medir con geometría
    clásica de forma fiable (probado con fotos reales, ver
    `facial_traits_analysis.py` y CLAUDE.md). Si el cliente tiene
    `consent_ai_analysis=true` y hay `ANTHROPIC_API_KEY` configurada, las 3
    fotos se envían también a la API de Claude para que las juzgue como lo
    haría un peluquero (ver `visagismo_vision_analysis.py`) -- ESTO SÍ
    implica enviar las fotos a un tercero, a diferencia del resto de este
    endpoint. Sin ese consentimiento, o sin esa clave configurada, esos 10
    campos se quedan sin rellenar (con un aviso explicando por qué) y el
    resto del análisis (la parte local) sigue funcionando igual: no es un
    422 que bloquee todo el endpoint, porque hay trabajo útil que hacer
    tanto si se puede como si no se puede analizar por IA.

    RGPD: las 3 fotos se procesan en memoria y NUNCA se guardan en disco
    ni en la base de datos -- solo se guarda el resultado ya resumido en
    categorías (igual que `POST /api/simulate`). Como el dato final que se
    persiste es el mismo `visagismo_profile` que ya cubre `consent_history`
    (misma finalidad de tratamiento que rellenarlo a mano vía `PATCH
    .../visagismo-profile`), la parte local no exige ningún consentimiento
    adicional -- a diferencia de `consent_save_photo` (que sería para
    guardar la foto en sí, cosa que este endpoint nunca hace) o
    `consent_ai_analysis` (que si está dado, hace que las fotos de perfil
    SÍ se decodifiquen y se envíen a la API de Claude, ver arriba).

    Fusiona el resultado con lo que ya hubiera guardado el barbero (no lo
    sustituye por completo, a diferencia de `PATCH .../visagismo-profile`):
    un campo detectado automáticamente (local o por IA) solo se escribe si
    el barbero no lo había rellenado ya a mano, para no pisar una
    corrección manual previa con una detección automática peor."""
    client = repository.get_client(client_id)
    if client is None:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")

    frontal_bgr = _read_upload_as_bgr(photo_frontal, "frontal")
    result = facial_traits_analysis.analyze_facial_traits(frontal_bgr)

    existing_profile = dict(client.visagismo_profile or {})
    existing_anatomical = dict(existing_profile.get("anatomical_metrics") or {})
    existing_features = dict(existing_anatomical.get("facial_features_profile") or {})
    existing_zones = dict(existing_anatomical.get("facial_horizontal_zones_ratio") or {})

    # Solo se auto-rellenan los campos que el barbero no hubiera rellenado
    # ya a mano (ver docstring de arriba y `merge_detected_features`).
    merged_features = facial_traits_analysis.merge_detected_features(
        existing_features, result.facial_features_profile
    )
    merged_zones = dict(existing_zones)

    vision_warnings: list[str] = []
    if client.consent_ai_analysis:
        # Solo se decodifican las fotos de perfil si hace falta enviarlas
        # -- no se procesa ningún dato biométrico que no se vaya a usar
        # (mismo criterio que antes, cuando estas fotos no se analizaban
        # nunca y por eso ni siquiera se decodificaban).
        left_bgr = _read_upload_as_bgr(photo_perfil_izquierdo, "perfil izquierdo")
        right_bgr = _read_upload_as_bgr(photo_perfil_derecho, "perfil derecho")
        try:
            vision_result = visagismo_vision_analysis.analyze_facial_traits_with_vision(
                frontal_bgr, left_bgr, right_bgr
            )
            merged_features = facial_traits_analysis.merge_detected_features(
                merged_features, vision_result.facial_features_profile
            )
            merged_zones = facial_traits_analysis.merge_detected_features(
                merged_zones, vision_result.facial_horizontal_zones_ratio
            )
            vision_warnings = vision_result.warnings
        except visagismo_vision_analysis.VisionAnalysisNotConfigured as exc:
            vision_warnings = [str(exc)]
        except visagismo_vision_analysis.VisionAnalysisError as exc:
            vision_warnings = [str(exc)]
    else:
        vision_warnings = [
            "Perfil, cejas, orejas, mentón, mandíbula, cuello, frente, "
            "nariz, labios y papada no se han analizado por IA: este "
            "cliente no tiene el permiso de IA marcado en su ficha."
        ]

    existing_anatomical["facial_features_profile"] = merged_features
    existing_anatomical["facial_horizontal_zones_ratio"] = merged_zones
    existing_profile["anatomical_metrics"] = existing_anatomical

    updated_client = repository.update_visagismo_profile(client_id, existing_profile)

    return VisagismoAutoAnalysisOut(
        client=ClientOut(**updated_client.__dict__),
        warnings=result.warnings + vision_warnings,
        detected_anomalies_notes=result.detected_anomalies_notes,
    )


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
    """Cortes recomendados para este cliente (ver `client_service`). Si nadie
    ha dicho aún su tipo de pelo (ni el peluquero ni el cliente en su
    cuestionario), se ordena el catálogo entero en vez de dar error: el
    cliente nuevo que espera en el sillón ya puede ver algo útil."""
    client = repository.get_client(client_id)
    if client is None:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    return client_service.build_recommendations(client)


@router.get("/clients/{client_id}/barber-sheet", response_model=BarberSheetOut)
def barber_sheet(client_id: str, style_id: str | None = None):
    """Ficha "cómo pedirlo" del corte recomendado (o del que se indique)."""
    return client_service.barber_sheet_for(_client_or_404(client_id), style_id)


# ---------- Gemelo digital 3D (app/pipeline/avatar3d.py) ----------
def _avatar3d_file(client_id: str):
    return config.CLIENT_PHOTOS_DIR / client_id / "avatar3d" / "model.glb"


def _avatar3d_out(client) -> Avatar3DOut:
    data = client.avatar3d_metrics or {}
    return Avatar3DOut(
        client_id=client.id,
        disponible=avatar3d.available(),
        tiene_modelo=bool(client.avatar3d_path),
        creado=client.avatar3d_at,
        consentimiento=client.consent_3d_scan,
        medidas=data.get("medidas"),
        rasgos=data.get("rasgos"),
        avisos=data.get("avisos") or [],
    )


@router.get("/clients/{client_id}/avatar3d", response_model=Avatar3DOut)
def get_avatar3d(client_id: str):
    """Estado del gemelo 3D del cliente (si hay modelo, cuándo se hizo y
    qué se midió). No genera nada."""
    return _avatar3d_out(_client_or_404(client_id))


@router.get("/clients/{client_id}/avatar3d/model.glb")
def get_avatar3d_model(client_id: str):
    client = _client_or_404(client_id)
    path = _avatar3d_file(client_id)
    if not client.avatar3d_path or not path.exists():
        raise HTTPException(status_code=404, detail="Este cliente no tiene gemelo 3D")
    return FileResponse(path, media_type="model/gltf-binary")


@router.delete("/clients/{client_id}/avatar3d", response_model=Avatar3DOut)
def delete_avatar3d(client_id: str):
    """Borra el modelo 3D del cliente (y sus medidas). Los rasgos que ya
    se hubieran copiado a la ficha se quedan: son parte del perfil, y el
    peluquero puede corregirlos a mano."""
    _client_or_404(client_id)
    path = _avatar3d_file(client_id)
    path.unlink(missing_ok=True)
    return _avatar3d_out(repository.set_avatar3d(client_id, None, None))


@router.post("/clients/{client_id}/avatar3d", response_model=Avatar3DOut)
async def create_avatar3d(
    client_id: str,
    photo_frontal: UploadFile = File(...),
    photo_perfil_izquierdo: UploadFile | None = File(None),
    photo_perfil_derecho: UploadFile | None = File(None),
):
    """Crea el gemelo digital 3D del cliente con las fotos guiadas, mide
    sus rasgos sobre la malla Y analiza la foto frontal en 2D (asimetría/
    separación de ojos, gafas) -- todo en la misma llamada, con las mismas
    fotos, para que este sea el ÚNICO paso de análisis (sept 2026, petición
    de Pedro: "que el peluquero no pierda tanto tiempo" en pasos manuales
    separados). Antes había que pasar además por `visagismo.html`
    (`POST .../visagismo-auto-analysis`, que sigue existiendo por API por
    si hace falta repetir solo esa parte, pero ya no es un paso obligatorio
    del alta -- ver CLAUDE.md).

    RGPD: las fotos se mandan a Tripo (tercero) y el modelo 3D de su cara
    SE GUARDA, así que hace falta `consent_3d_scan` (aparte del resto). Las
    fotos en sí no se guardan en ningún momento (ni para Tripo ni para el
    análisis 2D, que las descarta igual que hacía `visagismo-auto-analysis`).
    Tarda entre medio minuto y unos minutos; cada modelo cuesta ~0,25 $."""
    client = _client_or_404(client_id)
    if not avatar3d.available():
        raise HTTPException(status_code=503, detail="Falta configurar TRIPO_API_KEY para crear el gemelo 3D.")
    if not client.consent_3d_scan:
        raise HTTPException(status_code=422,
                            detail="El cliente no ha dado permiso para crear su modelo 3D.")

    photos = {"frontal": await photo_frontal.read()}
    if photo_perfil_izquierdo is not None:
        photos["perfil_izquierdo"] = await photo_perfil_izquierdo.read()
    if photo_perfil_derecho is not None:
        photos["perfil_derecho"] = await photo_perfil_derecho.read()

    try:
        glb = avatar3d.generate_twin(photos)
    except avatar3d.Avatar3DNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except avatar3d.Avatar3DError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    path = _avatar3d_file(client_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(glb)

    avisos = []
    try:
        analysis = mesh_metrics.analyze(glb)
    except Exception as exc:   # la malla puede venir con una forma inesperada
        analysis, avisos = {"medidas": None, "rasgos": {}}, [f"No se pudieron medir los rasgos: {exc}"]
    if len(photos) < 3:
        avisos.append("Sin las dos fotos de perfil, la nuca y los laterales del modelo son aproximados.")

    profile = dict(client.visagismo_profile or {})
    anat = dict(profile.get("anatomical_metrics") or {})
    feats = dict(anat.get("facial_features_profile") or {})

    # 2D sobre la misma foto frontal (antes exigía el paso aparte de
    # visagismo.html): simetría/separación de ojos, gafas.
    frontal_bgr = cv2.imdecode(np.frombuffer(photos["frontal"], dtype=np.uint8), cv2.IMREAD_COLOR)
    if frontal_bgr is not None:
        try:
            traits2d = facial_traits_analysis.analyze_facial_traits(frontal_bgr)
            feats = facial_traits_analysis.merge_detected_features(feats, traits2d.facial_features_profile)
            avisos.extend(traits2d.warnings)
            if traits2d.detected_anomalies_notes:
                avisos.append(traits2d.detected_anomalies_notes)
        except Exception as exc:
            avisos.append(f"No se pudieron analizar los rasgos de la foto frontal: {exc}")
    else:
        avisos.append("No se pudo leer la foto frontal para el análisis 2D (sí se usó para el gemelo).")

    # Los rasgos de la malla 3D se copian ENCIMA (sin pisar lo que el
    # peluquero ya hubiera puesto a mano en ningún caso, ver
    # `merge_detected_features`).
    rasgos = dict(analysis.get("rasgos") or {})
    if rasgos.get("facial_geometry") and not anat.get("facial_geometry"):
        anat["facial_geometry"] = rasgos["facial_geometry"]
    anat["facial_features_profile"] = facial_traits_analysis.merge_detected_features(
        feats, {k: v for k, v in rasgos.items() if k in ("profile_type", "jawline_definition", "neck_proportions")})
    profile["anatomical_metrics"] = anat
    repository.update_visagismo_profile(client_id, profile)

    metrics = {"medidas": analysis.get("medidas"), "rasgos": rasgos, "avisos": avisos}
    return _avatar3d_out(repository.set_avatar3d(client_id, str(path), metrics))


@router.post("/clients/{client_id}/avatar3d/simulate", response_model=SimulationResponse)
async def simulate_on_avatar3d(client_id: str, style_id: str = Form(...), provider: str | None = Form(None),
                               render: UploadFile = File(...)):
    """Simula un corte del catálogo sobre una captura frontal del gemelo
    3D (`render`, una foto de lo que se ve en el visor, no la malla) en
    vez de sobre la foto guardada -- ver `client_service.simulate_on_avatar3d`
    para por qué reutiliza el mismo `consent_simulation` en vez de uno
    nuevo. El visor (`gemelo.html`) la genera con `canvas.toBlob()` justo
    antes de mandarla."""
    client = _client_or_404(client_id)
    render_bgr = _read_upload_as_bgr(render, "captura del gemelo 3D")
    return client_service.simulate_on_avatar3d(client, style_id, provider, render_bgr, requested_by="peluquero")
