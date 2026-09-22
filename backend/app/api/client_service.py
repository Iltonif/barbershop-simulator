"""
Lógica común a las rutas del peluquero (`clients_routes.py`) y del propio
cliente (`session_routes.py`, `/api/me/*`): recomendaciones y simulación
con la foto guardada del cliente.
"""

from __future__ import annotations

import base64

import cv2
from fastapi import HTTPException

from app import config
from app.api.schemas import ReasonOut, RecommendationsOut, SimulationResponse, StyleOut, StyleRecommendationOut
from app.db import repository
from app.db.models import ClientProfile
from app.pipeline import avatar, growth_analysis, haircut_editor, maintenance, trait_rules
from app.pipeline.recommender import recommend_styles
from app.pipeline.style_catalog import get_style_by_id

# Lo que responde el cliente en "Mi perfil" (hair_pattern_shape) -> tipo de
# pelo del catálogo. La corrección del peluquero (hair_texture_override)
# siempre tiene prioridad.
_PATTERN_TO_TEXTURE = {"straight": "liso", "wavy": "ondulado", "curly": "rizado", "coily": "afro"}


def effective_hair_texture(client: ClientProfile) -> tuple[str | None, str | None]:
    """(tipo de pelo, quién lo dijo: "peluquero" | "cliente" | None)."""
    if client.hair_texture_override:
        return client.hair_texture_override, "peluquero"
    pattern = ((client.visagismo_profile or {}).get("hair_physical_metrics") or {}).get("hair_pattern_shape")
    if pattern in _PATTERN_TO_TEXTURE:
        return _PATTERN_TO_TEXTURE[pattern], "cliente"
    return None, None


def build_recommendations(client: ClientProfile) -> RecommendationsOut:
    texture, source = effective_hair_texture(client)
    whorls = (client.custom_growth_map or {}).get("whorls", [])
    # face_shape_override: la marca el peluquero o el propio cliente en su
    # cuestionario; nunca la detección automática sin confirmar.
    recs = recommend_styles(texture, face_shape=client.face_shape_override,
                            visagismo_profile=client.visagismo_profile, growth_map=client.custom_growth_map)
    growth = growth_analysis.summarize(client.custom_growth_map)
    return RecommendationsOut(
        client_id=client.id,
        hair_texture=texture,
        hair_texture_source=source,
        whorl_count=len(whorls),
        face_shape=client.face_shape_override,
        recommendations=[
            StyleRecommendationOut(
                style=StyleOut(**r.style.__dict__),
                note=r.note,
                reasons=[ReasonOut(label=e.label, detail=e.detail) for e in r.reasons],
                warnings=[ReasonOut(label=e.label, detail=e.detail) for e in r.warnings],
            )
            for r in recs
        ],
        beard_advice=[ReasonOut(**b) for b in trait_rules.beard_advice(client.visagismo_profile)],
        liked_styles=client.liked_styles,
        growth_summary=growth.lines(),
        natural_part=growth.natural_part,
    )


def simulate_with_stored_photo(client: ClientProfile, style_id: str, provider: str | None,
                               requested_by: str) -> SimulationResponse:
    """Simula un corte sobre la foto que el peluquero guardó en la primera
    visita. Exige el consentimiento de simulación guardado en la ficha
    (`consent_simulation`) y, si lo pide el propio cliente, respeta el tope
    diario `MAX_CLIENT_SIMULATIONS_PER_DAY`."""
    style = get_style_by_id(style_id)
    if style is None:
        raise HTTPException(status_code=404, detail="Corte no encontrado")
    if not client.simulation_photo_path:
        raise HTTPException(status_code=409, detail="Todavía no hay foto para simular: el peluquero la hace en la primera visita.")
    if not client.consent_simulation:
        raise HTTPException(status_code=422, detail="Falta el consentimiento para enviar la foto y simular cortes.")
    configured = [p.id for p in haircut_editor.available_providers()]
    if not configured:
        raise HTTPException(status_code=503, detail="La simulación todavía no está activada en la barbería.")
    if provider and provider not in configured:
        raise HTTPException(status_code=422, detail=f"El proveedor '{provider}' no está configurado")
    if requested_by == "cliente" and repository.count_simulations_today(client.id, "cliente") >= config.MAX_CLIENT_SIMULATIONS_PER_DAY:
        raise HTTPException(status_code=429, detail="Has llegado al máximo de simulaciones de hoy. Pídele al peluquero que te enseñe más.")

    image = cv2.imread(client.simulation_photo_path, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=409, detail="No se encuentra la foto guardada; el peluquero puede hacer otra.")
    texture = client.hair_texture_override  # solo si lo ha marcado una persona experta (ver routes.simulate)
    use = provider or configured[0]
    try:
        result = haircut_editor.edit_haircut(image, style, texture, use,
                                             reference=haircut_editor.load_reference_photo(style))
    except haircut_editor.HaircutEditorNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except haircut_editor.HaircutEditorError as exc:
        raise HTTPException(status_code=502, detail=f"No se pudo generar la simulación. {exc}")
    repository.log_simulation(client.id, requested_by)
    _, buf = cv2.imencode(".jpg", result)
    return SimulationResponse(
        style_id=style.id,
        detected_hair_texture=texture or "",
        detected_face_shape=client.face_shape_override or "",
        warnings=[],
        image_base64=base64.b64encode(buf).decode(),
        provider=use,
    )


def visit_frequency_days(client: ClientProfile) -> int | None:
    return ((client.visagismo_profile or {}).get("lifestyle_and_preferences") or {}).get("barbershop_visit_frequency_days")


def next_visit_for(client: ClientProfile, last: dict | None) -> dict | None:
    """Cuándo le toca volver (ver maintenance.next_visit), con el nombre del
    corte que lleva si quedó registrado."""
    if not last:
        return None
    style = get_style_by_id(last["style_id"]) if last.get("style_id") else None
    weeks = maintenance.weeks_for_style(style) if style else None
    nv = maintenance.next_visit(last["at"], weeks, visit_frequency_days(client))
    if nv:
        nv["last_visit"] = last["at"]
        nv["style_name"] = style.name if style else None
    return nv


def last_cut_lengths(client: ClientProfile) -> dict | None:
    """Largo con el que salió del último corte registrado (si era uno del
    catálogo), para el maniquí."""
    for record in repository.list_haircuts(client.id):
        style = get_style_by_id(record.style_id) if record.style_id else None
        if style:
            return {"at": record.created_at, "top": style.length_top_mm, "sides": style.length_sides_mm,
                    "back": style.length_back_mm, "fade": style.fade_type, "style_name": style.name}
    return None


def avatar_for(client: ClientProfile) -> dict:
    texture, _ = effective_hair_texture(client)
    cut = last_cut_lengths(client)
    params = avatar.avatar_params(client, cut, texture)
    params["length"]["style_name"] = cut["style_name"] if cut and params["length"]["source"] == "corte" else None
    return params
