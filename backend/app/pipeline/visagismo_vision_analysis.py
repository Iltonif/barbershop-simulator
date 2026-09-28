"""
Análisis de rasgos faciales de perfil (perfil de nariz, cejas, orejas,
mentón, mandíbula y cuello) usando visión de un modelo de IA, en vez de
geometría clásica (landmarks/contorno).

Por qué existe este módulo -- contexto completo en CLAUDE.md, sección
"Análisis automático de rasgos faciales" y su subsección "Calibración de
umbrales": la geometría clásica (68 landmarks dlib/LBF + segmentación
BiSeNet, ver `facial_traits_analysis.py`) se probó a fondo con 61 fotos
reales de calibración y con las 3 fotos guiadas de Pedro, y NINGUNO de
estos 6 rasgos se pudo medir de forma fiable:
- Cejas: la métrica de curvatura salía AL REVÉS de lo que ve una persona
  (cejas depiladas y arqueadas salían "rectas").
- Orejas: BiSeNet (entrenado con caras de frente) confunde barbilla y boca
  con oreja en una foto de perfil; desde la foto frontal sí segmenta bien
  la oreja, pero ninguna medida de "cuánto sobresale" resultó fiable
  (correlación izquierda-derecha de -0,05 en la misma cara).
- Perfil de nariz: el detector de caras (Haar + landmarks LBF) ni siquiera
  encuentra una cara en una foto de perfil.
- Mandíbula: algo de señal (AUC 0,66 sobre 1,0), pero muy lejos de lo que
  hace falta para clasificar a un cliente real sin que el barbero lo
  revise.
- Mentón: necesita información de profundidad (sagital) que una foto 2D
  no contiene.
- Cuello: la medida depende del encuadre de la foto (cuánto cuello entra
  en el plano), no de la persona -- AUC ~0,5, ruido puro.
- Una rama aparte (`wip-perfil-automatico`, sin fusionar) probó medir
  directamente sobre el CONTORNO de la silueta (convexidad de Legan,
  ángulo mentón-cuello, ángulo cervicomental, con normas publicadas) y
  tampoco funcionó con fotos reales: el contorno es frágil ante el fondo,
  la resolución y el ángulo de la cámara (la misma foto de Pedro dio
  convexidades de 15° a 33° según esos detalles).

En vez de seguir midiendo ángulos con más precisión, este módulo le pasa
las 3 fotos directamente a un modelo de IA con visión (API de Claude, ya
integrada en `visagismo_ai_advisor.py` para el informe de texto) y le pide
que JUZGUE estos 6 rasgos como lo haría un peluquero mirando las fotos --
sin intentar extraer coordenadas ni ángulos exactos. Decisión de Pedro
(sept 2026): usar la API de Claude, no Gemini (que también está integrada
en `haircut_editor.py` para la simulación de imagen, pero con otro fin).

Diferencia importante con el resto de `facial_traits_analysis.py`: aquello
es 100% local (nunca sale del servidor). Esto es una llamada a un servicio
EXTERNO de pago, y además, a diferencia de `visagismo_ai_advisor.py` (que
solo envía datos categóricos ya recogidos, NUNCA una foto), este módulo
envía las 3 FOTOS reales del cliente a la API de Claude. Por eso:
- Requiere `ANTHROPIC_API_KEY` configurada, igual que el informe de texto.
- Requiere `consent_ai_analysis=true` en el perfil del cliente -- se
  reutiliza el mismo consentimiento que ya exigía el informe de texto
  para "enviar datos a un tercero" (ver `clients_routes.py`), porque es
  la misma finalidad de tratamiento (análisis de visajismo por la API de
  Claude), ahora ampliada a incluir también las fotos, no solo el texto
  ya recogido. La comprobación de ese consentimiento vive en
  `clients_routes.override_visagismo_auto_analysis`, no aquí: este
  módulo no conoce el perfil del cliente, solo recibe las fotos.
- Las 3 fotos se procesan en memoria y se descartan justo después de la
  llamada -- no se guardan en disco ni en la base de datos, igual que el
  resto del pipeline. A diferencia de la geometría clásica de
  `facial_traits_analysis.py`, aquí las fotos SÍ viajan fuera del
  servidor (a la API de Claude): es inevitable si se quiere que el
  modelo las "vea", y es precisamente lo que distingue a este
  consentimiento del resto.

Tests: `backend/tests/test_visagismo_vision_analysis.py`, mismo patrón que
`test_visagismo_ai_advisor.py` (mockeando `anthropic.Anthropic`, sin
llamada real ni fotos reales).
"""

import base64
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL

# Mismo límite que `haircut_editor._MAX_SIDE_PX`: resolución de sobra para
# que el modelo distinga estos rasgos, sin pagar tokens de más por una
# foto de móvil sin comprimir.
_MAX_SIDE_PX = 1536

# Mismos valores que `FacialFeaturesProfileIn` (`app/api/schemas.py`) y el
# motor de reglas (`visagismo_rules.py`). Si el modelo devolviera otra
# cosa, se descarta con un aviso en vez de guardar un valor inventado.
_VALID_VALUES: dict[str, set[str]] = {
    "profile_type": {"straight", "convex_prominent_nose", "concave"},
    "ears_projection": {"flat", "prominent_protruding"},
    "neck_proportions": {"short_thick", "long_thin", "proportional"},
    "eyebrow_type": {"straight_low", "arched", "prominent_ridge"},
    "chin_projection": {"retruded", "balanced", "prominent"},
    "jawline_definition": {"defined", "soft"},
}

_FIELD_LABELS = {
    "profile_type": "perfil de nariz",
    "ears_projection": "proyección de orejas",
    "neck_proportions": "proporciones de cuello",
    "eyebrow_type": "forma de cejas",
    "chin_projection": "proyección de mentón",
    "jawline_definition": "definición de mandíbula",
}

SYSTEM_PROMPT = """Eres un peluquero/barbero experto en visajismo masculino. Vas a recibir 3 fotos guiadas de un cliente: frontal, perfil izquierdo y perfil derecho. Tu tarea es juzgar, solo a partir de lo que ves en las fotos (como lo haría un peluquero mirando a un cliente real, no midiendo ángulos con precisión clínica), estos 6 rasgos:

- profile_type: perfil de la nariz visto de lado -- "straight" (recto), "convex_prominent_nose" (convexo/nariz prominente) o "concave" (cóncavo).
- ears_projection: cuánto sobresalen las orejas de la cabeza -- "flat" (pegadas) o "prominent_protruding" (de soplillo/prominentes).
- neck_proportions: proporción del cuello -- "short_thick" (corto y grueso), "long_thin" (largo y fino) o "proportional" (proporcionado).
- eyebrow_type: forma de las cejas -- "straight_low" (recta/baja), "arched" (arqueada) o "prominent_ridge" (arco superciliar marcado).
- chin_projection: proyección del mentón visto de perfil -- "retruded" (retraído), "balanced" (equilibrado) o "prominent" (prominente).
- jawline_definition: definición de la línea mandibular -- "defined" (definida) o "soft" (poco definida).

Usa sobre todo las fotos de perfil para el perfil de nariz, el mentón, la mandíbula y el cuello, y la foto frontal (apoyándote en las de perfil si hace falta) para las cejas y las orejas.

Si alguno de los 6 rasgos no se puede juzgar con una confianza razonable en las fotos recibidas (foto borrosa, mal encuadrada, pelo tapando la zona, ángulo insuficiente, cabeza girada), devuelve null en ese campo en vez de adivinar: es preferible dejar un campo sin rellenar, para que el peluquero lo revise a mano, que rellenarlo mal. Responde siempre llamando a la herramienta `record_facial_traits`, nunca en texto libre."""

_TOOL_SCHEMA = {
    "name": "record_facial_traits",
    "description": (
        "Registra el juicio visual de los 6 rasgos faciales de perfil a "
        "partir de las 3 fotos, o null en el campo que no se pueda juzgar "
        "con confianza razonable."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            **{
                name: {"type": ["string", "null"], "enum": sorted(values) + [None]}
                for name, values in _VALID_VALUES.items()
            },
            "confidence_notes": {
                "type": "string",
                "description": (
                    "Nota breve (1-2 frases, en español) sobre la calidad de "
                    "las fotos o la confianza del juicio -- p.ej. si alguna "
                    "foto de perfil no permitía ver bien un rasgo."
                ),
            },
        },
        "required": [*list(_VALID_VALUES.keys()), "confidence_notes"],
    },
}


class VisionAnalysisNotConfigured(RuntimeError):
    """`ANTHROPIC_API_KEY` no está configurada en este despliegue, o el
    paquete `anthropic` no está instalado."""


class VisionAnalysisError(RuntimeError):
    """La llamada a la API de Claude falló (red, cuota, autenticación...),
    o devolvió algo que no se pudo interpretar. El mensaje ya viene
    pensado para mostrarse al barbero tal cual, como aviso, sin romper el
    resto del análisis (ver `clients_routes.override_visagismo_auto_analysis`,
    que captura esta excepción y sigue con la parte local del análisis)."""


@dataclass
class VisionTraitsResult:
    facial_features_profile: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _to_jpeg(image_bgr: np.ndarray) -> bytes:
    """Mismo criterio que `haircut_editor._to_jpeg`: recorta el lado mayor
    a 1536px y comprime a JPEG calidad 92 antes de enviar, para no pagar
    tokens de más por una foto de móvil sin comprimir ni arriesgarse a los
    límites de tamaño de la API."""
    h, w = image_bgr.shape[:2]
    scale = min(1.0, _MAX_SIDE_PX / max(h, w))
    if scale < 1.0:
        image_bgr = cv2.resize(image_bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise VisionAnalysisError("No se pudo preparar una foto para enviarla al análisis por IA")
    return buf.tobytes()


def _image_block(image_bgr: np.ndarray) -> dict:
    data = base64.standard_b64encode(_to_jpeg(image_bgr)).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


def analyze_facial_traits_with_vision(
    frontal_bgr: np.ndarray,
    left_profile_bgr: np.ndarray,
    right_profile_bgr: np.ndarray,
) -> VisionTraitsResult:
    """Envía las 3 fotos a la API de Claude y devuelve los rasgos que el
    modelo haya podido juzgar con confianza (los demás se quedan fuera de
    `facial_features_profile`, con un aviso en `warnings`).

    Lanza `VisionAnalysisNotConfigured` si no hay `ANTHROPIC_API_KEY` (o
    falta el paquete `anthropic`), y `VisionAnalysisError` si la llamada
    falla o la respuesta no se puede interpretar -- el llamador
    (`clients_routes.py`) decide qué hacer con esos casos; a diferencia de
    `visagismo_ai_advisor.generate_ai_report` (que si falla no da ningún
    informe), aquí lo normal es capturarlas y seguir con el resto del
    análisis local, que no depende de esto."""
    if not ANTHROPIC_API_KEY:
        raise VisionAnalysisNotConfigured(
            "No hay ninguna ANTHROPIC_API_KEY configurada en este despliegue: "
            "perfil, cejas, orejas, mentón, mandíbula y cuello no se pueden "
            "analizar por IA hasta que se configure esa variable de entorno "
            "(ver CLAUDE.md)."
        )
    try:
        import anthropic
    except ImportError as exc:
        raise VisionAnalysisNotConfigured(
            "El paquete 'anthropic' no está instalado en este despliegue."
        ) from exc

    try:
        api_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = api_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=[_TOOL_SCHEMA],
            tool_choice={"type": "tool", "name": "record_facial_traits"},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Foto frontal:"},
                    _image_block(frontal_bgr),
                    {"type": "text", "text": "Foto de perfil izquierdo:"},
                    _image_block(left_profile_bgr),
                    {"type": "text", "text": "Foto de perfil derecho:"},
                    _image_block(right_profile_bgr),
                ],
            }],
        )
    except anthropic.AnthropicError as exc:
        raise VisionAnalysisError(f"La llamada a la API de Claude falló: {exc}") from exc

    tool_use = next((block for block in response.content if getattr(block, "type", None) == "tool_use"), None)
    if tool_use is None:
        raise VisionAnalysisError(
            "La API de Claude no devolvió un resultado interpretable para estos rasgos."
        )

    raw = tool_use.input or {}
    result = VisionTraitsResult()
    for field_name, valid_values in _VALID_VALUES.items():
        value = raw.get(field_name)
        if value in valid_values:
            result.facial_features_profile[field_name] = value
        elif value is not None:
            result.warnings.append(
                f"La IA devolvió un valor no reconocido para {_FIELD_LABELS[field_name]} "
                f"({value!r}): se descarta, revísalo a mano."
            )
    note = (raw.get("confidence_notes") or "").strip()
    if note:
        result.warnings.append(f"Nota de la IA sobre estas fotos: {note}")
    missing = [_FIELD_LABELS[f] for f in _VALID_VALUES if f not in result.facial_features_profile]
    if missing:
        result.warnings.append(
            "La IA no pudo juzgar con confianza: " + ", ".join(missing) + " -- revísalo a mano."
        )
    return result
