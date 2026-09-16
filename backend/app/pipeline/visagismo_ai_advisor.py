"""
Informe de visagismo generado por IA (complementa, no sustituye, al motor
de reglas determinista de `visagismo_rules.py`).

Origen: el usuario pegó un system prompt completo ("Motor Experto en
Visajismo Masculino"), con un pipeline de razonamiento por prioridades y
un formato de salida clínico detallado (diagnóstico morfológico,
prescripción de corte, diseño de barba, guía de estilizado y un prompt
para generación de imagen). A diferencia de las 5 reglas de
`visagismo_rules.py` (traducidas a código determinista porque eran
condiciones simples if/then), este prompt razona sobre muchos rasgos a
la vez de forma cualitativa -- encaja mejor con un LLM que con reglas
escritas a mano, así que aquí se usa tal cual como `system` de una
llamada a la API de Claude, sin reescribir su contenido.

Diferencia importante con TODO lo demás en `app/pipeline/`: esto es la
PRIMERA llamada a un servicio externo de pago en el proyecto. Hasta
ahora todo el pipeline (segmentación, landmarks, catálogo, reglas de
recomendación) corre en local, sin salir del servidor ni tener coste
por petición. Esto rompe esa propiedad: cada informe cuesta dinero real
(tokens de la API de Claude) y envía el perfil de visagismo del cliente
a un tercero (Anthropic). Por eso:

- Requiere `ANTHROPIC_API_KEY` configurada (ver `app/config.py`); si no
  está configurada, `generate_ai_report` lanza `AIAdvisorNotConfigured`
  en vez de fallar de forma confusa más abajo o exponer un 500 críptico.
- Requiere un consentimiento SEPARADO del cliente (`consent_ai_analysis`,
  ver `ClientProfile` / `clients_routes.py`) -- es una finalidad de
  tratamiento distinta a guardar el perfil localmente (RGPD: cada
  finalidad, su propio consentimiento), y encima implica transferir
  datos a un proveedor externo, no solo guardarlos en el propio servidor.
- NUNCA se envía la foto del cliente ni ningún dato identificable
  (nombre, notas libres) -- solo los campos categóricos ya recogidos en
  `visagismo_profile` / `hair_texture_override` / `face_shape_override` /
  remolinos, igual que ya hace `visagismo_rules.py` con esos mismos datos.
- El texto que devuelve el modelo es una interpretación cualitativa de
  estética/peluquería, NO un diagnóstico médico real, a pesar del tono
  "clínico" del prompt original (términos como "evaluación de rasgos
  críticos" son terminología de peluquería, no medicina).
- No se persiste en BD (a diferencia del resto del perfil): cada llamada
  genera un informe nuevo bajo demanda, no se guarda historial de
  informes todavía -- ver nota de "pendiente" en CLAUDE.md.
"""

from app.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL
from app.db.models import ClientProfile

# System prompt tal cual lo definió el usuario -- no se ha reescrito su
# contenido (más allá de este comentario indicando su origen), solo se
# usa como el parámetro `system` de la llamada a la API de Claude.
SYSTEM_PROMPT = """[ROLE & CONTEXT]
Eres el Motor Experto en Visajismo Masculino y Simulación Estética más avanzado del mundo. Tu función es operar como un Asesor Clínico de Imagen y Especialista en Morfología Craneofacial. Tu objetivo es procesar métricas exactas del rostro, cráneo y rasgos de un usuario para diagnosticar su estructura y prescribir el corte de cabello, diseño de barba y ajustes volumétricos perfectos.

[PIPELINE LÓGICO DE INFERENCIA DE IA]
Para cada análisis, debes evaluar los datos en el siguiente orden estricto de prioridad:
1. Restricciones Anatómicas Óseas (Cráneo y Mandíbula) -> Determinan la estructura del corte.
2. Micro-Rasgos Faciales (Nariz, Orejas, Ojos, Cejas, Frente) -> Determinan los puntos de balance óptico.
3. Línea Capilar y Textura (Entradas, Densidad, Remolinos) -> Determinan la viabilidad real del cabello.
4. Estilo de vida y Mantenimiento -> Filtran el estilo final viable.

[MANUAL DE PONDERACIÓN DE RASGOS FACIALES AVANZADOS]
Debes cruzar las variables anatómicas con las siguientes reglas de compensación geométrica:

1. LA FRENTE Y LA LÍNEA CAPILAR (Zona Intelectual)
   - Frente Muy Alta / Despejada: Prohibido peinados hacia atrás (Slick Back) sin volumen frontal. Prescribir texturizados hacia adelante (French Crop, Flequillos desordenados) para acortar visualmente el tercio superior.
   - Frente Estrecha: Aumentar volumen superior (Pompadour, Quiff) y despejar el rostro para alargar la zona intelectual.

2. LA NARIZ (Perfilometría Convexo/Cóncavo)
   - Nariz Prominente / Aguileña / Perfil Convexo: Prohibido laterales rapados al cero con la parte superior plana (acentúa el efecto "pájaro"). Prescribir volumen moderado en los laterales (Taper Fades altos o cortes a tijera) y volumen texturizado en la coronilla y flequillo para equilibrar el plano de la nariz.
   - Nariz Pequeña / Chata: Permitir cortes muy limpios, frentes descubiertas y Fades comprimidos para dar protagonismo al centro del rostro.

3. LAS OREJAS (Proyección y Tamaño)
   - Orejas Prominentes / "En Asa" / Despegadas: Prohibido Skin Fades (rapados totales a los lados) que dejen la piel blanca, ya que hacen que las orejas resalten como un punto focal. Prescribir cortes a tijera con densidad (mínimo 2-3 cm de grosor en los laterales) o Low Taper Fades que mantengan oscuridad alrededor de la oreja para camuflar la proyección.

4. LOS OJOS Y LAS CEJAS (Distancia y Ángulo)
   - Ojos Muy Juntos: Evitar flequillos pesados y rectos que encierren la mirada. Recomendar frentes despejadas o rayas a un lado muy limpias.
   - Cejas Muy Rectas y Caídas: Evitar cortes con líneas muy horizontales en el flequillo. Añadir texturizado angular para romper la pesadez de la mirada.
   - Cejas Prominentes / Arco Superciliar Marcado: Evitar tupés extremadamente altos que generen sombras oscuras en los ojos.

5. LA MANDÍBULA Y EL MENTÓN (Zona Sensitiva)
   - Mentón Retraído (Perfil Convexo): Prescribir barba con longitud en la punta (tipo candado largo o barba completa con peso hacia adelante) para proyectar la mandíbula. Evitar cuellos rapados muy altos.
   - Mandíbula Cuadrada / Hipermasculina: Suavizar con contornos de barba ligeramente redondeados si se busca un look formal, o potenciar con líneas ultra-rectas para un look agresivo.

[MATRIZ DE RECREACIÓN Y RECOMENDACIÓN DIGITAL (Para el módulo de renderizado/imagen)]
Cuando emitas tu recomendación para que el sistema genere o recree el cabello (vía IA generativa de imágenes), debes mapear los resultados usando el principio de "Cuadrangulación Visajista": El objetivo final de toda composición capilar es simular visualmente un óvalo o un cuadrado simétrico.

[FORMATO OBLIGATORIO DE RESPUESTA EN OUTPUT]
Para cada cliente analizado, estructura tu respuesta en este formato limpio y altamente escaneable:

### 1. DIAGNÓSTICO MORFOLÓGICO AVANZADO
* **Estructura Craneal:** [Meso/Braqui/Dolicocéfalo] + Impacto en el corte.
* **Geometría Facial:** [Tipo de Rostro] + Análisis de tercios.
* **Evaluación de Rasgos Críticos:** [Análisis de Nariz, Orejas, Frente y Cejas detectadas].

### 2. PRESCRIPCIÓN TÉCNICA DEL CORTE IDEAL
* **Nombre del Estilo:** [Ej: Textured Layered Crop con Low Taper]
* **Configuración de Laterales (Fades/Tijera):** [Instrucción exacta en longitudes/guías].
* **Configuración Superior y Flequillo:** [Cómo cortar el área superior y el flequillo según la frente/nariz].
* **Tratamiento de Remolinos y Dirección:** [Indicación para el estilista].

### 3. DISEÑO DE BARBA Y COMPENSACIÓN FACIAL
* **Estructura Recomendada:** [Ej: Barba corporativa de 3 semanas con peso en mentón].
* **Línea de Mejilla y Cuello:** [Geometría de las líneas para corregir mandíbula/cuello].

### 4. GUÍA DE ESTILIZADO Y PRODUCTO
* **Herramientas:** [Secador, peine de dientes anchos, etc.]
* **Producto de Fijación:** [Pomada mate, cera de arcilla, polvos de textura] + Razón física de la elección.

### 5. PARÁMETROS PARA GENERACIÓN VISUAL (PROMPT HELPER)
* **Prompt de Recreación:** [Genera aquí un prompt optimizado en inglés para Midjourney/Stable Diffusion que muestre el corte ideal exacto aplicado a los rasgos del usuario].

[DATOS DISPONIBLES Y SUS LÍMITES]
Los datos del cliente que recibirás a continuación, en el mensaje del usuario, vienen de un formulario categórico rellenado por el barbero (nunca de una foto o medición real): cada campo puede faltar si todavía no se ha rellenado. Cuando un dato no esté indicado, dilo explícitamente como "no especificado" en el diagnóstico correspondiente y da la recomendación más general y segura para ese caso -- no inventes medidas, proporciones ni rasgos que no se te han dado."""


class AIAdvisorNotConfigured(RuntimeError):
    """`ANTHROPIC_API_KEY` no está configurada en este despliegue, o el
    paquete `anthropic` no está instalado."""


class AIAdvisorError(RuntimeError):
    """La llamada a la API de Claude falló (red, cuota, autenticación,
    respuesta inválida...). El mensaje ya viene pensado para mostrarse
    al barbero tal cual, sin volcar la excepción original de la SDK."""


def _get(d, *path, default=None):
    """Navega un dict anidado (el `visagismo_profile` guardado, cualquier
    nivel puede faltar) sin lanzar KeyError."""
    node = d
    for key in path:
        if not isinstance(node, dict):
            return default
        node = node.get(key)
        if node is None:
            return default
    return node


def _campo(etiqueta: str, valor) -> str:
    return f"- {etiqueta}: {valor if valor not in (None, '') else 'no especificado'}"


def build_user_message(client: ClientProfile) -> str:
    """Serializa los datos disponibles del cliente (perfil de visagismo +
    correcciones ya guardadas del barbero) en un texto legible para el
    modelo. Deliberadamente NO omite los campos vacíos: los marca como
    "no especificado" para que el propio modelo decida qué generalizar en
    vez de que la ausencia del campo se confunda con "no aplica"."""
    p = client.visagismo_profile or {}
    whorls = _get(client.custom_growth_map or {}, "whorls", default=[]) or []

    lineas = [
        "Perfil de visagismo del cliente (formulario categórico del barbero, sin foto ni medición real):",
        "",
        "## Morfología craneal y geometría facial",
        _campo("Estructura craneal", _get(p, "anatomical_metrics", "cranial_morphology")),
        _campo("Geometría/forma facial (detallada)", _get(p, "anatomical_metrics", "facial_geometry")),
        _campo("Forma de cara (confirmada por el barbero, categoría simple)", client.face_shape_override),
        _campo("Proporción zona intelectual (frente)", _get(p, "anatomical_metrics", "facial_horizontal_zones_ratio", "intellectual_zone_forehead")),
        _campo("Proporción zona afectiva (tercio medio)", _get(p, "anatomical_metrics", "facial_horizontal_zones_ratio", "affective_zone_mid_face")),
        _campo("Proporción zona sensitiva (mandíbula/mentón)", _get(p, "anatomical_metrics", "facial_horizontal_zones_ratio", "sensitive_zone_jaw_chin")),
        "",
        "## Rasgos faciales",
        _campo("Perfil (nariz)", _get(p, "anatomical_metrics", "facial_features_profile", "profile_type")),
        _campo("Proyección de orejas", _get(p, "anatomical_metrics", "facial_features_profile", "ears_projection")),
        _campo("Proporciones de cuello", _get(p, "anatomical_metrics", "facial_features_profile", "neck_proportions")),
        _campo("Separación de ojos", _get(p, "anatomical_metrics", "facial_features_profile", "eye_spacing")),
        _campo("Forma de cejas", _get(p, "anatomical_metrics", "facial_features_profile", "eyebrow_type")),
        "",
        "## Pelo",
        _campo("Textura de pelo (confirmada por el barbero)", client.hair_texture_override),
        _campo("Densidad", _get(p, "hair_physical_metrics", "hair_density")),
        _campo("Grosor de la fibra", _get(p, "hair_physical_metrics", "hair_texture_thickness")),
        _campo("Patrón de rizo", _get(p, "hair_physical_metrics", "hair_pattern_shape")),
        _campo("Forma de la línea de nacimiento", _get(p, "hair_physical_metrics", "frontal_hairline_shape")),
        _campo("Remolino de coronilla", _get(p, "hair_physical_metrics", "growth_directions_cowlicks", "crown_cowlick")),
        _campo("Dirección de flequillo", _get(p, "hair_physical_metrics", "growth_directions_cowlicks", "fringe_direction")),
        _campo("Remolinos marcados en el mapa de crecimiento (nº)", len(whorls) if whorls else "0 (ninguno marcado)"),
        "",
        "## Estilo de vida",
        _campo("Mantenimiento diario disponible", _get(p, "lifestyle_and_preferences", "daily_maintenance_commitment")),
        _campo("Usa producto de peinado", _get(p, "lifestyle_and_preferences", "styling_products_usage", "uses_product")),
        _campo("Acabado preferido", _get(p, "lifestyle_and_preferences", "styling_products_usage", "preferred_finish")),
        _campo("Frecuencia de visita a la barbería (días)", _get(p, "lifestyle_and_preferences", "barbershop_visit_frequency_days")),
        _campo("Entorno profesional/social", _get(p, "lifestyle_and_preferences", "professional_social_environment")),
        _campo("Preferencia de barba", _get(p, "lifestyle_and_preferences", "beard_preference")),
    ]
    return "\n".join(lineas)


def generate_ai_report(client: ClientProfile) -> str:
    """Genera el informe de visagismo llamando a la API de Claude con
    `SYSTEM_PROMPT` + los datos del cliente serializados por
    `build_user_message`. Lanza `AIAdvisorNotConfigured` si no hay API key
    (o falta el paquete `anthropic`), y `AIAdvisorError` si la llamada
    falla por cualquier otro motivo -- ambas pensadas para convertirse
    directamente en una respuesta HTTP clara en `clients_routes.py`, sin
    que el barbero vea una traza de Python."""
    if not ANTHROPIC_API_KEY:
        raise AIAdvisorNotConfigured(
            "No hay ninguna ANTHROPIC_API_KEY configurada en este despliegue: "
            "el informe de visagismo por IA no está disponible hasta que se "
            "configure esa variable de entorno (ver sección correspondiente "
            "en CLAUDE.md)."
        )
    try:
        import anthropic
    except ImportError as exc:
        raise AIAdvisorNotConfigured(
            "El paquete 'anthropic' no está instalado en este despliegue."
        ) from exc

    try:
        api_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = api_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_message(client)}],
        )
    except anthropic.AnthropicError as exc:
        raise AIAdvisorError(
            f"La llamada a la API de Claude falló: {exc}"
        ) from exc

    texto = "".join(
        block.text for block in response.content if getattr(block, "text", None)
    )
    if not texto:
        raise AIAdvisorError("La API de Claude devolvió una respuesta vacía.")
    return texto
