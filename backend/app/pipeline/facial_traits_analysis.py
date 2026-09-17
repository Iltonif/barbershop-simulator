"""
Etapa adicional del pipeline: análisis automático de rasgos faciales para
el perfil de visagismo del cliente.

Motivación (petición de Pedro): al registrar el perfil de un cliente,
detectar automáticamente rasgos como perfil de nariz, proyección de
orejas, separación de ojos, forma de cejas, asimetrías (p. ej. "un ojo
más abierto que el otro") y uso de gafas -- en vez de depender solo de
que el barbero los rellene a mano en `FacialFeaturesProfileIn`
(`app/api/schemas.py`).

Entrada: 3 fotos guiadas del cliente -- frontal, perfil izquierdo y
perfil derecho (decisión de producto: sin vídeo, ver CLAUDE.md). Salida:
un `FacialFeaturesProfileIn`-compatible (dict) con los campos que se han
podido detectar, más una lista de avisos (p.ej. "no se detectó cara en
la foto de perfil derecho") y un resumen en texto de las asimetrías
detectadas con sus porcentajes.

Reutiliza dos piezas YA existentes del pipeline, sin añadir ninguna
dependencia ni modelo nuevo:
- `face_analysis.analyze_face()` (68 landmarks, esquema dlib/iBUG) para
  ojos, cejas y una aproximación del perfil de nariz.
- `hair_segmentation.segment_face_parts()` (BiSeNet, licencia MIT, ya
  vendorizado para segmentar el pelo) para orejas y gafas: ese mismo
  modelo ya clasifica las clases "l_ear"/"r_ear"/"eye_g" como parte de
  su salida estándar de 19 clases (CelebAMask-HQ), documentado en el
  docstring de `hair_segmentation.py` como disponible "por si
  `compositor.py` las necesita más adelante" -- este módulo es el primer
  consumidor real de esas clases. Dicho de otro modo: NO hace falta un
  "detector de orejas" nuevo y separado (se evaluó explícitamente si
  existía algún modelo dedicado de orejas con licencia permisiva de uso
  comercial -- no se encontró ninguno maduro y mantenido; usar la
  segmentación de BiSeNet que ya tenemos es la opción real, sin coste de
  licencia ni de mantenimiento añadido).

Limitaciones honestas (igual que el resto de `face_analysis.py` ya
documenta sus propias aproximaciones):
- El modelo de landmarks (68 puntos, entrenado sobre caras frontales/
  moderadamente giradas) puede no detectar ninguna cara en un perfil muy
  cerrado (90°) -- por eso se pide al barbero un perfil "de 3/4", no un
  perfil puro, y cada foto que falla se reporta como aviso en vez de
  romper el análisis completo.
- La clasificación de perfil de nariz (convexo/cóncavo/recto) es una
  heurística geométrica 2D (desviación de la punta de la nariz respecto
  a la línea entrecejo-mentón), no una medición profilométrica real.
- La proyección de orejas es una proporción entre el ancho de la máscara
  de oreja y el ancho de cabeza (máscara de piel) en la misma foto de
  perfil -- un proxy razonable, no una medición en milímetros.
- RGPD: ninguna de las 3 fotos se guarda en ningún sitio -- se procesan
  en memoria en `clients_routes.override_visagismo_auto_analysis` y se
  descartan justo después de extraer estas categorías, igual que ya hace
  `POST /api/simulate` con la foto de simulación (ver CLAUDE.md).
"""

from dataclasses import dataclass, field

import numpy as np

from app.pipeline import face_analysis, hair_segmentation

# Mismo esquema de 68 puntos (dlib/iBUG) que usa `face_analysis.py`, 0-indexado.
# Se redefine aquí (en vez de importar los `_privados` de ese módulo) para no
# acoplar este módulo a detalles internos de `face_analysis.py`.
_RIGHT_EYE = list(range(36, 42))
_LEFT_EYE = list(range(42, 48))
_RIGHT_EYEBROW = list(range(17, 22))
_LEFT_EYEBROW = list(range(22, 27))

_SKIN_CLASS = 1
_EYE_GLASSES_CLASS = 6
_L_EAR_CLASS = 7
_R_EAR_CLASS = 8

# Umbrales de las heurísticas de abajo. Elegidos por criterio razonable,
# no calibrados contra un dataset real todavía -- si en el uso real dan
# demasiados falsos positivos/negativos, son el primer sitio a ajustar.
_EYE_ASYMMETRY_THRESHOLD_PERCENT = 15.0
_EAR_PROJECTION_RATIO_THRESHOLD = 0.16
_GLASSES_MIN_COVERAGE_RATIO = 0.004
_NOSE_CONVEXITY_THRESHOLD = 0.03  # proporción del alto de cara


@dataclass
class FacialTraitsResult:
    facial_features_profile: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    detected_anomalies_notes: str | None = None


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _eye_aspect_ratio(points: np.ndarray, eye_indices: list[int]) -> float:
    """EAR (Eye Aspect Ratio) clásico de 6 puntos: p1..p6 en orden
    (esquina externa, párpado superior x2, esquina interna, párpado
    inferior x2). Valor más alto = ojo más abierto."""
    p1, p2, p3, p4, p5, p6 = points[eye_indices]
    vertical = _dist(p2, p6) + _dist(p3, p5)
    horizontal = 2 * _dist(p1, p4)
    if horizontal < 1e-6:
        return 0.0
    return vertical / horizontal


def _eyebrow_type(points: np.ndarray, eyebrow_indices: list[int]) -> str:
    """Clasificación aproximada según la curvatura de la ceja: compara la
    altura del punto central con la línea entre sus dos extremos.

    Solo distingue "recta/baja" de "arqueada" -- "prominent_ridge" (arco
    superciliar marcado, cuestión de hueso, no de forma 2D de la ceja) no
    se puede inferir de forma fiable con estos landmarks, así que esa
    categoría se deja para que el barbero la confirme a mano."""
    pts = points[eyebrow_indices]
    start, mid, end = pts[0], pts[len(pts) // 2], pts[-1]
    baseline_y = (start[1] + end[1]) / 2
    arch_height = baseline_y - mid[1]  # positivo si el centro está más arriba
    eyebrow_width = max(_dist(start, end), 1e-6)
    if (arch_height / eyebrow_width) > 0.08:
        return "arched"
    return "straight_low"


def _eye_spacing(points: np.ndarray) -> str:
    right_inner, left_inner = points[39], points[42]
    right_width = _dist(points[36], points[39])
    left_width = _dist(points[42], points[45])
    avg_eye_width = max((right_width + left_width) / 2, 1e-6)
    inner_gap = _dist(right_inner, left_inner)
    ratio = inner_gap / avg_eye_width
    if ratio < 0.9:
        return "close_set"
    if ratio > 1.5:
        return "wide_set"
    return "proportional"


def _analyze_frontal(image_bgr: np.ndarray, result: FacialTraitsResult) -> None:
    face = face_analysis.analyze_face(image_bgr)
    if face is None:
        result.warnings.append(
            "No se detectó ninguna cara en la foto frontal: separación de "
            "ojos, forma de cejas, simetría ocular y gafas no se han "
            "podido analizar."
        )
        return

    points = face.landmarks
    profile = result.facial_features_profile

    profile["eye_spacing"] = _eye_spacing(points)

    right_eyebrow = _eyebrow_type(points, _RIGHT_EYEBROW)
    left_eyebrow = _eyebrow_type(points, _LEFT_EYEBROW)
    # Si ambas cejas coinciden, se guarda esa categoría; si difieren, se
    # deja sin auto-rellenar (mejor pedir al barbero que lo confirme que
    # forzar una de las dos).
    if right_eyebrow == left_eyebrow:
        profile["eyebrow_type"] = right_eyebrow

    ear_right = _eye_aspect_ratio(points, _RIGHT_EYE)
    ear_left = _eye_aspect_ratio(points, _LEFT_EYE)
    bigger = max(ear_right, ear_left)
    if bigger > 1e-6:
        diff_percent = abs(ear_right - ear_left) / bigger * 100
    else:
        diff_percent = 0.0

    if diff_percent >= _EYE_ASYMMETRY_THRESHOLD_PERCENT:
        more_open = "derecho" if ear_right > ear_left else "izquierdo"
        profile["eye_symmetry"] = "asymmetric"
        profile["eye_symmetry_percent"] = round(diff_percent, 1)
        result.detected_anomalies_notes = (
            f"Asimetría ocular detectada: el ojo {more_open} está "
            f"aproximadamente un {diff_percent:.0f}% más abierto que el otro "
            "(medido por proporción de apertura del ojo en la foto frontal)."
        )
    else:
        profile["eye_symmetry"] = "symmetric"
        profile["eye_symmetry_percent"] = round(diff_percent, 1)

    # Gafas: clase "eye_g" de la segmentación BiSeNet ya vendorizada para
    # el pelo (ver docstring del módulo) -- no es un modelo nuevo.
    try:
        parsing = hair_segmentation.segment_face_parts(image_bgr)
        glasses_mask = parsing == _EYE_GLASSES_CLASS
        coverage = hair_segmentation.mask_coverage_ratio(glasses_mask)
        profile["has_glasses"] = coverage >= _GLASSES_MIN_COVERAGE_RATIO
    except FileNotFoundError:
        # Pesos de BiSeNet no descargados en este despliegue -- no bloquea
        # el resto del análisis, solo se omite el dato de gafas.
        result.warnings.append(
            "No se pudo comprobar el uso de gafas (modelo de segmentación "
            "no disponible en este despliegue)."
        )


def _signed_deviation(point: np.ndarray, line_start: np.ndarray, line_vec: np.ndarray, line_len: float) -> float:
    """Distancia con signo de `point` a la recta que pasa por `line_start`
    con dirección `line_vec` (producto cruzado 2D / longitud de la línea)."""
    cross = line_vec[0] * (point[1] - line_start[1]) - line_vec[1] * (point[0] - line_start[0])
    return cross / line_len


def _nose_profile_type(points: np.ndarray) -> str | None:
    """Heurística 2D: desviación perpendicular de la punta de la nariz
    respecto a la línea entrecejo (27) - mentón (8), normalizada por el
    alto de cara. Ver limitaciones en el docstring del módulo.

    El signo de esa desviación, por sí solo, depende de hacia qué lado
    mira la cara en la foto (perfil izquierdo y derecho dan signos
    opuestos para la MISMA nariz) -- por eso se normaliza usando el labio
    superior (punto 51) como referencia de "hacia dónde está el frente de
    la cara": el labio superior siempre sobresale ligeramente hacia
    delante respecto a esa misma línea, sea cual sea el lado fotografiado,
    así que su signo sirve para poner el de la nariz en un eje comparable
    entre ambos perfiles."""
    glabella, chin, tip, upper_lip = points[27], points[8], points[33], points[51]
    face_height = max(_dist(glabella, chin), 1e-6)

    line_vec = chin - glabella
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-6:
        return None

    mouth_dev = _signed_deviation(upper_lip, glabella, line_vec, line_len)
    tip_dev = _signed_deviation(tip, glabella, line_vec, line_len)

    forward_sign = 1.0 if mouth_dev >= 0 else -1.0
    ratio = (tip_dev * forward_sign) / face_height

    if abs(ratio) < _NOSE_CONVEXITY_THRESHOLD:
        return "straight"
    return "convex_prominent_nose" if ratio > 0 else "concave"


def _ear_projection(image_bgr: np.ndarray) -> str | None:
    """Proporción entre el ancho de la máscara de oreja visible (la que
    tenga más píxeles: la cercana a la cámara en una foto de perfil) y el
    ancho de la máscara de piel (proxy del ancho de cabeza) en esa misma
    foto."""
    parsing = hair_segmentation.segment_face_parts(image_bgr)

    def _mask_width(class_idx: int) -> float:
        ys, xs = np.where(parsing == class_idx)
        if xs.size == 0:
            return 0.0
        return float(xs.max() - xs.min())

    left_ear_px = int(np.count_nonzero(parsing == _L_EAR_CLASS))
    right_ear_px = int(np.count_nonzero(parsing == _R_EAR_CLASS))
    ear_class = _L_EAR_CLASS if left_ear_px >= right_ear_px else _R_EAR_CLASS
    if max(left_ear_px, right_ear_px) == 0:
        return None

    ear_width = _mask_width(ear_class)
    skin_width = _mask_width(_SKIN_CLASS)
    if skin_width < 1e-6:
        return None

    ratio = ear_width / skin_width
    return "prominent_protruding" if ratio >= _EAR_PROJECTION_RATIO_THRESHOLD else "flat"


def _analyze_profile_photo(image_bgr: np.ndarray, side_label: str, result: FacialTraitsResult) -> None:
    profile = result.facial_features_profile
    face = face_analysis.analyze_face(image_bgr)

    if face is not None:
        nose_type = _nose_profile_type(face.landmarks)
        if nose_type is not None and "profile_type" not in profile:
            profile["profile_type"] = nose_type
    else:
        result.warnings.append(
            f"No se detectó ninguna cara en la foto de perfil {side_label}: "
            "el perfil de nariz no se ha podido estimar desde ese lado. "
            "Prueba con un giro algo menos cerrado (3/4 en vez de perfil puro)."
        )

    try:
        ear_result = _ear_projection(image_bgr)
        if ear_result is not None and "ears_projection" not in profile:
            profile["ears_projection"] = ear_result
        elif ear_result is None:
            result.warnings.append(
                f"No se detectó ninguna oreja en la foto de perfil {side_label}."
            )
    except FileNotFoundError:
        result.warnings.append(
            "No se pudo estimar la proyección de orejas (modelo de "
            "segmentación no disponible en este despliegue)."
        )


def analyze_facial_traits(
    frontal_bgr: np.ndarray,
    left_profile_bgr: np.ndarray,
    right_profile_bgr: np.ndarray,
) -> FacialTraitsResult:
    """Punto de entrada: analiza las 3 fotos guiadas y devuelve los
    campos de `FacialFeaturesProfileIn` que se han podido detectar, más
    avisos de lo que no se pudo analizar. Nunca lanza excepción por una
    foto individual sin cara/oreja -- esos casos se acumulan como avisos
    para que el barbero rellene esos campos concretos a mano."""
    result = FacialTraitsResult()

    _analyze_frontal(frontal_bgr, result)
    # El perfil derecho es el que, por convención de esta función, se
    # analiza en segundo lugar -- si ambos lados dan una detección de
    # nariz/orejas válida, se conserva la primera (izquierda) por ser
    # determinista; en la práctica ambos lados deberían coincidir.
    _analyze_profile_photo(left_profile_bgr, "izquierdo", result)
    _analyze_profile_photo(right_profile_bgr, "derecho", result)

    return result
