"""
Etapa adicional del pipeline: análisis automático de rasgos faciales para
el perfil de visagismo del cliente.

Motivación (petición de Pedro): al registrar el perfil de un cliente,
detectar automáticamente rasgos como separación de ojos, asimetrías (p.
ej. "un ojo más abierto que el otro") y uso de gafas -- en vez de
depender solo de que el barbero los rellene a mano en
`FacialFeaturesProfileIn` (`app/api/schemas.py`). También se intentó con
perfil de nariz, proyección de orejas y forma de cejas; se descartaron
tras probarlos con fotos reales (ver comentarios más abajo y CLAUDE.md).

Entrada: 3 fotos guiadas del cliente -- frontal, perfil izquierdo y
perfil derecho (decisión de producto: sin vídeo, ver CLAUDE.md) --,
aunque hoy solo se analiza la frontal. Salida: un
`FacialFeaturesProfileIn`-compatible (dict) con los campos que se han
podido detectar, más una lista de avisos (p.ej. "la foto frontal tiene
la cabeza algo girada") y un resumen en texto de las asimetrías
detectadas con sus porcentajes.

Reutiliza dos piezas YA existentes del pipeline, sin añadir ninguna
dependencia ni modelo nuevo:
- `face_analysis.analyze_face()` (68 landmarks, esquema dlib/iBUG) para
  los ojos.
- `hair_segmentation.segment_face_parts()` (BiSeNet, licencia MIT, ya
  vendorizado para segmentar el pelo) para las gafas: ese mismo modelo ya
  clasifica la clase "eye_g" como parte de su salida estándar de 19
  clases (CelebAMask-HQ).

Limitaciones honestas (igual que el resto de `face_analysis.py` ya
documenta sus propias aproximaciones):
- Los umbrales están calibrados con 61 fotos frontales reales (ver
  comentarios junto a las constantes y CLAUDE.md), pero ninguna de esas
  personas tenía asimetría ocular real ni ojos muy juntos/separados: está
  comprobado que no da falsas alarmas, no que detecte los casos reales.
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

_SKIN_CLASS = 1
_EYE_GLASSES_CLASS = 6

# --- Umbrales de la foto frontal: CALIBRADOS con fotos reales ---
# Medidos ejecutando este mismo pipeline (Haar + LBF + BiSeNet) sobre 61
# fotos frontales de un conjunto público de pruebas, varias de ellas de la
# misma persona (lo que permite separar el ruido de medición de la
# variación real entre personas). Detalle de cada decisión en CLAUDE.md,
# sección "Calibración de umbrales".
#
# Simetría ocular: en caras normales la diferencia de EAR llega hasta ~15%
# solo por expresión, pose y ruido de landmarks (p95 = 10%, máximo 15,1%).
# Con el 15% anterior se marcaba "asimétrica" a gente que no lo es. El 20%
# deja margen sobre ese máximo y coincide aproximadamente con la diferencia
# que ya se aprecia a simple vista (~2 mm sobre una apertura de ~10 mm).
_EYE_ASYMMETRY_THRESHOLD_PERCENT = 20.0
# Giro de cabeza = desplazamiento horizontal de la punta de la nariz
# respecto al punto medio entre los ojos, dividido por la distancia entre
# ojos. Era el principal causante de falsas asimetrías (correlación 0,46):
# con la cara girada, el ojo lejano sale escorzado y parece "más cerrado".
# Por encima de este valor no se mide simetría ni separación de ojos y se
# pide repetir la foto. En fotos frontales normales la mediana es 0,04.
_MAX_FRONTAL_TURN = 0.15
# Separación de ojos: distancia intercantal (entre lagrimales) dividida por
# el ancho de la cara. La fórmula anterior (entre ancho de ojo) daba 0/61
# "juntos" y 10/61 "separados" y apenas distinguía entre personas: la
# misma persona variaba casi tanto como dos personas distintas. Esta es
# algo más estable, pero sigue siendo ruidosa, así que solo se clasifica
# como juntos/separados en valores claramente extremos (rango observado en
# caras normales: 0,222-0,292).
_CLOSE_SET_MAX_RATIO = 0.215
_WIDE_SET_MIN_RATIO = 0.300
# Gafas: píxeles de gafas / píxeles de piel (no / imagen entera, que
# dependía de lo cerca que estuviera la cámara). Con gafas: 0,21-0,22;
# sin gafas: como mucho 0,003. Margen amplio a ambos lados.
_GLASSES_MIN_SKIN_RATIO = 0.05


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


# Forma de cejas: deliberadamente NO se detecta automáticamente. Se probó
# (curvatura de los 5 puntos de ceja del modelo LBF) y, al comparar con
# las fotos, el resultado iba al revés de lo que ve una persona: las cejas
# depiladas y muy arqueadas salían "rectas" y las cejas gruesas y planas
# salían "arqueadas". Los 5 puntos siguen una plantilla que no reproduce
# bien dónde está el pico real de la ceja. Ningún umbral arregla eso, así
# que `eyebrow_type` se queda como campo manual del barbero.


def _frontal_turn(points: np.ndarray) -> float:
    """Giro horizontal aproximado de la cabeza en la foto frontal:
    desplazamiento de la punta de la nariz (30) respecto al punto medio
    entre los centros de los ojos, en unidades de distancia entre ojos.
    0 = mirando de frente. Independiente de la resolución de la foto (a
    diferencia de `FaceAnalysisResult.yaw`, que divide por el ancho de la
    imagen)."""
    right_center = points[_RIGHT_EYE].mean(axis=0)
    left_center = points[_LEFT_EYE].mean(axis=0)
    interocular = max(_dist(right_center, left_center), 1e-6)
    eyes_mid_x = (right_center[0] + left_center[0]) / 2
    return float(abs(points[30][0] - eyes_mid_x) / interocular)


def _eye_spacing(points: np.ndarray) -> str:
    """Distancia intercantal (entre lagrimales, 39-42) / ancho de cara
    (extremos de la mandíbula, 0-16). Ver umbrales arriba.

    Ojo: con orejas muy separadas de la cabeza, el modelo LBF a veces
    coloca los puntos 0 y 16 sobre las propias orejas, así que el "ancho
    de cara" sale algo mayor y la separación algo menor. Los umbrales
    están calibrados con esta misma medida, y los extremos son amplios,
    pero es otro motivo para no bajarlos sin más datos."""
    face_width = max(_dist(points[0], points[16]), 1e-6)
    ratio = _dist(points[39], points[42]) / face_width
    if ratio < _CLOSE_SET_MAX_RATIO:
        return "close_set"
    if ratio > _WIDE_SET_MIN_RATIO:
        return "wide_set"
    return "proportional"


def _has_glasses(parsing: np.ndarray) -> bool:
    """`parsing` = mapa de clases de BiSeNet. Proporción de píxeles de
    gafas respecto a los de piel (ver `_GLASSES_MIN_SKIN_RATIO`)."""
    skin_px = int(np.count_nonzero(parsing == _SKIN_CLASS))
    if skin_px == 0:
        return False
    glasses_px = int(np.count_nonzero(parsing == _EYE_GLASSES_CLASS))
    return glasses_px / skin_px >= _GLASSES_MIN_SKIN_RATIO


def _measure_eye_symmetry(points: np.ndarray, result: FacialTraitsResult) -> None:
    profile = result.facial_features_profile
    ear_right = _eye_aspect_ratio(points, _RIGHT_EYE)
    ear_left = _eye_aspect_ratio(points, _LEFT_EYE)
    bigger = max(ear_right, ear_left)
    diff_percent = abs(ear_right - ear_left) / bigger * 100 if bigger > 1e-6 else 0.0

    profile["eye_symmetry_percent"] = round(diff_percent, 1)
    if diff_percent >= _EYE_ASYMMETRY_THRESHOLD_PERCENT:
        more_open = "derecho" if ear_right > ear_left else "izquierdo"
        profile["eye_symmetry"] = "asymmetric"
        result.detected_anomalies_notes = (
            f"Asimetría ocular detectada: el ojo {more_open} está "
            f"aproximadamente un {diff_percent:.0f}% más abierto que el otro "
            "(medido por proporción de apertura del ojo en la foto frontal)."
        )
    else:
        profile["eye_symmetry"] = "symmetric"


def _analyze_frontal(image_bgr: np.ndarray, result: FacialTraitsResult) -> None:
    face = face_analysis.analyze_face(image_bgr)
    if face is None:
        result.warnings.append(
            "No se detectó ninguna cara en la foto frontal: separación de "
            "ojos, simetría ocular y gafas no se han podido analizar."
        )
        return

    points = face.landmarks
    profile = result.facial_features_profile

    # Con la cabeza girada, el ojo lejano sale escorzado: parece más
    # cerrado y más cerca del otro. Medir simetría o separación así da
    # falsos positivos (ver `_MAX_FRONTAL_TURN`), así que se omiten y se
    # pide repetir la foto. Las gafas sí se siguen comprobando.
    turn = _frontal_turn(points)
    if turn > _MAX_FRONTAL_TURN:
        result.warnings.append(
            "La foto frontal tiene la cabeza algo girada: no se han medido "
            "la simetría ocular ni la separación de ojos para no dar un "
            "resultado falso. Repite la foto con el cliente mirando "
            "directamente a la cámara."
        )
    else:
        profile["eye_spacing"] = _eye_spacing(points)
        _measure_eye_symmetry(points, result)

    # Gafas: clase "eye_g" de la segmentación BiSeNet ya vendorizada para
    # el pelo (ver docstring del módulo) -- no es un modelo nuevo.
    try:
        parsing = hair_segmentation.segment_face_parts(image_bgr)
        profile["has_glasses"] = _has_glasses(parsing)
    except FileNotFoundError:
        # Pesos de BiSeNet no descargados en este despliegue -- no bloquea
        # el resto del análisis, solo se omite el dato de gafas.
        result.warnings.append(
            "No se pudo comprobar el uso de gafas (modelo de segmentación "
            "no disponible en este despliegue)."
        )


# Nariz y orejas: deliberadamente NO se detectan automáticamente. Se
# probaron con fotos reales (ver CLAUDE.md, "Calibración de umbrales"):
# - Perfil de nariz: el detector de caras (Haar frontal + landmarks LBF)
#   no encuentra la cara en una foto de perfil, así que nunca se llegaba a
#   medir. En una foto de 3/4 sí podría encontrarla, pero el propio giro
#   deforma justo la geometría de la nariz que se quería medir.
# - Proyección de orejas: desde el perfil, BiSeNet (entrenado con caras
#   frontales) confunde barbilla y boca con oreja. Desde la foto frontal sí
#   segmenta bien las orejas, pero la medida resultante es ruido: las dos
#   orejas de una misma cara, que son casi iguales, daban valores sin
#   ninguna relación entre sí (correlación -0,05).
# Ambos campos quedan como manuales del barbero.


def merge_detected_features(existing: dict, detected: dict) -> dict:
    """Fusiona lo detectado en las fotos con lo que ya hubiera en la ficha,
    sin pisar nunca un valor que el barbero haya puesto a mano.

    "Vacío" no es solo "la clave no existe": al guardar la ficha con
    `PATCH .../visagismo-profile`, Pydantic escribe TODOS los campos, con
    `None` en los que no se rellenaron. Con `dict.setdefault` (la versión
    anterior) esas claves contaban como ya rellenas y, en cuanto la ficha
    se había guardado una vez, el análisis automático no volvía a rellenar
    nada (devolvía 200 sin avisos, pero sin guardar ningún resultado).

    `has_glasses` además acepta `False` como vacío: la página lo guarda con
    una casilla, que no puede distinguir "no lleva gafas" de "sin
    especificar". La detección de gafas fue la más fiable de la
    calibración, así que una casilla sin marcar no debe bloquearla.

    `eye_symmetry_percent` solo se escribe junto con `eye_symmetry`, para
    no dejar un porcentaje medido que contradiga una simetría puesta a mano."""
    merged = dict(existing)

    def is_empty(key):
        value = merged.get(key)
        return value is None or value == "" or (key == "has_glasses" and value is False)

    symmetry_filled = False
    for key, value in detected.items():
        if key == "eye_symmetry_percent":
            continue
        if is_empty(key):
            merged[key] = value
            symmetry_filled = symmetry_filled or key == "eye_symmetry"
    if symmetry_filled and "eye_symmetry_percent" in detected:
        merged["eye_symmetry_percent"] = detected["eye_symmetry_percent"]
    return merged


def analyze_facial_traits(
    frontal_bgr: np.ndarray,
    left_profile_bgr: np.ndarray | None = None,
    right_profile_bgr: np.ndarray | None = None,
) -> FacialTraitsResult:
    """Punto de entrada: analiza la foto frontal y devuelve los campos de
    `FacialFeaturesProfileIn` que se han podido detectar, más avisos de lo
    que no se pudo analizar. Nunca lanza excepción por una foto sin cara:
    esos casos se acumulan como avisos para que el barbero rellene esos
    campos a mano.

    Las fotos de perfil se aceptan (el flujo sigue pidiendo 3 fotos, por
    decisión de producto) pero hoy no se analizan: ver el comentario
    justo encima. Se dejan en la firma para no tener que cambiar el
    endpoint ni la página si más adelante se incorpora un modelo que sí
    funcione de perfil."""
    result = FacialTraitsResult()
    _analyze_frontal(frontal_bgr, result)
    return result
