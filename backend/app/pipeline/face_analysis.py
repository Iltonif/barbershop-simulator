"""
Etapa 1 del pipeline: análisis facial.

Detecta landmarks faciales, estima la pose de la cabeza (yaw/pitch/roll),
aproxima la forma de la cara y comprueba simetría. Esto es lo que permite
anclar el corte generado sobre la cara real del cliente, en vez de "flotar"
desalineado.

Implementación: detector de caras Haar Cascade de OpenCV + Facemark LBF
(esquema clásico de 68 puntos, dlib/iBUG). NO se usa mediapipe aquí: en
macOS, mediapipe 1.0.1 (publicado hace pocas semanas) revienta con un
crash nativo ("Check failed: service_ Service is unavailable" dentro de
DrishtiMetalHelper) al detectar la cara, incluso forzando delegate=CPU —
parece un bug real de esa versión tan reciente. OpenCV/LBF es mucho más
antiguo y estable, sin ningún componente de GPU que pueda fallar así.

IMPORTANTE: el esquema de 68 puntos NO incluye puntos de frente/nacimiento
del pelo (solo mandíbula, cejas, ojos, nariz y boca), a diferencia del
esquema de 478 puntos de mediapipe que se usaba antes. `hairline_points`
es por tanto una extrapolación aproximada hacia arriba desde las cejas,
no una detección real del nacimiento del pelo — ver el TODO en
`_estimate_hairline`.

Requiere el modelo LBF: ejecutar una vez
`python -m app.pipeline.download_landmark_model` antes de usar este módulo.
"""

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from app.pipeline.download_landmark_model import CASCADE_PATH, MODEL_PATH

# Índices del esquema de 68 puntos (dlib/iBUG), 0-indexados.
_JAW = list(range(0, 17))
_RIGHT_EYEBROW = list(range(17, 22))
_LEFT_EYEBROW = list(range(22, 27))
_NOSE_BRIDGE = list(range(27, 31))
_NOSE_BOTTOM = list(range(31, 36))
_RIGHT_EYE = list(range(36, 42))
_LEFT_EYE = list(range(42, 48))
_MOUTH_OUTER = list(range(48, 60))


@dataclass
class FaceAnalysisResult:
    landmarks: np.ndarray  # (68, 2) puntos en coordenadas de píxel (x, y)
    yaw: float
    pitch: float
    roll: float
    face_shape: str  # "ovalada" | "redonda" | "cuadrada" | "alargada" | "desconocida"
    is_symmetric_enough: bool
    hairline_points: np.ndarray  # aproximación, ver docstring del módulo


@lru_cache(maxsize=1)
def _load_face_detector() -> cv2.CascadeClassifier:
    # No usar cv2.data.haarcascades: OpenCV 5.0 dejó de incluir estos
    # archivos dentro del paquete, así que se descargan aparte (ver
    # download_landmark_model.py) y se cargan desde ahí.
    if not CASCADE_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el clasificador en {CASCADE_PATH}. Ejecuta primero:\n"
            "    python -m app.pipeline.download_landmark_model"
        )
    classifier = cv2.CascadeClassifier(str(CASCADE_PATH))
    if classifier.empty():
        raise RuntimeError(
            f"El clasificador en {CASCADE_PATH} no se pudo cargar (archivo vacío o corrupto)."
        )
    return classifier


@lru_cache(maxsize=1)
def _load_facemark():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {MODEL_PATH}. Ejecuta primero:\n"
            "    python -m app.pipeline.download_landmark_model"
        )
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel(str(MODEL_PATH))
    return facemark


def analyze_face(image_bgr: np.ndarray) -> FaceAnalysisResult | None:
    """Analiza una imagen (BGR, formato OpenCV) y devuelve landmarks + pose.

    Devuelve None si no se detecta ninguna cara.
    """
    detector = _load_face_detector()
    facemark = _load_facemark()

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        return None

    # Si detecta varias caras, se queda con la más grande (asumimos que es
    # el sujeto principal de la foto).
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    faces_arr = np.array([faces[0]])

    ok, landmark_sets = facemark.fit(gray, faces_arr)
    if not ok or len(landmark_sets) == 0:
        return None

    # El shape exacto que devuelve OpenCV varía según versión
    # ((1, 68, 2) o (68, 1, 2)); aplanar a (68, 2) es robusto a ambos.
    points = np.array(landmark_sets[0]).reshape(-1, 2)
    h, w = image_bgr.shape[:2]

    yaw, pitch, roll = _estimate_head_pose(points, w, h)
    face_shape = _estimate_face_shape(points)
    is_symmetric_enough = _estimate_symmetry(points, w)
    hairline_points = _estimate_hairline(points)

    return FaceAnalysisResult(
        landmarks=points,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        face_shape=face_shape,
        is_symmetric_enough=is_symmetric_enough,
        hairline_points=hairline_points,
    )


def _eye_center(points: np.ndarray, eye_indices: list[int]) -> np.ndarray:
    return points[eye_indices].mean(axis=0)


def _estimate_head_pose(points: np.ndarray, w: int, h: int) -> tuple[float, float, float]:
    """Estimación aproximada de yaw/pitch/roll a partir de puntos clave.

    TODO: sustituir por solvePnP con un modelo 3D de cara canónico para
    obtener ángulos más precisos si el ángulo de la foto es muy pronunciado.
    """
    right_eye = _eye_center(points, _RIGHT_EYE)
    left_eye = _eye_center(points, _LEFT_EYE)
    nose_tip = points[30]

    dx = left_eye[0] - right_eye[0]
    dy = left_eye[1] - right_eye[1]
    roll = float(np.degrees(np.arctan2(dy, dx)))

    eyes_mid_x = (left_eye[0] + right_eye[0]) / 2
    yaw = float((nose_tip[0] - eyes_mid_x) / w * 180)

    eyes_mid_y = (left_eye[1] + right_eye[1]) / 2
    pitch = float((nose_tip[1] - eyes_mid_y) / h * 180)

    return yaw, pitch, roll


def _estimate_face_shape(points: np.ndarray) -> str:
    """Heurística simple de forma de cara a partir de proporciones.

    Con el esquema de 68 puntos no hay puntos de frente, así que la
    "longitud de cara" se aproxima desde el entrecejo (punto 27) hasta la
    barbilla (punto 8), más corta que la longitud real de la cara.

    TODO: esto es una aproximación razonable pero no un clasificador
    entrenado. Si la recomendación de corte por forma de cara se vuelve
    una feature seria, entrenar/usar un clasificador dedicado.
    """
    jaw_width = np.linalg.norm(points[0] - points[16])
    face_length = np.linalg.norm(points[27] - points[8])
    cheekbone_width = np.linalg.norm(points[1] - points[15])  # aproximación

    ratio = face_length / max(jaw_width, 1e-6)

    if ratio > 1.3:
        return "alargada"
    if abs(jaw_width - cheekbone_width) < 0.05 * cheekbone_width:
        return "cuadrada"
    if ratio < 1.05:
        return "redonda"
    return "ovalada"


def _estimate_symmetry(points: np.ndarray, image_width: float) -> bool:
    """Compara distancias de puntos simétricos respecto al eje central de la cara."""
    center_x = points[30][0]  # punta de la nariz como referencia del eje
    pairs = [(36, 45), (48, 54), (0, 16)]  # ojos, comisuras de boca, mandíbula

    deviations = []
    for right_idx, left_idx in pairs:
        right_dist = abs(points[right_idx][0] - center_x)
        left_dist = abs(points[left_idx][0] - center_x)
        deviations.append(abs(left_dist - right_dist) / image_width)

    return float(np.mean(deviations)) < 0.02


def _estimate_hairline(points: np.ndarray) -> np.ndarray:
    """Aproxima el nacimiento del pelo extrapolando hacia arriba desde las
    cejas, ya que el esquema de 68 puntos no incluye puntos de frente.

    TODO: esto es una aproximación gruesa. Si se necesita precisión real
    en el nacimiento del pelo (p.ej. para `head_mesh.py`), considerar un
    detector de landmarks con más puntos que sí cubra la frente, o una
    heurística basada en la máscara de piel/pelo de `hair_segmentation.py`.
    """
    eyebrow_points = points[_RIGHT_EYEBROW + _LEFT_EYEBROW]
    jaw_top = points[27]  # entrecejo, como referencia de escala
    chin = points[8]
    face_height = np.linalg.norm(jaw_top - chin)

    offset = np.array([0, -face_height * 0.9])  # hacia arriba en píxeles
    return eyebrow_points + offset
