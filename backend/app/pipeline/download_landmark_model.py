"""
Descarga el modelo LBF de OpenCV para detección de landmarks faciales
(68 puntos, esquema clásico dlib/iBUG).

Se usa en vez de mediapipe: en macOS, mediapipe 1.0.1 revienta con un
crash nativo ("Check failed: service_ Service is unavailable" dentro de
DrishtiMetalHelper) al intentar detectar la cara, incluso forzando
delegate=CPU — parece un bug de esta versión tan reciente (publicada hace
pocas semanas). El detector LBF de OpenCV es mucho más antiguo y estable,
sin ningún componente de GPU que pueda fallar así.

Uso:
    python -m app.pipeline.download_landmark_model
"""

import ssl
import urllib.request
from pathlib import Path

_MODEL_URL = (
    "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml"
)
_CASCADE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/"
    "haarcascade_frontalface_default.xml"
)

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "lbfmodel.yaml"
CASCADE_PATH = MODEL_DIR / "haarcascade_frontalface_default.xml"


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def download_landmark_model() -> Path:
    if MODEL_PATH.exists():
        print(f"Ya existe el modelo en {MODEL_PATH}, no se vuelve a descargar.")
        return MODEL_PATH

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Descargando {_MODEL_URL} a {MODEL_PATH} ...")

    context = _ssl_context()
    with urllib.request.urlopen(_MODEL_URL, context=context) as response, open(MODEL_PATH, "wb") as out_file:
        out_file.write(response.read())

    print("Descarga completada.")
    return MODEL_PATH


def download_face_cascade() -> Path:
    """Descarga el clasificador Haar Cascade para detectar caras.

    OpenCV 5.0 dejó de incluir estos archivos dentro del propio paquete
    (`cv2.data.haarcascades` apunta a una carpeta vacía en esta versión),
    así que hay que descargarlo aparte, igual que el modelo LBF.
    """
    if CASCADE_PATH.exists():
        print(f"Ya existe el clasificador en {CASCADE_PATH}, no se vuelve a descargar.")
        return CASCADE_PATH

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Descargando {_CASCADE_URL} a {CASCADE_PATH} ...")

    context = _ssl_context()
    with urllib.request.urlopen(_CASCADE_URL, context=context) as response, open(CASCADE_PATH, "wb") as out_file:
        out_file.write(response.read())

    print("Descarga completada.")
    return CASCADE_PATH


if __name__ == "__main__":
    download_landmark_model()
    download_face_cascade()
