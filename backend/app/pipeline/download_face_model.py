"""
Descarga el modelo `face_landmarker.task` que necesita `face_analysis.py`.

Mediapipe 1.0 eliminó la API "legacy" (`mp.solutions.face_mesh`) que
usábamos al principio, en favor de la nueva "Tasks API"
(`mediapipe.tasks.python.vision.FaceLandmarker`). Esa API no trae el
modelo empaquetado: hay que descargarlo aparte, una vez, desde el
repositorio oficial de modelos de Google (público, sin autenticación).

Uso:
    python -m app.pipeline.download_face_model
"""

import urllib.request
from pathlib import Path

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "face_landmarker.task"


def download_face_model() -> Path:
    if MODEL_PATH.exists():
        print(f"Ya existe el modelo en {MODEL_PATH}, no se vuelve a descargar.")
        return MODEL_PATH

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Descargando {_MODEL_URL} a {MODEL_PATH} ...")
    urllib.request.urlretrieve(_MODEL_URL, MODEL_PATH)
    print("Descarga completada.")
    return MODEL_PATH


if __name__ == "__main__":
    download_face_model()
