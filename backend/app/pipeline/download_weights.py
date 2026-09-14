"""
Descarga los pesos pre-entrenados de BiSeNet face-parsing (79999_iter.pth).

El autor original solo los aloja en Google Drive (no hay mirror oficial en
Hugging Face ni en un release de GitHub), así que se usa `gdown` en vez de
un simple `curl`/`requests.get` (Google Drive bloquea descargas directas de
archivos grandes con una interstitial de "virus scan").

Uso:
    python -m app.pipeline.download_weights
"""

from pathlib import Path

# ID del archivo tal como aparece en el enlace de Google Drive del README
# de https://github.com/zllrunning/face-parsing.PyTorch
_GDRIVE_FILE_ID = "154JgKpzCPW82qINcVieuPH3fZ2e0P812"

WEIGHTS_DIR = Path(__file__).resolve().parent / "bisenet" / "weights"
WEIGHTS_PATH = WEIGHTS_DIR / "79999_iter.pth"


def download_weights() -> Path:
    if WEIGHTS_PATH.exists():
        print(f"Ya existen los pesos en {WEIGHTS_PATH}, no se vuelve a descargar.")
        return WEIGHTS_PATH

    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "Falta 'gdown'. Instálalo con `pip install gdown` "
            "(ya está listado en requirements.txt)."
        ) from exc

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={_GDRIVE_FILE_ID}"
    print(f"Descargando pesos de BiSeNet face-parsing a {WEIGHTS_PATH} ...")
    gdown.download(url, str(WEIGHTS_PATH), quiet=False)

    if not WEIGHTS_PATH.exists():
        raise RuntimeError(
            "La descarga automática desde Google Drive falló (es frecuente si "
            "el archivo supera el límite de cuota del día). Descárgalo a mano "
            "desde el enlace del README de "
            "https://github.com/zllrunning/face-parsing.PyTorch y colócalo en "
            f"{WEIGHTS_PATH}"
        )

    return WEIGHTS_PATH


if __name__ == "__main__":
    download_weights()
