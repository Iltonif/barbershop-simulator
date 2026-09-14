"""
Etapa 2 del pipeline: segmentación de pelo.

Separa el pelo actual del cliente del resto de la imagen (cara, fondo,
ropa, orejas). La calidad de esta máscara condiciona directamente el
realismo final, porque de ella depende dónde se "pinta" el pelo nuevo y
qué partes de la foto original deben quedar intactas.

Implementación: BiSeNet (backbone ResNet-18) entrenado sobre
CelebAMask-HQ, vendorizado en `app/pipeline/bisenet/` desde
https://github.com/zllrunning/face-parsing.PyTorch (licencia MIT — permite
uso comercial, a diferencia de otras alternativas evaluadas; ver
CLAUDE.md). El modelo segmenta 19 clases faciales; aquí solo nos quedamos
con la clase "hair" (índice 17), pero el resto de clases (orejas, gafas,
cuello, ropa) quedan disponibles vía `segment_face_parts()` por si
`compositor.py` las necesita más adelante para resolver oclusiones.

Requiere los pesos pre-entrenados: ejecutar una vez
`python -m app.pipeline.download_weights` antes de usar este módulo.
"""

from functools import lru_cache

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from app.pipeline.bisenet.model import BiSeNet
from app.pipeline.download_weights import WEIGHTS_PATH

# Mapeo de clases tal como lo define este checkpoint (CelebAMask-HQ, 19
# clases + fondo). Ver prepropess_data.py del repo original.
FACE_PARSING_CLASSES = {
    0: "background",
    1: "skin",
    2: "l_brow",
    3: "r_brow",
    4: "l_eye",
    5: "r_eye",
    6: "eye_g",  # gafas
    7: "l_ear",
    8: "r_ear",
    9: "ear_r",  # pendiente
    10: "nose",
    11: "mouth",
    12: "u_lip",
    13: "l_lip",
    14: "neck",
    15: "neck_l",  # collar
    16: "cloth",
    17: "hair",
    18: "hat",
}
HAIR_CLASS_INDEX = 17

_INFERENCE_SIZE = 512  # el modelo se entrenó a esta resolución cuadrada

_to_tensor = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


@lru_cache(maxsize=1)
def _load_model() -> BiSeNet:
    """Carga perezosa y cacheada del modelo (pesado: solo se hace una vez
    por proceso).
    """
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"No se encontraron los pesos en {WEIGHTS_PATH}. Ejecuta primero:\n"
            "    python -m app.pipeline.download_weights"
        )

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"  # GPU de Apple Silicon via Metal Performance Shaders
    else:
        device = "cpu"
    net = BiSeNet(n_classes=19)
    net.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    net.to(device)
    net.eval()
    return net


def segment_face_parts(image_bgr: np.ndarray) -> np.ndarray:
    """Devuelve un mapa de segmentación (H, W) con valores 0-18, uno por
    cada clase de `FACE_PARSING_CLASSES`, ya reescalado al tamaño original
    de la imagen de entrada.
    """
    net = _load_model()
    device = next(net.parameters()).device

    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (_INFERENCE_SIZE, _INFERENCE_SIZE), interpolation=cv2.INTER_LINEAR)

    tensor = _to_tensor(resized).unsqueeze(0).to(device)

    with torch.no_grad():
        out = net(tensor)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0).astype(np.uint8)

    parsing_full_res = cv2.resize(parsing, (w, h), interpolation=cv2.INTER_NEAREST)
    return parsing_full_res


def segment_hair(image_bgr: np.ndarray) -> np.ndarray:
    """Devuelve una máscara binaria (mismo alto/ancho que la imagen) donde
    255 = píxel de pelo, 0 = el resto. Misma interfaz que la versión
    placeholder anterior, así que no hace falta tocar el resto del
    pipeline (routes.py, hair_type.py, compositor.py).
    """
    parsing = segment_face_parts(image_bgr)
    mask = np.where(parsing == HAIR_CLASS_INDEX, 255, 0).astype(np.uint8)

    # Limpieza morfológica ligera: BiSeNet ya da bordes razonablemente
    # limpios, así que aquí basta con quitar ruido de un par de píxeles.
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def mask_coverage_ratio(mask: np.ndarray) -> float:
    """Proporción de la imagen cubierta por la máscara (0.0 - 1.0).

    Útil para detectar segmentaciones claramente erróneas (p.ej. 0% o 90%
    de la imagen marcada como pelo) antes de seguir con el resto del
    pipeline.
    """
    return float(np.count_nonzero(mask)) / mask.size
