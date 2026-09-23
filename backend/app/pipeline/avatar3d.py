"""
Gemelo digital 3D del cliente: fotos -> modelo .glb (Tripo AI).

Pedro: "que el sistema extraiga resultados de visajismo y de pelo a través
de la creación de un avatar de tu cara" (anuncio de ILTONIF: foto -> modelo
3D -> "tu gemelo, en 360°" -> medidas -> corte recomendado).

Qué hace este módulo: manda a Tripo AI las tres fotos guiadas que ya hace
`frontend/visagismo.html` (frontal + los dos perfiles), espera a que
termine y devuelve el `.glb`. Las medidas sobre esa malla las calcula
`mesh_metrics.py`; aquí solo está la llamada al proveedor.

Límite importante y honesto: Tripo es un modelo GENERATIVO, no un escáner.
Reconstruye una cabeza plausible a partir de las fotos, así que la nuca, las
orejas y el pelo son en parte inventados y dos fotos distintas de la misma
persona no dan exactamente la misma malla. Sirve muy bien para verlo en 3D;
las medidas que se saquen de ahí son estimaciones, no medidas de la persona
(ver el aviso en `mesh_metrics.measure`).

RGPD (ver la sección del CLAUDE.md): la foto sale del servidor hacia un
tercero nuevo (Tripo) y, a diferencia de la simulación, el resultado -- una
reconstrucción 3D de la cara de una persona identificable -- SÍ se guarda.
Por eso hay un consentimiento propio (`consent_3d_scan`), el modelo se
borra al retirarlo o al borrar las fotos, y las fotos que se envían no se
guardan en ningún momento.

API (https://docs.tripo3d.ai): `POST {base}/upload/sts` para cada foto,
`POST {base}/task` con `type: multiview_to_model` y las vistas en el orden
[frontal, izquierda, atrás, derecha] (la de atrás va vacía: no tenemos foto
de la nuca), y `GET {base}/task/{id}` cada pocos segundos hasta que el
estado sea `success`. No se ha podido probar contra la API real desde el
entorno de desarrollo (sin salida a esa red ni clave), así que la primera
ejecución con clave real hay que mirarla: si el formato de subida o de
respuesta no coincide, lo que hay que tocar es `_upload_photo`/`_output_url`.
"""

from __future__ import annotations

import json
import mimetypes
import time
import urllib.error
import urllib.request
import uuid

from app import config

# Vistas que manda la web, en el orden que espera Tripo.
VIEWS = ("frontal", "perfil_izquierdo", "atras", "perfil_derecho")


class Avatar3DNotConfigured(RuntimeError):
    """No hay TRIPO_API_KEY configurada."""


class Avatar3DError(RuntimeError):
    """Tripo ha fallado, ha tardado demasiado o no ha devuelto modelo."""


def available() -> bool:
    return bool(config.TRIPO_API_KEY)


def _request(method: str, path: str, *, body: bytes | None = None, headers: dict | None = None,
             timeout: int = 60) -> dict:
    url = f"{config.TRIPO_BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Authorization", f"Bearer {config.TRIPO_API_KEY}")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # pragma: no cover - depende de la red
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise Avatar3DError(f"Tripo respondió {exc.code}: {detail}") from exc
    except Exception as exc:  # pragma: no cover - depende de la red
        raise Avatar3DError(f"No se pudo hablar con Tripo: {exc}") from exc
    if payload.get("code") not in (0, None):
        raise Avatar3DError(f"Tripo devolvió el error {payload.get('code')}: {payload.get('message')}")
    return payload.get("data") or {}


def _multipart(name: str, content: bytes) -> tuple[bytes, str]:
    boundary = f"----iltonif{uuid.uuid4().hex}"
    ctype = mimetypes.guess_type(name)[0] or "image/jpeg"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
        f"Content-Type: {ctype}\r\n\r\n"
    ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def _upload_photo(name: str, content: bytes) -> dict:
    """Sube una foto y devuelve cómo referirse a ella en la tarea.

    Tripo admite tres formas (`object` de la subida STS, `file_token` de la
    subida directa, o una `url` pública). Se usa la subida y se acepta la
    respuesta que venga: la de STS trae bucket/key, la directa un token."""
    body, ctype = _multipart(name, content)
    data = _request("POST", "/upload/sts", body=body, headers={"Content-Type": ctype})
    if data.get("bucket") and data.get("key"):
        return {"type": "image", "object": {"bucket": data["bucket"], "key": data["key"]}}
    token = data.get("image_token") or data.get("file_token") or data.get("token")
    if not token:
        raise Avatar3DError("Tripo no devolvió ninguna referencia de la foto subida")
    return {"type": "image", "file_token": token}


def _output_url(output: dict) -> str | None:
    """URL del .glb dentro de `data.output` (el nombre del campo ha ido
    cambiando entre versiones de la API)."""
    for key in ("pbr_model", "model", "base_model"):
        value = output.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict) and value.get("url"):
            return value["url"]
    return None


def _download(url: str, timeout: int = 120) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read()
    except Exception as exc:  # pragma: no cover - depende de la red
        raise Avatar3DError(f"No se pudo descargar el modelo: {exc}") from exc


def generate_twin(photos: dict[str, bytes], *, poll_seconds: float = 3.0) -> bytes:
    """Fotos guiadas -> bytes del .glb. `photos`: {vista: contenido}, con
    "frontal" obligatoria (Tripo no acepta la tarea sin ella) y al menos
    una más (con una sola foto se inventa casi toda la cabeza)."""
    if not available():
        raise Avatar3DNotConfigured("Falta TRIPO_API_KEY")
    if not photos.get("frontal"):
        raise Avatar3DError("Hace falta la foto frontal")
    if sum(1 for v in VIEWS if photos.get(v)) < 2:
        raise Avatar3DError("Hacen falta al menos dos fotos (frontal y un perfil)")

    files = []
    for view in VIEWS:
        content = photos.get(view)
        files.append(_upload_photo(f"{view}.jpg", content) if content else {})

    data = _request("POST", "/task", body=json.dumps({
        "type": "multiview_to_model",
        "model_version": config.TRIPO_MODEL_VERSION,
        "files": files,
        "texture": True,
        "pbr": True,
    }).encode(), headers={"Content-Type": "application/json"})
    task_id = data.get("task_id")
    if not task_id:
        raise Avatar3DError("Tripo no devolvió ninguna tarea")

    deadline = time.monotonic() + config.TRIPO_TIMEOUT_S
    while time.monotonic() < deadline:
        time.sleep(poll_seconds)
        task = _request("GET", f"/task/{task_id}")
        status = task.get("status")
        if status == "success":
            url = _output_url(task.get("output") or {})
            if not url:
                raise Avatar3DError("La tarea terminó sin modelo")
            return _download(url)
        if status in ("failed", "cancelled", "banned", "expired", "unknown"):
            raise Avatar3DError(f"Tripo no pudo generar el modelo (estado: {status})")
    raise Avatar3DError("Tripo está tardando demasiado; inténtalo de nuevo en un rato")
