"""
Sesiones de la web: peluquero (PIN) y cliente (teléfono).

Hasta sept 2026 la API no tenía ningún control de acceso: cualquiera que
supiera el nombre exacto de un cliente podía ver y cambiar su ficha desde
Recomendaciones. Con el flujo nuevo (el cliente usa la tablet de la
barbería o su propio móvil con un QR mientras espera) eso ya no vale:

- **Peluquero**: entra con `BARBER_PIN` (config) y recibe la cookie
  `barber_session`. Caduca tras `BARBER_IDLE_MINUTES` sin usar la web: en
  la tablet compartida el siguiente cliente no debe encontrarse la parte
  del peluquero abierta. Todo `/api/clients/*` y la simulación libre lo
  exigen.
- **Cliente**: entra con su teléfono (decisión de Pedro: solo teléfono,
  sin PIN; quien sepa el teléfono de otro podría entrar en su ficha) y
  recibe la cookie `client_session`, que solo da acceso a SU ficha
  (`/api/me/*`).

Las cookies van firmadas con HMAC-SHA256 (sin dependencias nuevas):
`<rol>.<id>.<emitida_en>.<firma>`. HttpOnly y SameSite=Lax; Secure cuando
la web va por HTTPS (Railway pone `x-forwarded-proto`).
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time

from fastapi import HTTPException, Request, Response

from app import config

BARBER_COOKIE = "barber_session"
CLIENT_COOKIE = "client_session"

_secret_cache: bytes | None = None


def _secret() -> bytes:
    global _secret_cache
    if _secret_cache is None:
        if config.APP_SECRET:
            _secret_cache = config.APP_SECRET.encode()
        else:
            path = config.DB_PATH.parent / ".app_secret"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text(secrets.token_hex(32))
            _secret_cache = path.read_text().strip().encode()
    return _secret_cache


def _sign(payload: str) -> str:
    return hmac.new(_secret(), payload.encode(), hashlib.sha256).hexdigest()


def make_token(role: str, subject: str) -> str:
    payload = f"{role}.{subject}.{int(time.time())}"
    return f"{payload}.{_sign(payload)}"


def read_token(token: str | None, role: str, max_age_s: int) -> str | None:
    """Devuelve el sujeto (id de cliente, o "barber") si el token es válido,
    es de ese rol y no ha caducado."""
    if not token or token.count(".") != 3:
        return None
    tok_role, subject, issued, sig = token.split(".")
    if tok_role != role or not hmac.compare_digest(sig, _sign(f"{tok_role}.{subject}.{issued}")):
        return None
    try:
        if time.time() - int(issued) > max_age_s:
            return None
    except ValueError:
        return None
    return subject


def _secure(request: Request) -> bool:
    return request.headers.get("x-forwarded-proto", request.url.scheme) == "https"


def set_cookie(response: Response, request: Request, name: str, token: str, max_age_s: int) -> None:
    response.set_cookie(name, token, max_age=max_age_s, httponly=True, samesite="lax",
                        secure=_secure(request), path="/")


def barber_max_age() -> int:
    return config.BARBER_IDLE_MINUTES * 60


def client_max_age() -> int:
    return config.CLIENT_SESSION_DAYS * 24 * 3600


def is_barber(request: Request) -> bool:
    return read_token(request.cookies.get(BARBER_COOKIE), "barber", barber_max_age()) == "barber"


def require_barber(request: Request, response: Response) -> None:
    """Dependencia de FastAPI para las rutas del peluquero. Renueva la cookie
    en cada uso (caducidad por inactividad, no desde que entró)."""
    if not is_barber(request):
        raise HTTPException(status_code=401, detail="Entra como peluquero para ver esto.")
    # Las recargas automáticas (la sala de espera se refresca sola) no
    # cuentan como uso: si no, la tablet no se bloquearía nunca.
    if request.headers.get("x-background") != "1":
        set_cookie(response, request, BARBER_COOKIE, make_token("barber", "barber"), barber_max_age())


def current_client_id(request: Request) -> str | None:
    return read_token(request.cookies.get(CLIENT_COOKIE), "client", client_max_age())


def require_client(request: Request) -> str:
    client_id = current_client_id(request)
    if not client_id:
        raise HTTPException(status_code=401, detail="Entra con tu teléfono.")
    return client_id


def check_pin(pin: str) -> bool:
    if not config.BARBER_PIN:
        return False
    return hmac.compare_digest(str(pin).strip(), str(config.BARBER_PIN).strip())
