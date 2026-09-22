"""
Cada cuánto hay que retocar un corte para que se vea bien, y cuándo le
toca volver a un cliente.

Pedro quiere evitar que los clientes "aguanten" demasiado entre visitas
(pérdidas para la barbería). Se hace de forma transparente, sin engañar al
cliente:

- cada corte muestra "retoque cada X-Y semanas" (catálogo, recomendaciones);
- el recomendador, entre cortes que encajan igual de bien, pone antes los
  que piden retoque más frecuente (ver `recommender.py`). Cada cuánto dice
  el cliente que viene ya no influye en las recomendaciones (Pedro lo
  quitó); solo sirve para calcular su próximo corte;
- el cliente ve en su espacio cuándo le toca, y el peluquero tiene en la
  sala la lista de quién se ha pasado de fecha (para avisarle).

Intervalos de guías de barbería (Barber's Take, Salt Grooming, Rusty Blade,
consultadas en sept 2026): degradados cada 2-3 semanas; cortes cortos cada
2-4; media melena cada 3-6; pelo largo cada 6-8. Aquí se afina un poco:
un degradado bajo/medio (progresivo) aguanta algo más que uno alto o a
piel.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

# Si no se sabe qué corte lleva (visita sin corte registrado), se asume
# el intervalo de un corte corto sin degradado.
DEFAULT_WEEKS = (3, 5)

_LARGOS = {"largo", "extra_largo"}
_FAMILIAS_LARGAS = {"rastas_trenzas_locs", "recogido_mono_coleta", "melena_larga"}


def weeks_for(fade_type: str | None, length_top_mm: int, length_sides_mm: int,
              length_category: str | None = None, style_family: str | None = None) -> tuple[int, int]:
    """(mínimo, máximo) de semanas hasta el siguiente retoque."""
    if length_category in _LARGOS or style_family in _FAMILIAS_LARGAS or length_top_mm > 150:
        return (6, 8)
    if fade_type in ("skin", "alto"):
        return (2, 3)
    if fade_type in ("bajo", "medio"):
        return (3, 4)
    if length_top_mm <= 15 or (length_category == "corto" or (length_top_mm <= 60 and length_sides_mm <= 25)):
        return (3, 4)
    return (4, 6)


def weeks_for_style(style) -> tuple[int, int]:
    return weeks_for(style.fade_type, style.length_top_mm, style.length_sides_mm,
                     style.length_category, style.style_family)


def label(weeks: tuple[int, int]) -> str:
    return f"Retoque cada {weeks[0]}-{weeks[1]} semanas"


def _parse(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def next_visit(last_cut_at: str | None, weeks: tuple[int, int] | None,
               client_frequency_days: int | None = None, now: datetime | None = None) -> dict | None:
    """Cuándo le toca volver: al final del intervalo del corte que lleva
    (o antes, si él mismo dijo que viene más a menudo).

    Devuelve {"due": iso, "days_left": int, "weeks": [min, max]} o None si
    no hay ninguna visita registrada."""
    if not last_cut_at:
        return None
    weeks = weeks or DEFAULT_WEEKS
    days = weeks[1] * 7
    if client_frequency_days:
        days = min(days, client_frequency_days)
    due = _parse(last_cut_at) + timedelta(days=days)
    now = now or datetime.now(timezone.utc)
    return {"due": due.isoformat(), "days_left": (due.date() - now.date()).days, "weeks": list(weeks)}
