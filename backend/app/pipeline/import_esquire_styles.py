"""
Importa al catálogo de cortes el ranking "100 peinados de hombre modernos
y actuales para 2023" de Esquire España
(https://www.esquire.com/es/cuidados-hombre/g36528457/peinados-hombre-modernos-actuales/),
a petición del usuario ("guarda todos los cortes de pelo que aparecen en
esta web, clasifícalos según longitud y tipo de cabello").

De dónde sale cada campo:
- `name`/`description`: la frase descriptiva original de cada puesto del
  ranking (texto editorial de Esquire, pegado directamente por el usuario
  en la conversación — la página no se pudo scrapear, está bloqueada para
  las herramientas de navegación web).
- `length_category` y `suitable_hair_types`: clasificación hecha a mano,
  leyendo cada frase, según las palabras clave de longitud (corto/media
  melena/melena larga/extra larga) y de tipo de cabello (liso/ondas-
  ondulado/rizado/muy rizado-afro/rastas) que menciona. Cuando la frase NO
  menciona el tipo de cabello (la mayoría de los 100 no lo hace), se
  asigna el conjunto de tipos para los que ese estilo de corte es
  razonable en la práctica, no un cabello concreto adivinado — así que
  estos son una aproximación de partida, pensada para corregirse desde la
  UI cuando un barbero real no esté de acuerdo con alguno.
- `length_top_mm`/`length_sides_mm`/`length_back_mm`/`fade_type`: NO están
  en el artículo (que no da medidas). Son una estimación orientativa
  derivada solo de `length_category` + si la frase menciona explícitamente
  "rapado"/"laterales rapados"/degradado (⇒ fade_type), usando las mismas
  convenciones de medidas que ya usaba `backend/data/styles/styles.json`.
  No se han medido/verificado sobre ninguna foto real.
- `reference_image`: siempre None. El artículo no da URLs de imagen
  descargables (solo créditos de foto tipo "© Zara" / "Getty Images", que
  no son imágenes que se puedan usar), y este proyecto ya decidió antes
  (ver CLAUDE.md, caso Figaro-1k) no incorporar fotos de personas
  identificables sin poder confirmar la licencia — así que aquí solo se
  guarda el texto/clasificación, ninguna imagen.

Idempotente: si ya existen ids con el prefijo "esq2023-" en el catálogo,
no se vuelven a añadir (evita duplicar si se ejecuta dos veces).
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app.config import STYLES_CATALOG_PATH  # noqa: E402

SOURCE = (
    'Esquire España, "100 peinados de hombre modernos y actuales para 2023" '
    "(https://www.esquire.com/es/cuidados-hombre/g36528457/peinados-hombre-modernos-actuales/), "
    "puesto #{num} del ranking"
)

# (num_ranking, descripcion, longitud, tipos_de_pelo, fade_type)
RAW = [
    (1, "Rastas en coleta alta", "largo", ["rizado", "afro"], "ninguno"),
    (2, "Mullet clásico de los 80 con pelo rizado", "medio", ["rizado"], "bajo"),
    (3, "Peinado con mechas con textura", "medio", ["liso", "ondulado", "rizado"], "ninguno"),
    (4, "Corte tazón con la parte trasera larga", "medio", ["liso", "ondulado"], "ninguno"),
    (5, "Melena larga con mechas delanteras escalonadas", "largo", ["liso", "ondulado"], "ninguno"),
    (6, "Melena corta pegada en la parte superior de la cabeza y con raya en medio", "corto", ["liso", "ondulado"], "ninguno"),
    (7, "Corte mullet con flequillo tazón y patilla larga", "medio", ["liso", "ondulado", "rizado"], "bajo"),
    (8, "Rastas peinadas en moño alto", "largo", ["rizado", "afro"], "ninguno"),
    (9, "Corte tazón con pelo rizado", "corto", ["rizado"], "ninguno"),
    (10, "Melena corta peinada hacia detrás", "corto", ["liso", "ondulado"], "ninguno"),
    (11, "Peinado hacia delante con patilla corta", "corto", ["liso", "ondulado", "rizado"], "ninguno"),
    (12, "Flequillo corto con textura y laterales rapados", "corto", ["liso", "ondulado", "rizado"], "alto"),
    (13, "Corte de pelo semilargo con rastas", "medio", ["rizado", "afro"], "ninguno"),
    (14, "Melena corta lisa con raya en medio", "corto", ["liso"], "ninguno"),
    (15, "Melena degradada con raya en medio", "medio", ["liso", "ondulado"], "ninguno"),
    (16, "Corte de pelo tazón con textura", "corto", ["liso", "ondulado", "rizado"], "ninguno"),
    (17, "Flequillo corto casi a mitad de frente", "corto", ["liso", "ondulado"], "ninguno"),
    (18, "Corte degradado con pelo rizado", "corto", ["rizado"], "alto"),
    (19, "Peinado hacia atrás generando algo de volumen en la zona superior", "medio", ["liso", "ondulado"], "ninguno"),
    (20, "Pelo engominado con raya a un lado", "corto", ["liso"], "ninguno"),
    (21, "Melena peinada con efecto mojado", "largo", ["liso", "ondulado"], "ninguno"),
    (22, "Flequillo corto cortado con las puntas desfiladas", "corto", ["liso", "ondulado"], "ninguno"),
    (23, "Flequillo semilargo peinado hacia delante", "medio", ["liso", "ondulado"], "ninguno"),
    (24, "Tupé mini con raya al lado", "corto", ["liso", "ondulado"], "bajo"),
    (25, 'Peinado de efecto "recién levantado"', "corto", ["liso", "ondulado", "rizado"], "ninguno"),
    (26, "Tupé extra largo de lado", "medio", ["liso", "ondulado"], "medio"),
    (27, "Peinado con textura efecto nido", "corto", ["liso", "ondulado", "rizado"], "ninguno"),
    (28, "Corte desfilado que enmarca el rostro", "medio", ["liso", "ondulado"], "ninguno"),
    (29, "Flequillo largo rizado y corto en laterales y parte de atrás", "medio", ["rizado"], "alto"),
    (30, "Rastas con pelo corto y coleta", "corto", ["rizado", "afro"], "ninguno"),
    (31, "Peinado con gomina hacia detrás manteniendo cierto volumen", "corto", ["liso", "ondulado"], "ninguno"),
    (32, "Pelo engominado con moño bajo deshecho", "largo", ["liso", "ondulado"], "ninguno"),
    (33, "Melena extra larga con raya en medio y puntas decoloradas", "extra_largo", ["liso", "ondulado"], "ninguno"),
    (34, "Tupé retro", "corto", ["liso", "ondulado"], "bajo"),
    (35, "Melena corta con raya en medio", "corto", ["liso", "ondulado"], "ninguno"),
    (36, "Con raya en diagonal y rubio platino", "medio", ["liso"], "ninguno"),
    (37, "Pelo corto peinado hacia delante con textura", "corto", ["liso", "ondulado", "rizado"], "ninguno"),
    (38, "Pelo corto con zona superior más larga de efecto trasquilón", "corto", ["liso", "ondulado", "rizado"], "bajo"),
    (39, "Pelo corto peinado con efecto mojado", "corto", ["liso", "ondulado"], "ninguno"),
    (40, 'Pelo muy rizado de peinado casi "afro"', "medio", ["rizado", "afro"], "ninguno"),
    (41, "Melena corta peinada hacia atrás de efecto despeinado", "corto", ["liso", "ondulado"], "ninguno"),
    (42, "Pelo corto peinado hacia atrás con volumen", "corto", ["liso", "ondulado"], "ninguno"),
    (43, "Tupé con mechón de inspiración rockabilly", "corto", ["liso", "ondulado"], "bajo"),
    (44, "Melena larga peinada hacia atrás con algo de volumen", "largo", ["liso", "ondulado"], "ninguno"),
    (45, "Melena media desfilada peinada hacia delante con raya en medio", "medio", ["liso", "ondulado"], "ninguno"),
    (46, "Melena corta con flequillo recto", "corto", ["liso"], "ninguno"),
    (47, "Flequillo XL peinado a un lado por encima del ojo y laterales y parte de atrás corta", "medio", ["liso", "ondulado"], "bajo"),
    (48, "Flequillo extra largo peinado hacia delante", "medio", ["liso", "ondulado"], "ninguno"),
    (49, "Flequillo largo hacia delante y un lado", "medio", ["liso", "ondulado"], "ninguno"),
    (50, "Flequillo largo en forma de uve", "medio", ["liso", "ondulado"], "ninguno"),
    (51, "Flequillo corto abierto en medio", "corto", ["liso", "ondulado"], "ninguno"),
    (52, "Corte de pelo estilo querubín", "corto", ["ondulado", "rizado"], "ninguno"),
    (53, "Flequillo muy largo abierto, con mechas desiguales", "medio", ["liso", "ondulado"], "ninguno"),
    (54, "Corte de pelo tazón con pelo rizado", "corto", ["rizado"], "ninguno"),
    (55, "Recogido mitad moño, mitad coleta", "largo", ["liso", "ondulado", "rizado"], "ninguno"),
    (56, "Peinado de efecto despeinado con flequillo largo y liso", "medio", ["liso"], "ninguno"),
    (57, "Corte de pelo degradado con volumen en la parte superior", "corto", ["liso", "ondulado", "rizado"], "alto"),
    (58, "Corte de pelo cuadrado con flequillo muy corto", "corto", ["liso", "ondulado"], "ninguno"),
    (59, "Corte de pelo asimétrico con flequillo abierto en medio", "medio", ["liso", "ondulado"], "ninguno"),
    (60, "Mullet con pelo liso", "medio", ["liso"], "bajo"),
    (61, "Híbrido de corte tazón y mullet con flequillo tupido", "medio", ["liso", "ondulado"], "ninguno"),
    (62, "Semirrecogido con moño", "largo", ["liso", "ondulado", "rizado"], "ninguno"),
    (63, "Flequillo tazón muy largo desfilado", "medio", ["liso", "ondulado"], "ninguno"),
    (64, "Tupé cuadrado", "corto", ["liso", "ondulado"], "bajo"),
    (65, "Pelo rizado hacia atrás de longitud media", "medio", ["rizado"], "ninguno"),
    (66, "Pelo corto sin flequillo de forma cuadrada y laterales rapados", "corto", ["liso", "ondulado", "rizado"], "alto"),
    (67, "Flequillo largo con mechones de efecto despeinado", "medio", ["liso", "ondulado"], "ninguno"),
    (68, "Peinado con raya al lado y medio tupé", "corto", ["liso", "ondulado"], "bajo"),
    (69, "Flequillo cuadrado a mitad de frente", "medio", ["liso", "ondulado"], "ninguno"),
    (70, "Flequillo largo que nace de la raíz superior de la cabeza", "medio", ["liso", "ondulado"], "ninguno"),
    (71, "Tupé hacia atrás y un lado", "corto", ["liso", "ondulado"], "bajo"),
    (72, "Rastas largas en moño alto", "largo", ["rizado", "afro"], "ninguno"),
    (73, "Tupé al estilo Tintín", "corto", ["liso"], "bajo"),
    (74, "Corte mullet con volumen en la parte superior", "medio", ["liso", "ondulado", "rizado"], "bajo"),
    (75, "Melena corta con raya en medio y zona baja rizada", "corto", ["ondulado", "rizado"], "ninguno"),
    (76, "Rapado con acabado en pico", "corto", ["liso", "ondulado", "rizado", "afro"], "skin"),
    (77, "Melena corta lisa con raya en medio y puntas delanteras un poco más largas", "corto", ["liso"], "ninguno"),
    (78, "Corte mitad mullet, mitad tazón", "medio", ["liso", "ondulado"], "ninguno"),
    (79, "Corte tazón con pelo rizado y efecto despeinado", "corto", ["rizado"], "ninguno"),
    (80, "Rapado por toda la cabeza, excepto la superior que va peinada con trenzas y coleta alta", "largo", ["rizado", "afro"], "skin"),
    (81, "Pelo rizado con flequillo largo y patillas", "medio", ["rizado"], "ninguno"),
    (82, "Peinado con volumen de efecto despeinado", "medio", ["liso", "ondulado", "rizado"], "ninguno"),
    (83, "Moño alto con pelo rizado", "largo", ["rizado"], "ninguno"),
    (84, "Flequillo desigual, imitando el efecto trasquilón", "corto", ["liso", "ondulado"], "ninguno"),
    (85, "Peinado hacia atrás con textura", "corto", ["liso", "ondulado", "rizado"], "ninguno"),
    (86, "Semirrecogido con moño", "largo", ["liso", "ondulado", "rizado"], "ninguno"),
    (87, "Peinado hacia atrás sin raya", "corto", ["liso", "ondulado"], "ninguno"),
    (88, "Flequillo extra largo, abierto con raya medio de lado", "medio", ["liso", "ondulado"], "ninguno"),
    (89, "Tupé ladeado", "corto", ["liso", "ondulado"], "bajo"),
    (90, "Pelo corto peinado hacia atrás con mechones con textura", "corto", ["liso", "ondulado", "rizado"], "ninguno"),
    (91, "Pelo corto híper rizado al natural", "corto", ["rizado", "afro"], "ninguno"),
    (92, "Flequillo largo, peinado con raya al lado y volumen", "medio", ["liso", "ondulado"], "ninguno"),
    (93, "Corte estilo mohicano, con rastas largas en la parte inferior", "largo", ["rizado", "afro"], "skin"),
    (94, "Media melena con raya en medio", "medio", ["liso", "ondulado"], "ninguno"),
    (95, "Flequillo extralargo con caída hacia atrás y un lado", "medio", ["liso", "ondulado"], "ninguno"),
    (96, "Pelo largo con laterales rapados peinado con gomina y medio moño bajo", "largo", ["liso", "ondulado"], "alto"),
    (97, "Peinado con tupé clásico y patillas anchas", "corto", ["liso", "ondulado"], "bajo"),
    (98, "Peinado de efecto recién levantado", "corto", ["liso", "ondulado", "rizado"], "ninguno"),
    (99, "Pelo desfilado con patillas largas y finas", "medio", ["liso", "ondulado"], "ninguno"),
    (100, "Melena con ondas estilo surfero", "largo", ["ondulado"], "ninguno"),
]

# mm orientativos por (longitud, fade) — ver docstring: no vienen del
# artículo, es una estimación a partir de la categoría + si se menciona
# rapado/degradado explícitamente.
TOP_MM = {"corto": 50, "medio": 95, "largo": 180, "extra_largo": 280}
SIDES_BACK_MM = {
    ("corto", "ninguno"): 20, ("corto", "bajo"): 15, ("corto", "medio"): 10,
    ("corto", "alto"): 4, ("corto", "skin"): 1,
    ("medio", "ninguno"): 70, ("medio", "bajo"): 45, ("medio", "medio"): 25,
    ("medio", "alto"): 10, ("medio", "skin"): 2,
    ("largo", "ninguno"): 160, ("largo", "bajo"): 100, ("largo", "alto"): 8,
    ("largo", "skin"): 1,
    ("extra_largo", "ninguno"): 260,
}


def slugify(text: str) -> str:
    text = text.lower()
    text = (text.replace("á", "a").replace("é", "e").replace("í", "i")
                .replace("ó", "o").replace("ú", "u").replace("ñ", "n"))
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:60].strip("-")


def build_entries():
    entries = []
    for num, desc, longitud, tipos, fade in RAW:
        slug = slugify(desc)
        sides_back = SIDES_BACK_MM[(longitud, fade)]
        entries.append({
            "id": f"esq2023-{num:03d}-{slug}",
            "name": desc,
            "description": desc,
            "length_top_mm": TOP_MM[longitud],
            "length_sides_mm": sides_back,
            "length_back_mm": sides_back,
            "fade_type": fade,
            "suitable_hair_types": tipos,
            "reference_image": None,
            "length_category": longitud,
            "source": SOURCE.format(num=num),
        })
    return entries


def main():
    with open(STYLES_CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    already = any(item["id"].startswith("esq2023-") for item in catalog)
    if already:
        print("Ya existen estilos 'esq2023-' en el catálogo, no se vuelve a importar nada.")
        return

    new_entries = build_entries()
    catalog.extend(new_entries)

    with open(STYLES_CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Añadidos {len(new_entries)} cortes nuevos. Catálogo total: {len(catalog)} estilos.")


if __name__ == "__main__":
    main()
