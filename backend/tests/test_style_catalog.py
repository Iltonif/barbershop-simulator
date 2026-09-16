import json
import tempfile
import unittest
from pathlib import Path

from app.pipeline.style_catalog import HaircutStyle, load_catalog


class TestLoadCatalogSchemaDrift(unittest.TestCase):
    """Ver el incidente documentado en `STYLES_CATALOG_PATH`
    (`app/config.py`): un volumen persistente de Railway montado sobre
    `data/` tapó en producción una copia antigua de `styles.json` con una
    forma distinta a la que espera `HaircutStyle` hoy, y `load_catalog`
    reventaba con `TypeError` para TODO el catálogo por una sola entrada
    con una clave inesperada. Estos tests fijan el comportamiento
    correcto: ignorar claves desconocidas en vez de reventar."""

    def _cargar(self, entradas: list[dict]) -> list[HaircutStyle]:
        tmp = Path(tempfile.mktemp(suffix=".json"))
        tmp.write_text(json.dumps(entradas), encoding="utf-8")
        try:
            return load_catalog(path=tmp)
        finally:
            tmp.unlink(missing_ok=True)

    def test_clave_desconocida_no_revienta_la_carga(self):
        entradas = [
            {
                "id": "corte-viejo",
                "name": "Corte de catálogo antiguo",
                "description": "Entrada con una clave que ya no existe en el dataclass actual.",
                "length_top_mm": 30,
                "length_sides_mm": 5,
                "length_back_mm": 5,
                "fade_type": "bajo",
                "suitable_hair_types": ["liso"],
                "campo_que_ya_no_existe": "esto habria roto TypeError antes del fix",
            }
        ]
        catalogo = self._cargar(entradas)
        self.assertEqual(len(catalogo), 1)
        self.assertEqual(catalogo[0].id, "corte-viejo")
        # El campo desconocido se ignora, no aparece en el objeto.
        self.assertFalse(hasattr(catalogo[0], "campo_que_ya_no_existe"))

    def test_campos_opcionales_ausentes_usan_default(self):
        entradas = [
            {
                "id": "corte-minimo",
                "name": "Corte con solo los campos obligatorios",
                "description": "Sin length_category/source/style_family/reference_image.",
                "length_top_mm": 20,
                "length_sides_mm": 2,
                "length_back_mm": 2,
                "fade_type": "alto",
                "suitable_hair_types": ["rizado"],
            }
        ]
        catalogo = self._cargar(entradas)
        self.assertEqual(len(catalogo), 1)
        self.assertIsNone(catalogo[0].style_family)
        self.assertIsNone(catalogo[0].reference_image)

    def test_catalogo_real_sigue_cargando_sin_perder_entradas(self):
        # Regresión de que el propio catálogo real del proyecto (104
        # cortes en app/pipeline/catalog_data/styles.json) sigue cargando
        # bien tras mover su ubicación fuera de data/.
        catalogo = load_catalog()
        self.assertEqual(len(catalogo), 104)


if __name__ == "__main__":
    unittest.main()
