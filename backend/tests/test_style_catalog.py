import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import config
from app.db import database
from app.pipeline.style_catalog import (
    HaircutStyle,
    get_style_by_id_anywhere,
    load_catalog,
    load_custom_styles,
    load_full_catalog,
)


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


class TestCustomStyles(unittest.TestCase):
    """Cortes que el peluquero añade desde el selector (ver POST /api/styles,
    sept 2026), guardados en la base de datos y combinados con el catálogo
    base por `load_full_catalog`."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.patches = [
            patch.object(database, "DB_PATH", tmp / "clients.db"),
            patch.object(config, "DB_PATH", tmp / "clients.db"),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_sin_base_de_datos_iniciada_no_revienta(self):
        # Una base de datos recién creada (sqlite3.connect la crea sola)
        # pero sin `init_db()` no tiene la tabla `custom_styles` todavía --
        # p.ej. tests que llaman a `recommend_styles`/`load_full_catalog`
        # de forma aislada sin arrancar la app entera (ver test_growth.py).
        self.assertEqual(load_custom_styles(), [])
        self.assertEqual(len(load_full_catalog()), 104)

    def test_corte_propio_se_combina_con_el_catalogo_base(self):
        database.init_db()
        from app.db import repository
        style_id = repository.create_custom_style(
            "Corte de prueba", "Descripción de prueba", 40, 10, 10, "bajo", ["liso", "afro"],
        )
        self.assertTrue(style_id.startswith("personalizado-"))

        propios = load_custom_styles()
        self.assertEqual(len(propios), 1)
        self.assertEqual(propios[0].name, "Corte de prueba")
        self.assertEqual(propios[0].suitable_hair_types, ["liso", "afro"])
        # Campos que este corte no tiene (no viene del catálogo base):
        # opcionales, no un dato inventado.
        self.assertIsNone(propios[0].style_family)
        self.assertIsNone(propios[0].reference_image)

        completo = load_full_catalog()
        self.assertEqual(len(completo), 105)  # 104 del catálogo base + 1 propio
        self.assertIn(style_id, {s.id for s in completo})
        self.assertEqual(get_style_by_id_anywhere(style_id).name, "Corte de prueba")
        # El catálogo base a secas no se entera -- lo usan los tests/scripts
        # que necesitan aislarse de la base de datos.
        self.assertEqual(len(load_catalog()), 104)


if __name__ == "__main__":
    unittest.main()
