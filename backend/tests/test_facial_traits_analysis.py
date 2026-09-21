"""Tests de las heurísticas de `facial_traits_analysis.py` (EAR, simetría
ocular, giro de cabeza, separación de ojos, gafas) usando
puntos y mapas de segmentación sintéticos -- sin cargar ningún modelo
pesado (igual que `test_visagismo_rules.py`/`test_visagismo_ai_advisor.py`).
Donde hace falta la cara o la segmentación, se sustituyen
`face_analysis.analyze_face` y `hair_segmentation.segment_face_parts` por
dobles de prueba con `unittest.mock.patch`.

Los valores concretos de los umbrales vienen de la calibración con fotos
reales documentada en CLAUDE.md; varios tests fijan justo los casos que
motivaron cambiarlos (p.ej. un 18% de diferencia entre ojos, que antes se
marcaba como asimetría y en caras normales es solo ruido)."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.pipeline import facial_traits_analysis as fta
from app.pipeline.facial_traits_analysis import (
    _RIGHT_EYE,
    _eye_aspect_ratio,
    _eye_spacing,
    _frontal_turn,
    _has_glasses,
)


def _make_landmarks(overrides: dict[int, tuple[float, float]]) -> np.ndarray:
    """68 puntos base en una rejilla neutra y razonable, con los índices
    de `overrides` sustituidos por coordenadas concretas para el test."""
    points = np.zeros((68, 2), dtype=float)
    # Rejilla base: nada realista, solo evita ceros degenerados en los
    # puntos que un test concreto no vaya a sobrescribir.
    for i in range(68):
        points[i] = (100 + i, 100 + i)
    for idx, xy in overrides.items():
        points[idx] = xy
    return points


def _frontal_face(left_eye_half_height: float = 3.0, nose_x: float = 50.0) -> np.ndarray:
    """Cara frontal sintética. Ojo derecho (36-41) con EAR = 0,30 (párpados
    a ±3 del centro, 20 de ancho). El ojo izquierdo (42-47) usa
    `left_eye_half_height`: con 3,0 es idéntico al derecho, con menos
    está más cerrado. Centros de los ojos en x=30 y x=70 (distancia 40),
    así que la punta de la nariz (30) en x=50 = mirando de frente."""
    h = left_eye_half_height
    return _make_landmarks({
        0: (0, 60), 16: (100, 60),  # ancho de cara = 100
        36: (20, 50), 37: (25, 47), 38: (35, 47), 39: (40, 50), 40: (35, 53), 41: (25, 53),
        42: (60, 50), 43: (65, 50 - h), 44: (75, 50 - h), 45: (80, 50), 46: (75, 50 + h), 47: (65, 50 + h),
        30: (nose_x, 70),
    })


def _parsing(skin_px: int, glasses_px: int) -> np.ndarray:
    parsing = np.zeros((100, 100), dtype=np.uint8)
    flat = parsing.reshape(-1)
    flat[:skin_px] = fta._SKIN_CLASS
    flat[skin_px:skin_px + glasses_px] = fta._EYE_GLASSES_CLASS
    return parsing


class TestEyeAspectRatio(unittest.TestCase):
    def test_open_eye_has_higher_ear_than_closed_eye(self):
        # Ojo "abierto": párpados bien separados verticalmente.
        open_eye = _make_landmarks({
            36: (0, 10), 37: (2, 6), 38: (4, 6), 39: (6, 10), 40: (4, 14), 41: (2, 14),
        })
        # Mismo ancho horizontal, párpados casi juntos (ojo entrecerrado).
        closed_eye = _make_landmarks({
            36: (0, 10), 37: (2, 9.5), 38: (4, 9.5), 39: (6, 10), 40: (4, 10.5), 41: (2, 10.5),
        })
        ear_open = _eye_aspect_ratio(open_eye, _RIGHT_EYE)
        ear_closed = _eye_aspect_ratio(closed_eye, _RIGHT_EYE)
        self.assertGreater(ear_open, ear_closed)

    def test_degenerate_zero_width_eye_returns_zero(self):
        pts = _make_landmarks({36: (5, 5), 39: (5, 5), 37: (5, 4), 38: (5, 4), 40: (5, 6), 41: (5, 6)})
        self.assertEqual(_eye_aspect_ratio(pts, _RIGHT_EYE), 0.0)


class TestFrontalTurn(unittest.TestCase):
    def test_looking_straight_is_zero(self):
        self.assertAlmostEqual(_frontal_turn(_frontal_face(nose_x=50)), 0.0)

    def test_turned_head_scales_with_interocular_distance(self):
        # Nariz 8 px a un lado con 40 px entre ojos = 0,2.
        self.assertAlmostEqual(_frontal_turn(_frontal_face(nose_x=58)), 0.2)
        self.assertAlmostEqual(_frontal_turn(_frontal_face(nose_x=42)), 0.2)


class TestEyeSpacing(unittest.TestCase):
    """Intercantal (39-42) / ancho de cara (0-16)."""

    def _pts(self, inner_right_x, inner_left_x):
        return _make_landmarks({0: (0, 0), 16: (100, 0), 39: (inner_right_x, 0), 42: (inner_left_x, 0)})

    def test_close_set_eyes(self):
        self.assertEqual(_eye_spacing(self._pts(40, 60)), "close_set")  # 0,20

    def test_wide_set_eyes(self):
        self.assertEqual(_eye_spacing(self._pts(34, 66)), "wide_set")  # 0,32

    def test_typical_face_is_proportional(self):
        # Mediana medida en caras reales: ~0,25.
        self.assertEqual(_eye_spacing(self._pts(37.5, 62.5)), "proportional")


class TestHasGlasses(unittest.TestCase):
    def test_glasses_relative_to_skin(self):
        self.assertTrue(_has_glasses(_parsing(skin_px=1000, glasses_px=200)))  # 0,20

    def test_segmentation_noise_is_not_glasses(self):
        # 0,003 fue el máximo medido en fotos sin gafas.
        self.assertFalse(_has_glasses(_parsing(skin_px=1000, glasses_px=3)))

    def test_no_skin_detected(self):
        self.assertFalse(_has_glasses(_parsing(skin_px=0, glasses_px=50)))


class TestAnalyzeFrontal(unittest.TestCase):
    def _run(self, landmarks, parsing=None):
        parsing = _parsing(1000, 0) if parsing is None else parsing
        result = fta.FacialTraitsResult()
        with patch.object(fta.face_analysis, "analyze_face", return_value=SimpleNamespace(landmarks=landmarks)), \
             patch.object(fta.hair_segmentation, "segment_face_parts", return_value=parsing):
            fta._analyze_frontal(np.zeros((10, 10, 3), dtype=np.uint8), result)
        return result

    def test_symmetric_eyes(self):
        result = self._run(_frontal_face(left_eye_half_height=3.0))
        self.assertEqual(result.facial_features_profile["eye_symmetry"], "symmetric")
        self.assertIsNone(result.detected_anomalies_notes)

    def test_18_percent_is_noise_not_asymmetry(self):
        # Con el umbral anterior (15%) esto se marcaba como asimetría; en
        # caras normales se llega hasta ~15% solo por expresión y pose.
        result = self._run(_frontal_face(left_eye_half_height=3.0 * 0.82))
        profile = result.facial_features_profile
        self.assertEqual(profile["eye_symmetry"], "symmetric")
        self.assertAlmostEqual(profile["eye_symmetry_percent"], 18.0, places=0)

    def test_clear_asymmetry_is_reported_with_side(self):
        result = self._run(_frontal_face(left_eye_half_height=3.0 * 0.75))
        profile = result.facial_features_profile
        self.assertEqual(profile["eye_symmetry"], "asymmetric")
        self.assertAlmostEqual(profile["eye_symmetry_percent"], 25.0, places=0)
        self.assertIn("ojo derecho", result.detected_anomalies_notes)

    def test_turned_head_skips_symmetry_and_spacing_but_not_glasses(self):
        # Mismo ojo izquierdo "más cerrado" que el caso anterior, pero con
        # la cabeza girada: podría ser solo escorzo, así que no se mide.
        result = self._run(
            _frontal_face(left_eye_half_height=3.0 * 0.75, nose_x=58),
            parsing=_parsing(skin_px=1000, glasses_px=200),
        )
        profile = result.facial_features_profile
        self.assertNotIn("eye_symmetry", profile)
        self.assertNotIn("eye_spacing", profile)
        self.assertIsNone(result.detected_anomalies_notes)
        self.assertTrue(profile["has_glasses"])
        self.assertTrue(any("girada" in w for w in result.warnings))

    def test_eyebrow_type_is_never_auto_filled(self):
        result = self._run(_frontal_face())
        self.assertNotIn("eyebrow_type", result.facial_features_profile)


class TestMergeDetectedFeatures(unittest.TestCase):
    DETECTED = {"eye_spacing": "proportional", "eye_symmetry": "asymmetric",
                "eye_symmetry_percent": 24.0, "has_glasses": True}

    def test_fields_saved_as_none_by_the_form_are_filled(self):
        # Bug real: tras guardar la ficha, todos los campos existen con
        # None y el análisis no rellenaba nada.
        existing = {"profile_type": "straight", "eye_spacing": None, "eye_symmetry": None,
                    "eye_symmetry_percent": None, "has_glasses": None, "eyebrow_type": None}
        merged = fta.merge_detected_features(existing, self.DETECTED)
        self.assertEqual(merged["eye_spacing"], "proportional")
        self.assertEqual(merged["eye_symmetry"], "asymmetric")
        self.assertEqual(merged["eye_symmetry_percent"], 24.0)
        self.assertTrue(merged["has_glasses"])
        self.assertEqual(merged["profile_type"], "straight")
        self.assertIsNone(merged["eyebrow_type"])

    def test_manual_values_are_never_overwritten(self):
        existing = {"eye_spacing": "wide_set", "eye_symmetry": "symmetric", "eye_symmetry_percent": None}
        merged = fta.merge_detected_features(existing, self.DETECTED)
        self.assertEqual(merged["eye_spacing"], "wide_set")
        self.assertEqual(merged["eye_symmetry"], "symmetric")
        # No se añade un 24% medido que contradiga el "simétrico" manual.
        self.assertIsNone(merged["eye_symmetry_percent"])

    def test_unchecked_glasses_checkbox_does_not_block_detection(self):
        merged = fta.merge_detected_features({"has_glasses": False}, {"has_glasses": True})
        self.assertTrue(merged["has_glasses"])

    def test_checked_glasses_are_kept(self):
        merged = fta.merge_detected_features({"has_glasses": True}, {"has_glasses": False})
        self.assertTrue(merged["has_glasses"])

    def test_empty_profile(self):
        self.assertEqual(fta.merge_detected_features({}, self.DETECTED), self.DETECTED)


class TestAnalyzeFacialTraits(unittest.TestCase):
    def test_profile_photos_do_not_fill_nose_or_ears(self):
        # Probado con fotos reales: en foto de perfil no se detecta la cara
        # y BiSeNet confunde barbilla con oreja. Nariz, orejas y cejas son
        # campos manuales, aunque se manden las 3 fotos.
        frontal = SimpleNamespace(landmarks=_frontal_face())
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        with patch.object(fta.face_analysis, "analyze_face", return_value=frontal) as analyze, \
             patch.object(fta.hair_segmentation, "segment_face_parts", return_value=_parsing(1000, 0)):
            result = fta.analyze_facial_traits(image, image, image)
        profile = result.facial_features_profile
        for manual_field in ("profile_type", "ears_projection", "eyebrow_type"):
            self.assertNotIn(manual_field, profile)
        self.assertEqual(profile["eye_symmetry"], "symmetric")
        self.assertEqual(analyze.call_count, 1)  # solo la foto frontal

    def test_frontal_only_call_is_supported(self):
        with patch.object(fta.face_analysis, "analyze_face", return_value=None):
            result = fta.analyze_facial_traits(np.zeros((10, 10, 3), dtype=np.uint8))
        self.assertEqual(result.facial_features_profile, {})
        self.assertTrue(any("frontal" in w for w in result.warnings))


if __name__ == "__main__":
    unittest.main()
