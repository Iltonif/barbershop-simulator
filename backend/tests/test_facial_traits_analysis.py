"""Tests de las heurísticas geométricas puras de `facial_traits_analysis.py`
(EAR, forma de ceja, separación de ojos, perfil de nariz) usando puntos
sintéticos -- sin cargar ningún modelo pesado (igual que
`test_visagismo_rules.py`/`test_visagismo_ai_advisor.py`, que tampoco
cargan mediapipe/BiSeNet). La detección de orejas y gafas (que sí
necesitan `hair_segmentation.segment_face_parts`, y por tanto los pesos
de BiSeNet) no se cubre aquí a propósito, por el mismo motivo por el que
el resto del repo no testea directamente ese modelo."""

import unittest

import numpy as np

from app.pipeline.facial_traits_analysis import (
    _RIGHT_EYE,
    _eye_aspect_ratio,
    _eye_spacing,
    _eyebrow_type,
    _nose_profile_type,
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


class TestEyebrowType(unittest.TestCase):
    def test_flat_eyebrow_is_straight_low(self):
        pts = _make_landmarks({17: (0, 10), 19: (10, 10), 21: (20, 10)})
        self.assertEqual(_eyebrow_type(pts, [17, 18, 19, 20, 21]), "straight_low")

    def test_curved_eyebrow_is_arched(self):
        # Punto central bastante más arriba (y menor) que los extremos.
        pts = _make_landmarks({17: (0, 10), 19: (10, 2), 21: (20, 10)})
        self.assertEqual(_eyebrow_type(pts, [17, 18, 19, 20, 21]), "arched")


class TestEyeSpacing(unittest.TestCase):
    def test_close_set_eyes(self):
        pts = _make_landmarks({36: (0, 0), 39: (10, 0), 42: (11, 0), 45: (21, 0)})
        self.assertEqual(_eye_spacing(pts), "close_set")

    def test_wide_set_eyes(self):
        pts = _make_landmarks({36: (0, 0), 39: (10, 0), 42: (30, 0), 45: (40, 0)})
        self.assertEqual(_eye_spacing(pts), "wide_set")

    def test_proportional_eyes(self):
        pts = _make_landmarks({36: (0, 0), 39: (10, 0), 42: (20, 0), 45: (30, 0)})
        self.assertEqual(_eye_spacing(pts), "proportional")


class TestNoseProfileType(unittest.TestCase):
    def test_straight_profile(self):
        # Punta de nariz (33) justo sobre la línea entrecejo(27)-mentón(8);
        # el labio (51) marca "delante" hacia +x.
        pts = _make_landmarks({27: (0, 0), 8: (0, 100), 33: (0, 50), 51: (5, 70)})
        self.assertEqual(_nose_profile_type(pts), "straight")

    def test_convex_profile_faces_positive_x(self):
        # Perfil "mirando hacia +x" (el labio sobresale hacia +x) y la
        # nariz sobresale aún más en esa misma dirección -> convexo.
        pts = _make_landmarks({27: (0, 0), 8: (0, 100), 33: (20, 50), 51: (5, 70)})
        self.assertEqual(_nose_profile_type(pts), "convex_prominent_nose")

    def test_convex_profile_faces_negative_x(self):
        # Mismo caso pero en el perfil "espejo" (mirando hacia -x): debe
        # seguir dando convexo, no concave, pese a que el signo bruto de
        # la desviación de la nariz se ha invertido.
        pts = _make_landmarks({27: (0, 0), 8: (0, 100), 33: (-20, 50), 51: (-5, 70)})
        self.assertEqual(_nose_profile_type(pts), "convex_prominent_nose")

    def test_concave_profile(self):
        # La cara mira hacia +x (labio en +x) pero la nariz se queda
        # retrasada respecto a esa línea -> cóncavo.
        pts = _make_landmarks({27: (0, 0), 8: (0, 100), 33: (-20, 50), 51: (5, 70)})
        self.assertEqual(_nose_profile_type(pts), "concave")


if __name__ == "__main__":
    unittest.main()
