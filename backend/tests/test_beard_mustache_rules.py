"""Reglas de recomendación de barba y bigote (`beard_mustache_rules.py`),
que sustituyeron en sept 2026 a la antigua `trait_rules.beard_advice`."""

import unittest

from app.pipeline import beard_mustache_rules as bmr


def profile(features=None, forehead=None, hairline=None, lifestyle=None):
    anat = {"facial_features_profile": features or {}}
    if forehead is not None:
        anat["facial_horizontal_zones_ratio"] = {"intellectual_zone_forehead": forehead}
    out = {"anatomical_metrics": anat}
    if hairline is not None:
        out["hair_physical_metrics"] = {"frontal_hairline_shape": hairline}
    if lifestyle is not None:
        out["lifestyle_and_preferences"] = lifestyle
    return out


def labels(advice):
    return [a["label"] for a in advice]


class TestPorFormaDeRostro(unittest.TestCase):
    def test_no_face_shape_no_advice(self):
        self.assertEqual(bmr.advice(None, None), [])
        self.assertEqual(bmr.advice({}, None), [])

    def test_las_seis_formas_cubiertas_tienen_regla(self):
        for shape in ("ovalada", "redonda", "alargada", "diamante", "triangular", "triangular_invertida"):
            adv = bmr.advice(None, shape)
            self.assertEqual(len(adv), 1, shape)
            self.assertTrue(adv[0]["label"])
            self.assertTrue(adv[0]["detail"])

    def test_cuadrada_se_deja_sin_regla_a_proposito(self):
        # Ninguna fuente proporcionada cubre "cuadrada" -- hueco honesto,
        # no una regla inventada.
        self.assertEqual(bmr.advice(None, "cuadrada"), [])

    def test_unknown_shape_no_advice(self):
        self.assertEqual(bmr.advice(None, "no_existe"), [])

    def test_shaved_prefix_applies_to_face_shape_entry(self):
        p = profile(lifestyle={"beard_preference": "clean_shaven"})
        adv = bmr.advice(p, "redonda")
        self.assertTrue(adv[0]["detail"].startswith("Si quiere probar barba y bigote"))


class TestMentonYMandibula(unittest.TestCase):
    def test_retruded_chin(self):
        p = profile({"chin_projection": "retruded"})
        adv = bmr.advice(p, None)
        self.assertEqual(labels(adv), ["Barba para el mentón retraído"])

    def test_prominent_chin(self):
        p = profile({"chin_projection": "prominent"})
        adv = bmr.advice(p, None)
        self.assertEqual(labels(adv), ["Barba y bigote con mentón/mandíbula prominente"])

    def test_balanced_chin_no_advice(self):
        p = profile({"chin_projection": "balanced"})
        self.assertEqual(bmr.advice(p, None), [])

    def test_soft_jawline(self):
        p = profile({"jawline_definition": "soft"})
        adv = bmr.advice(p, None)
        self.assertEqual(labels(adv), ["Barba para marcar la mandíbula"])

    def test_retruded_chin_and_soft_jaw_both_present_not_duplicated(self):
        p = profile({"chin_projection": "retruded", "jawline_definition": "soft"})
        adv = bmr.advice(p, None)
        self.assertEqual(labels(adv), ["Barba para el mentón retraído", "Barba para marcar la mandíbula"])


class TestPapadaNarizLabios(unittest.TestCase):
    def test_double_chin(self):
        p = profile({"has_double_chin": True})
        self.assertEqual(labels(bmr.advice(p, None)), ["Barba con papada"])

    def test_no_double_chin_no_advice(self):
        p = profile({"has_double_chin": False})
        self.assertEqual(bmr.advice(p, None), [])

    def test_nose_large(self):
        p = profile({"nose_size": "large"})
        self.assertEqual(labels(bmr.advice(p, None)), ["Bigote y nariz grande"])

    def test_nose_small(self):
        p = profile({"nose_size": "small"})
        self.assertEqual(labels(bmr.advice(p, None)), ["Barba y bigote con nariz pequeña"])

    def test_nose_proportional_no_advice(self):
        p = profile({"nose_size": "proportional"})
        self.assertEqual(bmr.advice(p, None), [])

    def test_lips_prominent(self):
        p = profile({"lip_thickness": "prominent"})
        self.assertEqual(labels(bmr.advice(p, None)), ["Bigote y labios prominentes"])

    def test_lips_thin(self):
        p = profile({"lip_thickness": "thin"})
        self.assertEqual(labels(bmr.advice(p, None)), ["Bigote y labios finos"])


class TestFrenteYEntradas(unittest.TestCase):
    def test_narrow_forehead(self):
        p = profile(forehead="narrow")
        self.assertEqual(labels(bmr.advice(p, None)), ["Barba y frente pequeña"])

    def test_prominent_forehead(self):
        p = profile(forehead="prominent")
        self.assertEqual(labels(bmr.advice(p, None)), ["Barba y frente ancha"])

    def test_proportional_forehead_no_advice(self):
        p = profile(forehead="proportional")
        self.assertEqual(bmr.advice(p, None), [])

    def test_receding_hairline(self):
        p = profile(hairline="m_shaped_receding")
        self.assertEqual(labels(bmr.advice(p, None)), ["Barba con entradas"])

    def test_other_hairline_shapes_no_advice(self):
        for shape in ("linear_straight", "widows_peak", "high_forehead"):
            p = profile(hairline=shape)
            self.assertEqual(bmr.advice(p, None), [], shape)


class TestCombinacionCompleta(unittest.TestCase):
    def test_todo_junto_sin_duplicar(self):
        p = profile(
            features={"chin_projection": "retruded", "jawline_definition": "soft",
                      "has_double_chin": True, "nose_size": "large", "lip_thickness": "prominent"},
            forehead="prominent",
            hairline="m_shaped_receding",
        )
        adv = bmr.advice(p, "redonda")
        self.assertEqual(labels(adv), [
            "Rostro redondo: barba angulosa y bigote grande",
            "Barba para el mentón retraído",
            "Barba para marcar la mandíbula",
            "Barba con papada",
            "Barba y frente ancha",
            "Barba con entradas",
            "Bigote y nariz grande",
            "Bigote y labios prominentes",
        ])


if __name__ == "__main__":
    unittest.main()
