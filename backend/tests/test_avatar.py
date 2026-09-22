"""Maniquí del cliente: rasgos -> morphs, largo de hoy y endpoints."""

import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.db.models import ClientProfile
from app.pipeline import avatar

RASGOS = Path(__file__).resolve().parents[2] / "frontend" / "assets" / "rasgos"


def client(**kw):
    base = dict(id="c", created_at="2026-09-01T00:00:00+00:00", display_name="X", consent_history=True,
                consent_model_improvement=False, consent_save_photo=False, consent_ai_analysis=False)
    base.update(kw)
    return ClientProfile(**base)


class MorphsTest(unittest.TestCase):
    def test_profile_to_morphs(self):
        c = client(face_shape_override="redonda", visagismo_profile={"anatomical_metrics": {
            "cranial_morphology": "brachycephalic",
            "facial_features_profile": {"chin_projection": "retruded", "jawline_definition": "soft",
                                        "ears_projection": "prominent_protruding", "eye_spacing": "wide_set",
                                        "profile_type": "convex_prominent_nose", "neck_proportions": "short_thick",
                                        "eyebrow_type": "prominent_ridge"},
            "facial_horizontal_zones_ratio": {"intellectual_zone_forehead": "prominent"}}})
        w = avatar.morph_weights(c)
        self.assertEqual(set(w), {"face-round", "skull-short", "chin-retruded", "jaw-soft", "ears-prominent",
                                  "eyes-wide", "nose-convex", "neck-short", "brow-ridge", "forehead-tall"})
        self.assertTrue(all(0 < v <= 1 for v in w.values()))

    def test_face_geometry_when_no_override(self):
        c = client(visagismo_profile={"anatomical_metrics": {"facial_geometry": "heart"}})
        self.assertEqual(avatar.morph_weights(c), {"face-heart": 0.8})

    def test_eye_asymmetry_needs_the_side(self):
        ff = {"eye_symmetry": "asymmetric", "eye_symmetry_percent": 20}
        c = client(visagismo_profile={"anatomical_metrics": {"facial_features_profile": ff}})
        self.assertEqual(avatar.morph_weights(c), {})   # sin saber qué ojo, no se inventa
        ff["detected_anomalies_notes"] = "Asimetría ocular detectada: el ojo izquierdo está aproximadamente un 20% más abierto"
        self.assertEqual(avatar.morph_weights(c), {"eye-r-small": 0.5})

    def test_every_morph_has_its_file(self):
        index = json.loads((RASGOS / "index.json").read_text())
        names = set(index["morphs"])
        used = {"face-oval", "face-round", "face-square", "face-rectangular", "face-diamond", "face-triangle",
                "face-heart", "skull-short", "skull-long", "chin-retruded", "chin-prominent", "jaw-soft",
                "jaw-defined", "ears-prominent", "eyes-close", "eyes-wide", "eye-l-small", "eye-r-small",
                "nose-convex", "nose-concave", "neck-short", "neck-long", "brow-ridge", "forehead-tall",
                "forehead-short"}
        self.assertEqual(names, used)
        for n in names:
            self.assertTrue((RASGOS / f"{n}.bin").exists(), n)


class LengthTest(unittest.TestCase):
    NOW = datetime(2026, 9, 22, tzinfo=timezone.utc)

    def test_from_last_cut_plus_growth(self):
        cut = {"at": (self.NOW - timedelta(days=30)).isoformat(), "top": 40, "sides": 3, "back": 3, "fade": "alto"}
        L = avatar.current_length(client(), cut, self.NOW)
        self.assertEqual(L["source"], "corte")
        self.assertAlmostEqual(L["top"], 40 + 30 * avatar.GROWTH_MM_PER_DAY, places=0)
        self.assertEqual((L["fade"], L["fade_mm"]), ("alto", L["grown_mm"]))

    def test_manual_length_wins_and_keeps_growing(self):
        c = client(current_length={"top": 100, "sides": 50, "back": 60},
                   current_length_at=(self.NOW - timedelta(days=10)).isoformat())
        L = avatar.current_length(c, {"at": self.NOW.isoformat(), "top": 5, "sides": 5, "back": 5}, self.NOW)
        self.assertEqual(L["source"], "peluquero")
        self.assertAlmostEqual(L["top"], 104.1, places=1)

    def test_growth_is_capped_and_default(self):
        cut = {"at": (self.NOW - timedelta(days=2000)).isoformat(), "top": 10, "sides": 10, "back": 10}
        L = avatar.current_length(client(), cut, self.NOW)
        self.assertAlmostEqual(L["grown_mm"], avatar.MAX_GROWTH_DAYS * avatar.GROWTH_MM_PER_DAY, places=1)
        self.assertEqual(avatar.current_length(client(), None, self.NOW)["source"], "defecto")

    def test_params(self):
        c = client(hair_color="rubio", visagismo_profile={"hair_physical_metrics": {
            "hair_pattern_shape": "coily", "frontal_hairline_shape": "m_shaped_receding", "hair_density": "low_thinning"}})
        p = avatar.avatar_params(c, None, None)
        self.assertEqual((p["hair_texture"], p["hair_color"], p["hairline"], p["hair_density"]),
                         ("afro", "rubio", "m_shaped_receding", "low_thinning"))
        self.assertEqual(avatar.avatar_params(c, None, "liso")["hair_texture"], "liso")   # manda el peluquero


if __name__ == "__main__":
    unittest.main()
