"""Reglas de rasgos faciales (`trait_rules.py`) y su efecto en el orden y
las explicaciones de `recommend_styles`."""

import unittest

from app.pipeline import trait_rules
from app.pipeline.recommender import recommend_styles
from app.pipeline.style_catalog import HaircutStyle


def style(**kw):
    base = dict(id="x", name="Corte", description="", length_top_mm=50, length_sides_mm=20,
                length_back_mm=20, fade_type="ninguno", suitable_hair_types=["liso"], style_family=None)
    base.update(kw)
    return HaircutStyle(**base)


def profile(**features):
    return {"anatomical_metrics": {"facial_features_profile": features}}


def labels(effects):
    return [e.label for e in effects]


class TestTraitRules(unittest.TestCase):
    def test_no_features_no_effects(self):
        self.assertEqual(trait_rules.evaluate_traits(style(), None), [])
        self.assertEqual(trait_rules.evaluate_traits(style(), {}), [])

    def test_prominent_ears(self):
        p = profile(ears_projection="prominent_protruding")
        high = trait_rules.evaluate_traits(style(fade_type="skin", length_sides_mm=1), p)
        covered = trait_rules.evaluate_traits(style(length_sides_mm=25), p)
        self.assertEqual(labels(high), ["Deja las orejas a la vista"])
        self.assertGreater(high[0].score, 0)
        self.assertEqual(labels(covered), ["Disimula las orejas"])
        self.assertLess(covered[0].score, 0)
        # Un buzz a #1 también deja la oreja a la vista aunque no tenga fade.
        self.assertEqual(labels(trait_rules.evaluate_traits(style(length_sides_mm=3), p)),
                         ["Deja las orejas a la vista"])

    def test_retruded_chin_and_slicked_back(self):
        p = profile(chin_projection="retruded")
        back = style(name="Pelo engominado hacia atrás", length_back_mm=40)
        nape = style(name="Corte texturizado", length_back_mm=40)
        self.assertEqual(labels(trait_rules.evaluate_traits(back, p)), ["Resalta el mentón retraído"])
        self.assertEqual(labels(trait_rules.evaluate_traits(nape, p)), ["Equilibra el mentón"])

    def test_parte_de_atras_is_not_slicked_back(self):
        s = style(name="Flequillo largo y corto en laterales y parte de atrás")
        self.assertFalse(trait_rules._es_hacia_atras(s))

    def test_convex_profile(self):
        p = profile(profile_type="convex_prominent_nose")
        back = style(name="Peinado hacia atrás sin raya")
        volume = style(name="Tupé retro", style_family="tupe_pompadour_clasico", length_top_mm=80)
        self.assertEqual(labels(trait_rules.evaluate_traits(back, p)), ["Resalta la nariz"])
        self.assertEqual(labels(trait_rules.evaluate_traits(volume, p)), ["Equilibra la nariz"])

    def test_short_neck(self):
        p = profile(neck_proportions="short_thick")
        self.assertEqual(labels(trait_rules.evaluate_traits(style(length_back_mm=150), p)), ["Acorta el cuello"])
        self.assertEqual(labels(trait_rules.evaluate_traits(style(length_back_mm=5), p)), ["Alarga el cuello"])

    def test_beard_advice(self):
        self.assertEqual(trait_rules.beard_advice(None), [])
        advice = trait_rules.beard_advice(profile(chin_projection="retruded", jawline_definition="soft"))
        self.assertEqual([a["label"] for a in advice], ["Barba para el mentón", "Barba para marcar la mandíbula"])
        shaved = {**profile(chin_projection="retruded"),
                  "lifestyle_and_preferences": {"beard_preference": "clean_shaven"}}
        self.assertTrue(trait_rules.beard_advice(shaved)[0]["detail"].startswith("Si quiere probar barba"))


class TestRecommenderExplanations(unittest.TestCase):
    def test_reasons_and_warnings_are_separated_and_order_follows_them(self):
        p = profile(ears_projection="prominent_protruding")
        recs = recommend_styles("liso", visagismo_profile=p)
        first, last = recs[0], recs[-1]
        self.assertIn("Disimula las orejas", labels(first.reasons))
        self.assertEqual(first.warnings, [])
        self.assertIn("Deja las orejas a la vista", labels(last.warnings))
        self.assertIsNotNone(last.note)  # los avisos siguen juntos en `note`

    def test_visagismo_positive_rules_are_reasons_not_warnings(self):
        p = {"lifestyle_and_preferences": {"daily_maintenance_commitment": "zero_minutes"}}
        recs = recommend_styles("liso", visagismo_profile=p)
        self.assertIn("Sin peinarse cada día", labels(recs[0].reasons))
        self.assertTrue(all("Sin peinarse cada día" not in labels(r.warnings) for r in recs))


if __name__ == "__main__":
    unittest.main()
