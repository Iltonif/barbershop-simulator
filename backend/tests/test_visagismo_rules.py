import unittest

from app.pipeline.style_catalog import HaircutStyle
from app.pipeline.visagismo_rules import evaluate_profile


def style(**overrides):
    base = dict(
        id="x", name="x", description="x",
        length_top_mm=40, length_sides_mm=5, length_back_mm=5,
        fade_type="ninguno", suitable_hair_types=["liso"],
        style_family=None,
    )
    base.update(overrides)
    return HaircutStyle(**base)


class TestVisagismoRules(unittest.TestCase):
    def test_no_profile_is_neutral(self):
        adj = evaluate_profile(style(), None)
        self.assertEqual(adj.score, 0)
        self.assertIsNone(adj.note)

    def test_empty_profile_is_neutral(self):
        adj = evaluate_profile(style(), {})
        self.assertEqual(adj.score, 0)
        self.assertIsNone(adj.note)

    def test_brachycephalic_boosts_compact_sides_high_top(self):
        profile = {"anatomical_metrics": {"cranial_morphology": "brachycephalic"}}
        good = style(fade_type="skin", length_top_mm=40)
        bad = style(fade_type="ninguno", length_top_mm=10)
        self.assertLess(evaluate_profile(good, profile).score, 0)
        self.assertEqual(evaluate_profile(bad, profile).score, 0)

    def test_round_face_boosts_pompadour_and_high_fade(self):
        profile = {"anatomical_metrics": {"facial_geometry": "round"}}
        pompadour = style(style_family="tupe_pompadour_clasico")
        high_fade = style(fade_type="alto")
        neutral = style(style_family="melena_larga", fade_type="ninguno")
        self.assertLess(evaluate_profile(pompadour, profile).score, 0)
        self.assertLess(evaluate_profile(high_fade, profile).score, 0)
        self.assertEqual(evaluate_profile(neutral, profile).score, 0)

    def test_receding_hairline_prioritizes_texture_penalizes_slick_back(self):
        profile = {"hair_physical_metrics": {"frontal_hairline_shape": "m_shaped_receding"}}
        texture = style(style_family="corte_texturizado_general")
        slick_back = style(style_family="tupe_pompadour_clasico", length_top_mm=90)
        short_pompadour = style(style_family="tupe_pompadour_clasico", length_top_mm=50)
        self.assertLess(evaluate_profile(texture, profile).score, 0)
        self.assertGreater(evaluate_profile(slick_back, profile).score, 0)
        self.assertEqual(evaluate_profile(short_pompadour, profile).score, 0)

    def test_zero_maintenance_strongly_boosts_buzz_and_penalizes_styling_heavy(self):
        profile = {"lifestyle_and_preferences": {"daily_maintenance_commitment": "zero_minutes"}}
        buzz = style(style_family="buzz_corto_uniforme", fade_type="ninguno", length_top_mm=6)
        needs_styling = style(style_family="tupe_pompadour_clasico", length_top_mm=90)
        neutral = style(style_family="fade_undercut_textura", length_top_mm=40, fade_type="medio")
        self.assertEqual(evaluate_profile(buzz, profile).score, -2)
        self.assertEqual(evaluate_profile(needs_styling, profile).score, 2)
        self.assertEqual(evaluate_profile(neutral, profile).score, 0)

    def test_infrequent_visits_penalizes_skin_fade_boosts_taper_and_classic(self):
        profile = {"lifestyle_and_preferences": {"barbershop_visit_frequency_days": 30}}
        skin = style(fade_type="skin")
        taper = style(fade_type="bajo")
        classic = style(style_family="clasico_raya_lateral", fade_type="ninguno")
        neutral = style(fade_type="alto", style_family="fade_undercut_textura")
        self.assertGreater(evaluate_profile(skin, profile).score, 0)
        self.assertLess(evaluate_profile(taper, profile).score, 0)
        self.assertLess(evaluate_profile(classic, profile).score, 0)
        self.assertEqual(evaluate_profile(neutral, profile).score, 0)

    def test_frequent_visits_does_not_trigger_rule(self):
        profile = {"lifestyle_and_preferences": {"barbershop_visit_frequency_days": 10}}
        skin = style(fade_type="skin")
        self.assertEqual(evaluate_profile(skin, profile).score, 0)

    def test_combined_rules_accumulate_and_notes_concatenate(self):
        profile = {
            "anatomical_metrics": {"cranial_morphology": "brachycephalic", "facial_geometry": "round"},
        }
        s = style(fade_type="skin", length_top_mm=40, style_family="tupe_pompadour_clasico")
        adj = evaluate_profile(s, profile)
        # brachycephalic (-1) + round face via pompadour/high fade (-1) = -2
        self.assertEqual(adj.score, -2)
        self.assertIsNotNone(adj.note)
        self.assertIn("braquicéfala", adj.note)
        self.assertIn("redonda", adj.note)


if __name__ == "__main__":
    unittest.main()
