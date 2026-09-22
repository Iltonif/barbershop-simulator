"""Mapa de crecimiento -> zonas/direcciones -> reglas de corte, y frecuencia
de retoque. Ver growth_analysis.py, growth_rules.py y maintenance.py."""

import unittest
from datetime import datetime, timezone

import numpy as np

from app.pipeline import growth_analysis as ga
from app.pipeline import maintenance
from app.pipeline.growth_rules import evaluate_growth
from app.pipeline.recommender import recommend_styles
from app.pipeline.style_catalog import HaircutStyle


def world(direction, r=0.3):
    """Punto de mundo en la dirección `direction` desde el centro de la cabeza."""
    d = np.asarray(direction, float)
    local = ga.HEAD_CENTER + d / np.linalg.norm(d) * r
    return local * ga.WORLD_SCALE + ga.WORLD_OFFSET


def stroke(a_dir, b_dir):
    a, b = world(a_dir), world(b_dir)
    mid = world((np.asarray(a_dir, float) + np.asarray(b_dir, float)) / 2)
    return {"x1": a[0], "y1": a[1], "z1": a[2], "x2": b[0], "y2": b[1], "z2": b[2],
            "points": [list(a), list(mid), list(b)]}


def whorl(direction, rotation="horario"):
    p = world(direction)
    return {"x": p[0], "y": p[1], "z": p[2], "rotation": rotation}


def style(**kw):
    base = dict(id="x", name="Corte", description="", length_top_mm=40, length_sides_mm=10,
                length_back_mm=10, fade_type="ninguno", suitable_hair_types=["liso"])
    base.update(kw)
    return HaircutStyle(**base)


CROWN = (0.0, 0.85, -0.5)
FRONT_TOP = (0.0, 0.7, 0.7)
FRONT_LOW = (0.0, 0.4, 0.92)
NAPE = (0.0, -0.3, -0.95)


class ZonesTest(unittest.TestCase):
    def test_zones(self):
        self.assertEqual(ga.zone_of(world(CROWN)), "coronilla")
        self.assertEqual(ga.zone_of(world((0, 1, 0.1))), "arriba")
        self.assertEqual(ga.zone_of(world(FRONT_TOP)), "frente")
        self.assertEqual(ga.zone_of(world((1, 0.2, 0))), "lateral_izq")
        self.assertEqual(ga.zone_of(world((-1, 0.2, 0))), "lateral_dcha")
        self.assertEqual(ga.zone_of(world(NAPE)), "nuca")

    def test_direction_ignores_the_part_that_leaves_the_head(self):
        # Por la frente, de arriba hacia las cejas: "hacia abajo" o "delante", nunca "arriba".
        d = ga.direction_of(world(FRONT_TOP), world(FRONT_LOW))
        self.assertIn(d, ("abajo", "delante"))
        self.assertEqual(ga.direction_of(world((0.9, 0.3, 0.2)), world((0.9, 0.3, -0.3))), "atras")

    def test_summary_and_natural_part(self):
        s = ga.summarize({"strokes": [stroke(FRONT_TOP, FRONT_LOW)], "whorls": [whorl(CROWN, "antihorario")]})
        self.assertEqual(s.front_growth, "delante")
        self.assertEqual(s.crown_whorls, 1)
        self.assertEqual((s.natural_part, s.natural_part_source), ("derecha", "remolino"))
        self.assertTrue(any("Raya natural" in l for l in s.lines()))
        # Sin remolino en la coronilla: la raya sale del lado al que va el pelo de la frente.
        s = ga.summarize({"strokes": [stroke((0.1, 0.6, 0.8), (-0.4, 0.55, 0.72))]})
        self.assertEqual(s.front_growth, "lado_derecha")
        self.assertEqual(s.natural_part, "izquierda")

    def test_old_maps_without_points_still_work(self):
        st = stroke(FRONT_TOP, FRONT_LOW)
        del st["points"]
        st["zone"] = "flequillo"
        self.assertEqual(ga.summarize({"strokes": [st], "whorls": []}).front_growth, "delante")


class GrowthRulesTest(unittest.TestCase):
    def labels(self, st, growth_map):
        return [(e.score < 0, e.label) for e in evaluate_growth(st, ga.summarize(growth_map))]

    def test_crown_whorl_depends_on_length(self):
        m = {"whorls": [whorl(CROWN)]}
        self.assertEqual(self.labels(style(length_top_mm=40), m), [(False, "Se levanta en la coronilla")])
        self.assertEqual(self.labels(style(length_top_mm=12), m), [(True, "Remolino sin levantar")])
        self.assertEqual(self.labels(style(length_top_mm=90), m), [(True, "El peso aplana el remolino")])
        self.assertEqual(self.labels(style(length_top_mm=40, description="con textura"), m),
                         [(True, "La textura disimula el remolino")])

    def test_front_growth(self):
        forward = {"strokes": [stroke(FRONT_TOP, FRONT_LOW)]}
        self.assertIn((True, "A favor del crecimiento"), self.labels(style(name="Flequillo recto"), forward))
        self.assertIn((False, "Contra el crecimiento"), self.labels(style(name="Peinado hacia atrás"), forward))
        back = {"strokes": [stroke(FRONT_LOW, FRONT_TOP)]}
        self.assertIn((True, "A favor del crecimiento"), self.labels(style(name="Tupé", length_top_mm=80), back))

    def test_front_whorl(self):
        m = {"whorls": [whorl(FRONT_TOP)]}
        self.assertIn((False, "El remolino abre el flequillo"), self.labels(style(name="Flequillo recto"), m))
        self.assertIn((True, "El tupé aprovecha el remolino"), self.labels(style(name="Tupé", length_top_mm=80), m))

    def test_natural_part_only_for_side_parts(self):
        m = {"whorls": [whorl(CROWN, "horario")]}
        self.assertIn((True, "Raya a su izquierda"),
                      self.labels(style(name="Corte clásico con raya lateral", length_top_mm=90), m))
        self.assertNotIn((True, "Raya a su izquierda"), self.labels(style(name="Buzz", length_top_mm=90), m))

    def test_nape(self):
        m = {"whorls": [whorl(NAPE)]}
        self.assertEqual(self.labels(style(length_back_mm=20), m), [(False, "Se nota en la nuca")])
        self.assertEqual(self.labels(style(length_back_mm=5, fade_type="bajo"), m), [(True, "Nuca degradada")])
        self.assertEqual(self.labels(style(length_back_mm=80), m), [(True, "El largo tapa la nuca")])


class MaintenanceTest(unittest.TestCase):
    def test_weeks(self):
        self.assertEqual(maintenance.weeks_for("skin", 30, 0), (2, 3))
        self.assertEqual(maintenance.weeks_for("bajo", 40, 3), (3, 4))
        self.assertEqual(maintenance.weeks_for("ninguno", 90, 60, "medio"), (4, 6))
        self.assertEqual(maintenance.weeks_for("ninguno", 200, 150, "largo"), (6, 8))

    def test_next_visit(self):
        now = datetime(2026, 9, 22, tzinfo=timezone.utc)
        nv = maintenance.next_visit("2026-09-01T10:00:00+00:00", (2, 3), now=now)
        self.assertEqual(nv["days_left"], 0)
        # Si él viene más a menudo que el intervalo del corte, manda su frecuencia.
        nv = maintenance.next_visit("2026-09-01T10:00:00+00:00", (4, 6), client_frequency_days=14, now=now)
        self.assertEqual(nv["days_left"], -7)
        self.assertIsNone(maintenance.next_visit(None, (2, 3)))

    def test_ties_go_to_frequent_retouch_but_respect_the_client(self):
        recs = recommend_styles("liso")
        firsts = [r.maintenance_weeks[0] for r in recs[:5]]
        self.assertTrue(all(w == 2 for w in firsts), firsts)
        # Viene cada 8 semanas: lo que se ve crecido a las 3 lleva aviso y va detrás.
        profile = {"lifestyle_and_preferences": {"barbershop_visit_frequency_days": 56}}
        recs = recommend_styles("liso", visagismo_profile=profile)
        top = recs[0]
        self.assertGreaterEqual(top.maintenance_weeks[1], 7)
        skinny = [r for r in recs if r.maintenance_weeks == (2, 3)]
        self.assertTrue(all(r.warnings for r in skinny))
        # Viene cada 2 semanas: los de retoque frecuente llevan razón a favor.
        profile = {"lifestyle_and_preferences": {"barbershop_visit_frequency_days": 14}}
        recs = recommend_styles("liso", visagismo_profile=profile)
        self.assertIn("Encaja con sus visitas", [e.label for e in recs[0].reasons])


if __name__ == "__main__":
    unittest.main()
