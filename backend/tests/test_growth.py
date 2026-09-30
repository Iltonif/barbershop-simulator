"""Mapa de crecimiento -> zonas/direcciones -> reglas de corte, y frecuencia
de retoque. Ver growth_analysis.py, growth_rules.py y maintenance.py."""

import unittest
from datetime import datetime, timezone

import numpy as np

from app.pipeline import growth_analysis as ga
from app.pipeline import maintenance
from app.pipeline.combined_rules import evaluate_combined
from app.pipeline.growth_rules import evaluate_growth
from app.pipeline.recommender import _efecto_forma_cara, _efecto_forma_cara_especifica, recommend_styles
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


class CombinedRulesTest(unittest.TestCase):
    """Reglas que cruzan el mapa de crecimiento con la forma de cara/
    visagismo (combined_rules.py), pedidas por Pedro para no dejar sueltas
    dos notas que en realidad están relacionadas."""

    def labels(self, st, growth_map=None, face_shape=None, profile=None):
        growth = ga.summarize(growth_map) if growth_map else None
        return [(e.score < 0, e.label) for e in evaluate_combined(st, growth, face_shape, profile)]

    def test_redonda_raya_lateral_a_favor_del_crecimiento(self):
        st = style(name="Corte clásico con raya lateral", length_top_mm=40)
        m = {"whorls": [whorl(CROWN, "horario")]}  # remolino horario -> raya natural a la izquierda
        self.assertIn((True, "Asimetría a favor del crecimiento"), self.labels(st, m, face_shape="redonda"))
        # Sin cara redonda, sin mapa de crecimiento o sin raya lateral: no se activa.
        self.assertEqual(self.labels(st, m, face_shape=None), [])
        self.assertEqual(self.labels(st, None, face_shape="redonda"), [])
        self.assertEqual(self.labels(style(length_top_mm=40), m, face_shape="redonda"), [])

    def test_alargada_remolino_corona_doble_motivo(self):
        m = {"whorls": [whorl(CROWN)]}
        st = style(name="Tupé", length_top_mm=50)
        self.assertIn((False, "Doble motivo para no dar más altura"), self.labels(st, m, face_shape="alargada"))
        # Con textura ya se disimula (growth_rules lo boostea aparte): no hace falta este aviso extra.
        self.assertEqual(self.labels(style(name="Tupé con textura", length_top_mm=50), m, face_shape="alargada"), [])
        # Fuera del rango de largo donde "gana el remolino", o sin remolino, o sin cara alargada: no se activa.
        self.assertEqual(self.labels(style(name="Tupé", length_top_mm=90), m, face_shape="alargada"), [])
        self.assertEqual(self.labels(st, None, face_shape="alargada"), [])
        self.assertEqual(self.labels(st, m, face_shape="redonda"), [])

    def test_entradas_remolino_frente_pesa_mas_taparlas(self):
        m = {"whorls": [whorl(FRONT_TOP)]}
        st = style(name="Tupé", style_family="tupe_pompadour_clasico", length_top_mm=80)
        profile = {"hair_physical_metrics": {"frontal_hairline_shape": "m_shaped_receding"}}
        self.assertIn((False, "El impulso del remolino expone las entradas"), self.labels(st, m, profile=profile))
        # Sin remolino en la frente, sin entradas registradas, o con un corte que no sea ese: no se activa.
        self.assertEqual(self.labels(st, None, profile=profile), [])
        self.assertEqual(self.labels(st, m, profile=None), [])
        self.assertEqual(self.labels(style(name="Buzz", length_top_mm=6), m, profile=profile), [])

    def test_combined_effects_reach_recommend_styles(self):
        recs = recommend_styles(
            "liso",
            face_shape="redonda",
            growth_map={"whorls": [whorl(CROWN, "horario")], "strokes": []},
        )
        raya = next(r for r in recs if "raya lateral" in r.style.name.lower())
        self.assertTrue(any(e.label == "Asimetría a favor del crecimiento" for e in raya.reasons))


class NamedStyleFaceShapeTest(unittest.TestCase):
    """Recomendación de forma de cara para 5 cortes CONCRETOS del catálogo
    (sept 2026, documento aportado por Pedro) -- ver
    `HaircutStyle.recommended_face_shapes` y
    `recommender._efecto_forma_cara_especifica`. A diferencia de
    `CombinedRulesTest`/`_efecto_forma_cara` (reglas GENERALES por
    atributos del corte), esto es la recomendación explícita que trae la
    ficha de un corte en particular, sin inferir nada por sus atributos."""

    def test_boost_solo_si_la_forma_de_cara_esta_en_la_lista_del_corte(self):
        st = style(name="Fade clásico", recommended_face_shapes=["redonda", "ovalada", "cuadrada"])
        efecto = _efecto_forma_cara_especifica(st, "redonda")
        self.assertIsNotNone(efecto)
        self.assertEqual(efecto.label, "Recomendado para su rostro")
        self.assertLess(efecto.score, 0)  # negativo = razón a favor
        self.assertIn("redondos", efecto.detail)
        # Forma de cara fuera de la lista, sin forma de cara, o corte sin
        # esa recomendación en absoluto: no se activa.
        self.assertIsNone(_efecto_forma_cara_especifica(st, "alargada"))
        self.assertIsNone(_efecto_forma_cara_especifica(st, None))
        self.assertIsNone(_efecto_forma_cara_especifica(style(name="Otro corte"), "redonda"))

    def test_convive_con_el_aviso_general_para_pompadour_moderno_alargada(self):
        # Contradicción conocida y deliberada (ver el comentario en
        # recommender.py): "Pompadour moderno" trae "alargada" en su
        # propia recomendación, pero la regla general avisa en contra de
        # dar más altura con cara alargada para toda la familia
        # tupé/pompadour. Las dos señales conviven -- no se resuelve a
        # mano -- así que un corte de esa familia con esa recomendación
        # saca a la vez un aviso general Y una razón a favor específica.
        st = style(name="Pompadour moderno", style_family="tupe_pompadour_clasico",
                   fade_type="bajo", length_top_mm=70, recommended_face_shapes=["ovalada", "cuadrada", "alargada"])
        general = _efecto_forma_cara(st, "alargada")
        especifico = _efecto_forma_cara_especifica(st, "alargada")
        self.assertIsNotNone(general)
        self.assertGreater(general.score, 0)  # positivo = aviso
        self.assertIsNotNone(especifico)
        self.assertLess(especifico.score, 0)  # negativo = razón a favor

    def test_llega_hasta_recommend_styles_con_el_catalogo_real(self):
        recs = recommend_styles("liso", face_shape="redonda")
        fade_clasico = next(r for r in recs if r.style.id == "fade-clasico")
        self.assertTrue(any(e.label == "Recomendado para su rostro" for e in fade_clasico.reasons))
        # Un corte del catálogo real sin esta recomendación no la saca.
        clasico_raya = next(r for r in recs if r.style.id == "clasico-raya")
        self.assertFalse(any(e.label == "Recomendado para su rostro" for e in clasico_raya.reasons))


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

    def test_ties_go_to_frequent_retouch_and_visit_frequency_is_ignored(self):
        recs = recommend_styles("liso")
        firsts = [r.maintenance_weeks[0] for r in recs[:5]]
        self.assertTrue(all(w == 2 for w in firsts), firsts)
        # Cada cuánto viene no cambia nada: ni orden ni etiquetas.
        for dias in (14, 56):
            profile = {"lifestyle_and_preferences": {"barbershop_visit_frequency_days": dias}}
            other = recommend_styles("liso", visagismo_profile=profile)
            self.assertEqual([r.style.id for r in other], [r.style.id for r in recs])
            self.assertFalse(any(e.label in ("Aguanta entre visitas", "Se nota crecido pronto",
                                             "Se verá crecido antes", "Encaja con sus visitas")
                                 for r in other for e in r.reasons + r.warnings))


class BarberSheetTest(unittest.TestCase):
    """La ficha "cómo pedirlo" (barber_sheet.py) y el % de encaje."""

    def test_sheet_reads_the_style_and_the_growth_map(self):
        from app.pipeline import barber_sheet
        st = style(name="Textured crop", fade_type="medio", length_top_mm=55, length_sides_mm=6, length_back_mm=8)
        rows = {r["campo"]: r["valor"] for r in barber_sheet.build_sheet(
            st, "liso", ga.summarize({"strokes": [stroke(FRONT_TOP, FRONT_LOW)]}))}
        self.assertIn("Fade medio", rows["Laterales"])
        self.assertIn("cm", rows["Parte superior"])
        self.assertEqual(rows["Nuca"], "Cónica, degradada")
        self.assertEqual(rows["Peinado"], "Hacia delante, con volumen")
        self.assertIn("semanas", rows["Mantenimiento"])
        # Rapado: ni producto ni textura que explicar.
        rapado = {r["campo"]: r["valor"] for r in barber_sheet.build_sheet(style(length_top_mm=6), "liso")}
        self.assertNotIn("Producto", rapado)
        self.assertIn("máquina", rapado["Parte superior"])

    def test_match_percent(self):
        from app.pipeline.recommender import match_percent
        self.assertEqual(match_percent(0), 60)          # sin señales
        self.assertGreater(match_percent(-3), match_percent(0))
        self.assertLess(match_percent(3), match_percent(0))
        self.assertTrue(20 <= match_percent(-99) <= 99 and 20 <= match_percent(99) <= 99)


if __name__ == "__main__":
    unittest.main()
