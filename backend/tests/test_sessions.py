"""Flujo "cliente esperando en el sillón": sesiones de cliente y peluquero,
lista de espera, favoritos, cuestionario y foto de simulación. Usa la app
entera con una base de datos temporal."""

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app import config
from app.db import database


def _jpeg():
    return cv2.imencode(".jpg", np.full((80, 60, 3), 128, np.uint8))[1].tobytes()


class SessionFlowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.patches = [
            patch.object(database, "DB_PATH", tmp / "clients.db"),
            patch.object(config, "DB_PATH", tmp / "clients.db"),
            patch.object(config, "CLIENT_PHOTOS_DIR", tmp / "photos"),
            patch.object(config, "BARBER_PIN", "2468"),
            patch.object(config, "APP_SECRET", "test-secret"),
        ]
        for p in self.patches:
            p.start()
        from app.api import auth
        auth._secret_cache = None
        database.init_db()
        from app.main import app
        self.app = app
        self.client_tab = TestClient(app)   # tablet / móvil del cliente
        self.barber = TestClient(app)       # otra pantalla, la del peluquero

    def tearDown(self):
        for p in self.patches:
            p.stop()
        from app.api import auth
        auth._secret_cache = None
        self.tmp.cleanup()

    def _register(self, **kw):
        body = {"display_name": "Pedro", "phone": "612 34 56 78", "consent_history": True,
                "consent_save_photo": True, "consent_simulation": True}
        body.update(kw)
        return self.client_tab.post("/api/me/register", json=body)

    def _barber_login(self):
        return self.barber.post("/api/barber/login", json={"pin": "2468"})

    def test_barber_routes_need_the_pin(self):
        self.assertEqual(self.barber.get("/api/clients").status_code, 401)
        self.assertEqual(self.barber.get("/api/waiting").status_code, 401)
        self.assertEqual(self.barber.post("/api/barber/login", json={"pin": "0000"}).status_code, 401)
        self.assertEqual(self._barber_login().status_code, 200)
        self.assertEqual(self.barber.get("/api/clients").status_code, 200)

    def test_client_session_does_not_open_the_barber_side(self):
        self._register()
        self.assertEqual(self.client_tab.get("/api/clients").status_code, 401)
        self.assertEqual(self.client_tab.get("/api/me").json()["display_name"], "Pedro")

    def test_first_visit_then_barber_then_next_visit(self):
        # 1. El cliente se da de alta en la tablet y responde su perfil.
        r = self._register()
        self.assertEqual(r.status_code, 200)
        me = r.json()
        self.assertEqual(me["phone"], "612345678")
        self.client_tab.patch("/api/me/questionnaire", json={"hair_pattern_shape": "straight", "face_shape": "redonda"})
        recs = self.client_tab.get("/api/me/recommendations").json()
        self.assertEqual((recs["hair_texture"], recs["hair_texture_source"]), ("liso", "cliente"))
        self.client_tab.put("/api/me/likes", json={"style_ids": [recs["recommendations"][0]["style"]["id"], "no-existe"]})
        self.assertEqual(len(self.client_tab.get("/api/me").json()["liked_styles"]), 1)

        # 2. El peluquero lo ve en la sala de espera y completa lo técnico.
        self._barber_login()
        waiting = self.barber.get("/api/waiting").json()
        self.assertEqual(len(waiting), 1)
        self.assertTrue(waiting[0]["is_new"])
        self.assertTrue(waiting[0]["questionnaire_done"])
        cid = waiting[0]["client"]["id"]
        self.barber.patch(f"/api/clients/{cid}/hair-type", json={"texture": "ondulado"})
        up = self.barber.put(f"/api/clients/{cid}/simulation-photo",
                             files={"photo": ("f.jpg", io.BytesIO(_jpeg()), "image/jpeg")})
        self.assertEqual(up.status_code, 200)
        self.assertTrue(up.json()["has_simulation_photo"])
        self.assertNotIn("simulation_photo_path", up.json())
        self.barber.patch(f"/api/waiting/{waiting[0]['id']}", json={"status": "done"})
        self.assertEqual(self.barber.get("/api/waiting").json(), [])

        # 3. Otro día (otra sesión) entra con el teléfono, escrito distinto.
        again = TestClient(self.app)
        self.assertEqual(again.post("/api/me/login", json={"phone": "+34 612345678"}).status_code, 200)
        recs = again.get("/api/me/recommendations").json()
        self.assertEqual((recs["hair_texture"], recs["hair_texture_source"]), ("ondulado", "peluquero"))
        self.assertEqual(again.get("/api/me/simulation-photo").status_code, 200)
        # Vuelve a aparecer en la lista de espera.
        self.assertEqual(len(self.barber.get("/api/waiting").json()), 1)

    def test_photo_needs_consent_and_is_deleted_when_withdrawn(self):
        cid = self._register(consent_save_photo=False).json()["id"]
        self._barber_login()
        files = {"photo": ("f.jpg", io.BytesIO(_jpeg()), "image/jpeg")}
        self.assertEqual(self.barber.put(f"/api/clients/{cid}/simulation-photo", files=files).status_code, 422)
        self.barber.patch(f"/api/clients/{cid}/consents", json={"consent_save_photo": True})
        files = {"photo": ("f.jpg", io.BytesIO(_jpeg()), "image/jpeg")}
        self.assertEqual(self.barber.put(f"/api/clients/{cid}/simulation-photo", files=files).status_code, 200)
        me = self.client_tab.patch("/api/me/consents", json={"consent_save_photo": False}).json()
        self.assertFalse(me["has_simulation_photo"])
        self.assertFalse(list((Path(self.tmp.name) / "photos").rglob("*.jpg")))

    def test_duplicate_phone_and_missing_consent(self):
        self.assertEqual(self._register(consent_history=False).status_code, 422)
        self.assertEqual(self._register().status_code, 200)
        self.assertEqual(TestClient(self.app).post("/api/me/register", json={
            "display_name": "Otro", "phone": "612345678", "consent_history": True}).status_code, 409)

    def test_client_simulation_daily_cap(self):
        cid = self._register().json()["id"]
        self._barber_login()
        self.barber.put(f"/api/clients/{cid}/simulation-photo",
                        files={"photo": ("f.jpg", io.BytesIO(_jpeg()), "image/jpeg")})
        style_id = self.client_tab.get("/api/me/recommendations").json()["recommendations"][0]["style"]["id"]
        with patch.object(config, "GEMINI_API_KEY", "g"), patch.object(config, "MAX_CLIENT_SIMULATIONS_PER_DAY", 2), \
             patch("app.pipeline.haircut_editor.edit_haircut", return_value=np.zeros((8, 8, 3), np.uint8)):
            codes = [self.client_tab.post("/api/me/simulate", json={"style_id": style_id}).status_code for _ in range(3)]
            # El peluquero no tiene tope.
            barber_code = self.barber.post(f"/api/clients/{cid}/simulate", json={"style_id": style_id}).status_code
        self.assertEqual(codes, [200, 200, 429])
        self.assertEqual(barber_code, 200)

    def test_haircut_history_max_request_and_privacy(self):
        cid = self._register().json()["id"]
        self._barber_login()
        style_id = self.client_tab.get("/api/me/recommendations").json()["recommendations"][0]["style"]["id"]
        with patch.object(config, "MAX_HAIRCUT_HISTORY", 3):
            for i in range(4):
                r = self.barber.post(f"/api/clients/{cid}/history",
                                     data={"style_id": style_id, "notes": f"visita {i}"},
                                     files={"photo": ("f.jpg", io.BytesIO(_jpeg()), "image/jpeg")})
                self.assertEqual(r.status_code, 200)
        history = r.json()
        # Solo los 3 últimos, y la foto del más antiguo se ha borrado del disco.
        self.assertEqual([h["notes"] for h in history], ["visita 3", "visita 2", "visita 1"])
        self.assertTrue(all(h["has_photo"] and h["reference_image"] for h in history))
        self.assertEqual(len(list((Path(self.tmp.name) / "photos").rglob("history/*.jpg"))), 3)
        # Corte libre, sin foto.
        self.assertEqual(self.barber.post(f"/api/clients/{cid}/history", data={"style_name": "Rapado a máquina"}).status_code, 200)
        self.assertEqual(self.barber.post(f"/api/clients/{cid}/history", data={}).status_code, 422)

        # El cliente lo ve y pide repetir uno; el peluquero lo ve en la sala.
        mine = self.client_tab.get("/api/me/history").json()
        self.assertEqual(self.client_tab.get(f"/api/me/history/{mine[1]['id']}/photo").status_code, 200)
        self.client_tab.put("/api/me/request", json={"history_id": mine[1]["id"]})
        waiting = self.barber.get("/api/waiting").json()
        self.assertEqual(waiting[0]["requested"]["id"], mine[1]["id"])
        # Otro cliente no puede ver ni pedir cortes ajenos.
        other = TestClient(self.app)
        other.post("/api/me/register", json={"display_name": "Otro", "phone": "699000111", "consent_history": True})
        self.assertEqual(other.get(f"/api/me/history/{mine[1]['id']}/photo").status_code, 404)
        self.assertEqual(other.put("/api/me/request", json={"history_id": mine[1]["id"]}).status_code, 404)
        # Retirar el permiso de fotos borra las del historial, no los cortes.
        self.client_tab.patch("/api/me/consents", json={"consent_save_photo": False})
        after = self.client_tab.get("/api/me/history").json()
        self.assertEqual(len(after), 4)
        self.assertFalse(any(h["has_photo"] for h in after))
        self.assertFalse(list((Path(self.tmp.name) / "photos").rglob("*.jpg")))

    def test_history_photo_needs_consent(self):
        cid = self._register(consent_save_photo=False).json()["id"]
        self._barber_login()
        style_id = self.client_tab.get("/api/me/recommendations").json()["recommendations"][0]["style"]["id"]
        r = self.barber.post(f"/api/clients/{cid}/history", data={"style_id": style_id},
                             files={"photo": ("f.jpg", io.BytesIO(_jpeg()), "image/jpeg")})
        self.assertEqual(r.status_code, 422)

    def test_return_due_and_next_visit(self):
        cid = self._register().json()["id"]
        self._barber_login()
        self.assertIsNone(self.client_tab.get("/api/me/next-visit").json())
        # Corte con degradado alto (retoque cada 2-3 semanas) registrado hace 30 días.
        from app.pipeline.style_catalog import load_catalog
        fade = next(s for s in load_catalog() if s.fade_type == "alto")
        self.barber.post(f"/api/clients/{cid}/history", data={"style_id": fade.id})
        with database.get_connection() as conn:
            conn.execute("UPDATE haircut_history SET created_at = datetime('now', '-30 days')")
            conn.execute("UPDATE waiting SET created_at = datetime('now', '-30 days'), updated_at = datetime('now', '-30 days')")
        nv = self.client_tab.get("/api/me/next-visit").json()
        self.assertEqual(nv["weeks"], [2, 3])
        self.assertEqual(nv["days_left"], 21 - 30)
        self.assertEqual(nv["style_name"], fade.name)
        due = self.barber.get("/api/return-due").json()
        self.assertEqual([d["client"]["id"] for d in due], [cid])
        # El cliente no puede ver la lista; y si hoy está en la sala, no sale.
        self.assertEqual(self.client_tab.get("/api/return-due").status_code, 401)
        self.barber.post(f"/api/clients/{cid}/check-in")
        self.assertEqual(self.barber.get("/api/return-due").json(), [])

    def test_growth_summary_endpoint(self):
        self._barber_login()
        body = {"strokes": [{"x1": 0, "y1": 0.9, "z1": 0.4, "x2": 0, "y2": 0.7, "z2": 0.6,
                             "points": [[0, 0.9, 0.4], [0, 0.8, 0.52], [0, 0.7, 0.6]]}],
                "whorls": [{"x": 0.02, "y": 0.72, "z": -0.46, "rotation": "horario"}]}
        r = self.barber.post("/api/growth-map/summary", json=body)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["natural_part"], "izquierda")
        body["strokes"][0]["points"] = [[0, 1]]
        self.assertEqual(self.barber.post("/api/growth-map/summary", json=body).status_code, 422)
        self.assertEqual(self.client_tab.post("/api/growth-map/summary", json={}).status_code, 401)


if __name__ == "__main__":
    unittest.main()
