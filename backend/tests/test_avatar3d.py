"""Gemelo digital 3D: llamada a Tripo (simulada) y medidas sobre la malla.

Igual que el resto del repo, no se llama a ninguna API real: se sustituye
la capa de red. Las medidas se comprueban contra el maniquí del propio
repo (`frontend/assets/head.glb`), que es simétrico y de forma conocida.
"""

import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from app import config
from app.pipeline import avatar3d, mesh_metrics

HEAD = Path(__file__).resolve().parents[2] / "frontend" / "assets" / "head.glb"


class MeshMetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verts = mesh_metrics.load_glb(HEAD.read_bytes())

    def test_reads_the_glb(self):
        self.assertGreater(len(self.verts), 10000)
        self.assertEqual(self.verts.shape[1], 3)
        with self.assertRaises(ValueError):
            mesh_metrics.load_glb(b"esto no es un glb")

    def test_measures_a_symmetric_head(self):
        m = mesh_metrics.measure(self.verts)
        self.assertTrue(m["estimado"])
        self.assertGreater(m["simetria_pct"], 95)          # el maniquí es simétrico
        self.assertTrue(0.8 < m["proporcion"] < 2.2)
        self.assertTrue(0.6 < m["mandibula_indice"] < 1.2)
        self.assertEqual(mesh_metrics.classify(m)["profile_type"], "straight")

    def test_does_not_depend_on_la_escala_ni_el_giro(self):
        """Tripo no devuelve el modelo ni centrado ni mirando a un lado
        concreto: las medidas tienen que salir igual."""
        th = np.radians(55)
        rot = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
        moved = self.verts @ rot.T * 2.3 + np.array([5.0, 0.0, -4.0])
        a, b = mesh_metrics.measure(self.verts), mesh_metrics.measure(moved)
        self.assertAlmostEqual(a["proporcion"], b["proporcion"], delta=0.15)
        self.assertAlmostEqual(a["mandibula_indice"], b["mandibula_indice"], delta=0.08)
        self.assertAlmostEqual(a["perfil_grados"], b["perfil_grados"], delta=6)

    def test_a_flipped_head_is_turned_back(self):
        """Si el modelo viene mirando al revés, `orient` lo gira: las
        medidas tienen que salir las mismas que con la cabeza de frente."""
        flipped = self.verts.copy()
        flipped[:, [0, 2]] *= -1
        a, b = mesh_metrics.measure(self.verts), mesh_metrics.measure(flipped)
        self.assertAlmostEqual(a["proporcion"], b["proporcion"], delta=0.05)
        self.assertAlmostEqual(a["perfil_grados"], b["perfil_grados"], delta=2)


class TripoClientTest(unittest.TestCase):
    """Se sustituye `_request`/`_download`, que es todo lo que toca la red."""

    def setUp(self):
        self.key = config.TRIPO_API_KEY
        config.TRIPO_API_KEY = "test"

    def tearDown(self):
        config.TRIPO_API_KEY = self.key

    def _fake(self, statuses):
        calls = []

        def request(method, path, **kw):
            calls.append((method, path, kw.get("body")))
            if path == "/upload/sts":
                return {"bucket": "tripo-data", "key": f"k{len(calls)}"}
            if path == "/task" and method == "POST":
                return {"task_id": "t1"}
            return {"status": statuses.pop(0), "output": {"pbr_model": "https://x/model.glb"}}
        return request, calls

    def test_uploads_each_photo_and_waits(self):
        request, calls = self._fake(["running", "success"])
        with mock.patch.object(avatar3d, "_request", request), \
             mock.patch.object(avatar3d, "_download", return_value=b"glb!"):
            out = avatar3d.generate_twin(
                {"frontal": b"a", "perfil_izquierdo": b"b", "perfil_derecho": b"c"}, poll_seconds=0)
        self.assertEqual(out, b"glb!")
        self.assertEqual(sum(1 for c in calls if c[1] == "/upload/sts"), 3)
        body = json.loads([c for c in calls if c[1] == "/task"][0][2])
        self.assertEqual(body["type"], "multiview_to_model")
        # Orden de Tripo: [frontal, izquierda, atrás, derecha]; la de atrás
        # va vacía porque no se le hace foto a la nuca.
        self.assertEqual([bool(f) for f in body["files"]], [True, True, False, True])

    def test_errors(self):
        request, _ = self._fake(["failed"])
        with mock.patch.object(avatar3d, "_request", request):
            with self.assertRaises(avatar3d.Avatar3DError):
                avatar3d.generate_twin({"frontal": b"a", "perfil_izquierdo": b"b"}, poll_seconds=0)
        with self.assertRaises(avatar3d.Avatar3DError):      # solo una foto
            avatar3d.generate_twin({"frontal": b"a"}, poll_seconds=0)
        config.TRIPO_API_KEY = None
        with self.assertRaises(avatar3d.Avatar3DNotConfigured):
            avatar3d.generate_twin({"frontal": b"a", "perfil_izquierdo": b"b"}, poll_seconds=0)


if __name__ == "__main__":
    unittest.main()
