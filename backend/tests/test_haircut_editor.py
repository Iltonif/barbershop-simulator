"""Tests de `haircut_editor.py` y de las comprobaciones de `/api/simulate`
que van antes de tocar la foto (proveedor y consentimiento). Sin llamadas
reales: los SDK de Gemini y fal se sustituyen por dobles, igual que se hace
con Anthropic en `test_visagismo_ai_advisor.py`."""

import io
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import config
from app.api import routes
from app.pipeline import haircut_editor as he
from app.pipeline.style_catalog import HaircutStyle

STYLE = HaircutStyle(
    id="fade-bajo", name="Fade bajo con textura arriba",
    description="Degradado bajo en laterales/nuca, largo texturizado arriba.",
    length_top_mm=50, length_sides_mm=3, length_back_mm=3, fade_type="bajo",
    suitable_hair_types=["liso", "ondulado"],
)


def _jpeg(w=40, h=60, color=(10, 20, 30)) -> bytes:
    img = np.full((h, w, 3), color, dtype=np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


class TestPrompt(unittest.TestCase):
    def test_includes_catalog_parameters_and_client_texture(self):
        prompt = he.build_edit_prompt(STYLE, "ondulado")
        self.assertIn("Fade bajo con textura arriba", prompt)
        self.assertIn("about 5 cm", prompt)          # 50 mm arriba
        self.assertIn("very short (~3 mm)", prompt)  # laterales
        self.assertIn("low fade", prompt)
        self.assertIn("wavy", prompt)
        self.assertIn("Keep the client's current hair colour", prompt)
        self.assertIn("same person", prompt)

    def test_colour_and_reference(self):
        prompt = he.build_edit_prompt(STYLE, None, target_color_hex="#aa5500", with_reference=True)
        self.assertIn("#aa5500", prompt)
        self.assertIn("SECOND image is only a reference", prompt)
        self.assertNotIn("hair texture", prompt)  # sin tipo de pelo no se inventa uno


class TestProviders(unittest.TestCase):
    def test_only_configured_providers(self):
        with patch.object(config, "GEMINI_API_KEY", None), patch.object(config, "FAL_KEY", None):
            self.assertEqual(he.available_providers(), [])
        with patch.object(config, "GEMINI_API_KEY", "g"), patch.object(config, "FAL_KEY", "f"):
            self.assertEqual([p.id for p in he.available_providers()], ["gemini", "flux"])

    def test_aspect_ratio_is_closest_supported(self):
        self.assertEqual(he._closest_aspect(np.zeros((1600, 1200, 3), np.uint8)), "3:4")
        self.assertEqual(he._closest_aspect(np.zeros((1080, 1920, 3), np.uint8)), "16:9")


class TestEditHaircut(unittest.TestCase):
    def test_gemini_returns_decoded_image_and_sends_reference(self):
        out = _jpeg(color=(200, 100, 50))
        fake_response = SimpleNamespace(candidates=[SimpleNamespace(
            finish_reason=None,
            content=SimpleNamespace(parts=[SimpleNamespace(inline_data=None),
                                           SimpleNamespace(inline_data=SimpleNamespace(data=out))]))])
        client = MagicMock()
        client.models.generate_content.return_value = fake_response
        with patch.object(config, "GEMINI_API_KEY", "g"), patch("google.genai.Client", return_value=client):
            result = he.edit_haircut(np.zeros((60, 40, 3), np.uint8), STYLE, "liso", "gemini",
                                     reference=_jpeg())
        self.assertEqual(result.shape, (60, 40, 3))
        contents = client.models.generate_content.call_args.kwargs["contents"]
        self.assertEqual(len(contents), 3)  # instrucción + cliente + referencia
        self.assertIn("SECOND image", contents[0])

    def test_gemini_without_image_is_an_error(self):
        empty = SimpleNamespace(candidates=[SimpleNamespace(finish_reason="SAFETY",
                                                            content=SimpleNamespace(parts=[]))])
        client = MagicMock()
        client.models.generate_content.return_value = empty
        with patch.object(config, "GEMINI_API_KEY", "g"), patch("google.genai.Client", return_value=client):
            with self.assertRaises(he.HaircutEditorError) as ctx:
                he.edit_haircut(np.zeros((60, 40, 3), np.uint8), STYLE, "liso", "gemini")
        self.assertIn("SAFETY", str(ctx.exception))

    def test_flux_uses_single_image_model_and_sync_mode(self):
        import base64
        data_uri = "data:image/jpeg;base64," + base64.b64encode(_jpeg()).decode()
        fake = MagicMock()
        fake.subscribe.return_value = {"images": [{"url": data_uri}]}
        with patch.object(config, "FAL_KEY", "f"), patch.object(config, "FAL_USE_REFERENCE", False), \
             patch("fal_client.SyncClient", return_value=fake):
            he.edit_haircut(np.zeros((60, 40, 3), np.uint8), STYLE, "liso", "flux", reference=_jpeg())
        app, = fake.subscribe.call_args.args
        args = fake.subscribe.call_args.kwargs["arguments"]
        self.assertEqual(app, config.FAL_MODEL)
        self.assertIn("image_url", args)
        self.assertNotIn("image_urls", args)          # sin referencia salvo FAL_USE_REFERENCE
        self.assertTrue(args["sync_mode"])            # que no quede en su historial
        self.assertNotIn("SECOND image", args["prompt"])

    def test_missing_key(self):
        with patch.object(config, "GEMINI_API_KEY", None):
            with self.assertRaises(he.HaircutEditorNotConfigured):
                he.edit_haircut(np.zeros((6, 4, 3), np.uint8), STYLE, None, "gemini")


class TestSimulateGuards(unittest.TestCase):
    """Se comprueban antes de leer la foto: no cargan ningún modelo."""

    def setUp(self):
        app = FastAPI()
        app.include_router(routes.router, prefix="/api")
        self.client = TestClient(app)
        self.files = {"photo": ("f.jpg", io.BytesIO(_jpeg()), "image/jpeg")}
        style_id = routes.load_catalog()[0].id
        self.data = {"style_id": style_id}

    def test_providers_endpoint(self):
        with patch.object(config, "GEMINI_API_KEY", "g"), patch.object(config, "FAL_KEY", None):
            res = self.client.get("/api/simulate/providers")
        self.assertEqual(res.json(), [{"id": "gemini", "label": "Gemini", "company": "Google"}])

    def test_consent_required_when_photo_leaves_the_server(self):
        with patch.object(config, "GEMINI_API_KEY", "g"):
            res = self.client.post("/api/simulate", data=self.data, files=self.files)
        self.assertEqual(res.status_code, 422)
        self.assertIn("consentimiento", res.json()["detail"])
        self.assertIn("Google", res.json()["detail"])

    def test_auto_detected_texture_is_not_sent_to_the_model(self):
        # La heurística de tipo de pelo falla a menudo: solo se manda si lo
        # ha dicho una persona (manual_hair_texture o la ficha del cliente).
        captured = {}

        def fake_edit(image, style, texture, provider, **kw):
            captured.setdefault("textures", []).append(texture)
            return np.zeros((8, 8, 3), np.uint8)

        face = SimpleNamespace(is_symmetric_enough=True, landmarks=None, face_shape="ovalada")
        auto = SimpleNamespace(texture=routes.hair_type.HairTexture("afro"))
        with patch.object(config, "GEMINI_API_KEY", "g"), \
             patch.object(routes.face_analysis, "analyze_face", return_value=face), \
             patch.object(routes.hair_segmentation, "segment_hair", return_value=np.zeros((60, 40), np.uint8)), \
             patch.object(routes, "_save_debug_hair_mask"), \
             patch.object(routes.hair_type, "classify_hair_type", side_effect=lambda *a: SimpleNamespace(texture=auto.texture)), \
             patch.object(routes.head_mesh, "build_default_growth_map", return_value=None), \
             patch.object(routes.haircut_editor, "edit_haircut", side_effect=fake_edit):
            base = {**self.data, "consent_external_photo": "true"}
            r1 = self.client.post("/api/simulate", data=base,
                                  files={"photo": ("f.jpg", io.BytesIO(_jpeg()), "image/jpeg")})
            r2 = self.client.post("/api/simulate", data={**base, "manual_hair_texture": "liso"},
                                  files={"photo": ("f.jpg", io.BytesIO(_jpeg()), "image/jpeg")})
        self.assertEqual((r1.status_code, r2.status_code), (200, 200))
        self.assertEqual(captured["textures"], [None, "liso"])
        self.assertEqual(r1.json()["provider"], "gemini")

    def test_unconfigured_provider_rejected(self):
        with patch.object(config, "GEMINI_API_KEY", None), patch.object(config, "FAL_KEY", None):
            res = self.client.post("/api/simulate", data={**self.data, "provider": "flux",
                                                          "consent_external_photo": "true"},
                                   files=self.files)
        self.assertEqual(res.status_code, 422)


if __name__ == "__main__":
    unittest.main()
