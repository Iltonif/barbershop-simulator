import unittest
from unittest import mock

from app.db.models import ClientProfile
from app.pipeline import visagismo_ai_advisor as advisor


def make_client(**overrides):
    base = dict(
        id="c1", created_at="now", display_name="Test",
        consent_history=True, consent_model_improvement=False, consent_save_photo=False,
        consent_ai_analysis=True,
        hair_texture_override=None, face_shape_override=None,
        custom_growth_map=None, visagismo_profile=None, notes=None,
    )
    base.update(overrides)
    return ClientProfile(**base)


class TestBuildUserMessage(unittest.TestCase):
    def test_empty_profile_marks_everything_no_especificado(self):
        msg = advisor.build_user_message(make_client())
        self.assertIn("no especificado", msg)
        self.assertNotIn("None", msg)

    def test_filled_profile_includes_values(self):
        profile = {
            "anatomical_metrics": {
                "cranial_morphology": "brachycephalic",
                "facial_geometry": "round",
                "facial_features_profile": {"profile_type": "convex_prominent_nose", "ears_projection": "prominent_protruding"},
            },
            "hair_physical_metrics": {"hair_density": "high_dense", "frontal_hairline_shape": "m_shaped_receding"},
            "lifestyle_and_preferences": {"daily_maintenance_commitment": "zero_minutes", "barbershop_visit_frequency_days": 25},
        }
        client = make_client(visagismo_profile=profile, hair_texture_override="rizado", face_shape_override="redonda")
        msg = advisor.build_user_message(client)
        self.assertIn("brachycephalic", msg)
        self.assertIn("convex_prominent_nose", msg)
        self.assertIn("prominent_protruding", msg)
        self.assertIn("m_shaped_receding", msg)
        self.assertIn("rizado", msg)
        self.assertIn("redonda", msg)
        self.assertIn("25", msg)

    def test_whorl_count_reflected(self):
        client = make_client(custom_growth_map={"whorls": [{"x": 0, "y": 0, "z": 0, "rotation": "horario"}]})
        msg = advisor.build_user_message(client)
        self.assertIn("Remolinos marcados en el mapa de crecimiento (nº): 1", msg)


class TestGenerateAIReport(unittest.TestCase):
    def test_raises_not_configured_when_no_api_key(self):
        with mock.patch.object(advisor, "ANTHROPIC_API_KEY", None):
            with self.assertRaises(advisor.AIAdvisorNotConfigured):
                advisor.generate_ai_report(make_client())

    def test_success_path_returns_text(self):
        fake_block = mock.Mock()
        fake_block.text = "### 1. DIAGNÓSTICO...\ncontenido"
        fake_response = mock.Mock()
        fake_response.content = [fake_block]

        fake_client = mock.Mock()
        fake_client.messages.create.return_value = fake_response

        with mock.patch.object(advisor, "ANTHROPIC_API_KEY", "sk-fake"):
            with mock.patch("anthropic.Anthropic", return_value=fake_client):
                result = advisor.generate_ai_report(make_client())

        self.assertIn("DIAGNÓSTICO", result)
        # Verifica que se llamó con el system prompt y el modelo correctos
        _, kwargs = fake_client.messages.create.call_args
        self.assertEqual(kwargs["system"], advisor.SYSTEM_PROMPT)
        self.assertEqual(kwargs["model"], advisor.ANTHROPIC_MODEL)
        self.assertEqual(kwargs["messages"][0]["role"], "user")

    def test_api_error_wrapped(self):
        import anthropic as anthropic_module

        fake_client = mock.Mock()
        fake_client.messages.create.side_effect = anthropic_module.APIConnectionError(request=mock.Mock())

        with mock.patch.object(advisor, "ANTHROPIC_API_KEY", "sk-fake"):
            with mock.patch("anthropic.Anthropic", return_value=fake_client):
                with self.assertRaises(advisor.AIAdvisorError):
                    advisor.generate_ai_report(make_client())

    def test_empty_response_raises(self):
        fake_response = mock.Mock()
        fake_response.content = []
        fake_client = mock.Mock()
        fake_client.messages.create.return_value = fake_response

        with mock.patch.object(advisor, "ANTHROPIC_API_KEY", "sk-fake"):
            with mock.patch("anthropic.Anthropic", return_value=fake_client):
                with self.assertRaises(advisor.AIAdvisorError):
                    advisor.generate_ai_report(make_client())


if __name__ == "__main__":
    unittest.main()
