import unittest
from unittest import mock

import numpy as np

from app.pipeline import visagismo_vision_analysis as vision


def _fake_image():
    return np.full((40, 30, 3), 128, np.uint8)


def _fake_tool_use(input_dict):
    block = mock.Mock()
    block.type = "tool_use"
    block.input = input_dict
    response = mock.Mock()
    response.content = [block]
    return response


class TestAnalyzeFacialTraitsWithVision(unittest.TestCase):
    def test_raises_not_configured_when_no_api_key(self):
        with mock.patch.object(vision, "ANTHROPIC_API_KEY", None):
            with self.assertRaises(vision.VisionAnalysisNotConfigured):
                vision.analyze_facial_traits_with_vision(_fake_image(), _fake_image(), _fake_image())

    def test_success_path_keeps_only_valid_values(self):
        fake_response = _fake_tool_use({
            "profile_type": "straight",
            "ears_projection": "prominent_protruding",
            "neck_proportions": "proportional",
            "eyebrow_type": "arched",
            "chin_projection": "balanced",
            "jawline_definition": "defined",
            "confidence_notes": "Fotos claras, buena confianza.",
        })
        fake_client = mock.Mock()
        fake_client.messages.create.return_value = fake_response

        with mock.patch.object(vision, "ANTHROPIC_API_KEY", "sk-fake"):
            with mock.patch("anthropic.Anthropic", return_value=fake_client):
                result = vision.analyze_facial_traits_with_vision(_fake_image(), _fake_image(), _fake_image())

        self.assertEqual(result.facial_features_profile, {
            "profile_type": "straight",
            "ears_projection": "prominent_protruding",
            "neck_proportions": "proportional",
            "eyebrow_type": "arched",
            "chin_projection": "balanced",
            "jawline_definition": "defined",
        })
        self.assertTrue(any("Fotos claras" in w for w in result.warnings))

        # Se llamó forzando la única herramienta, con las 3 fotos en el mensaje.
        _, kwargs = fake_client.messages.create.call_args
        self.assertEqual(kwargs["tool_choice"], {"type": "tool", "name": "record_facial_traits"})
        self.assertEqual(kwargs["model"], vision.ANTHROPIC_MODEL)
        content = kwargs["messages"][0]["content"]
        image_blocks = [b for b in content if b.get("type") == "image"]
        self.assertEqual(len(image_blocks), 3)

    def test_success_path_routes_new_fields_to_correct_destination(self):
        """`intellectual_zone_forehead` (Frente) vive en un diccionario distinto
        (`facial_horizontal_zones_ratio`) al resto, y `has_double_chin` es un
        booleano, no una categoría -- ver el docstring del módulo (ampliación
        sept 2026, Pedro pidió que la IA también juzgue estos 4 campos)."""
        fake_response = _fake_tool_use({
            "profile_type": "straight",
            "ears_projection": "prominent_protruding",
            "neck_proportions": "proportional",
            "eyebrow_type": "arched",
            "chin_projection": "balanced",
            "jawline_definition": "defined",
            "intellectual_zone_forehead": "proportional",
            "nose_size": "small",
            "lip_thickness": "thin",
            "has_double_chin": False,
            "confidence_notes": "Fotos claras, buena confianza.",
        })
        fake_client = mock.Mock()
        fake_client.messages.create.return_value = fake_response

        with mock.patch.object(vision, "ANTHROPIC_API_KEY", "sk-fake"):
            with mock.patch("anthropic.Anthropic", return_value=fake_client):
                result = vision.analyze_facial_traits_with_vision(_fake_image(), _fake_image(), _fake_image())

        self.assertEqual(result.facial_features_profile, {
            "profile_type": "straight",
            "ears_projection": "prominent_protruding",
            "neck_proportions": "proportional",
            "eyebrow_type": "arched",
            "chin_projection": "balanced",
            "jawline_definition": "defined",
            "nose_size": "small",
            "lip_thickness": "thin",
            "has_double_chin": False,
        })
        self.assertEqual(result.facial_horizontal_zones_ratio, {"intellectual_zone_forehead": "proportional"})
        # No debe quedar como "sin juzgar" solo por ser `False`.
        self.assertFalse(any("papada" in w and "no pudo juzgar" in w for w in result.warnings))

    def test_boolean_field_true_is_kept(self):
        fake_response = _fake_tool_use({
            "profile_type": "straight",
            "ears_projection": "flat",
            "neck_proportions": "proportional",
            "eyebrow_type": "arched",
            "chin_projection": "balanced",
            "jawline_definition": "defined",
            "has_double_chin": True,
            "confidence_notes": "Se aprecia papada en las fotos de perfil.",
        })
        fake_client = mock.Mock()
        fake_client.messages.create.return_value = fake_response

        with mock.patch.object(vision, "ANTHROPIC_API_KEY", "sk-fake"):
            with mock.patch("anthropic.Anthropic", return_value=fake_client):
                result = vision.analyze_facial_traits_with_vision(_fake_image(), _fake_image(), _fake_image())

        self.assertEqual(result.facial_features_profile["has_double_chin"], True)

    def test_boolean_field_with_non_bool_value_is_discarded_with_warning(self):
        fake_response = _fake_tool_use({
            "profile_type": "straight",
            "ears_projection": "flat",
            "neck_proportions": "proportional",
            "eyebrow_type": "arched",
            "chin_projection": "balanced",
            "jawline_definition": "defined",
            "has_double_chin": "sí",
            "confidence_notes": "",
        })
        fake_client = mock.Mock()
        fake_client.messages.create.return_value = fake_response

        with mock.patch.object(vision, "ANTHROPIC_API_KEY", "sk-fake"):
            with mock.patch("anthropic.Anthropic", return_value=fake_client):
                result = vision.analyze_facial_traits_with_vision(_fake_image(), _fake_image(), _fake_image())

        self.assertNotIn("has_double_chin", result.facial_features_profile)
        self.assertTrue(any("valor no reconocido" in w and "papada" in w for w in result.warnings))

    def test_null_fields_from_model_are_left_out_with_warning(self):
        fake_response = _fake_tool_use({
            "profile_type": None,
            "ears_projection": "flat",
            "neck_proportions": None,
            "eyebrow_type": None,
            "chin_projection": None,
            "jawline_definition": None,
            "confidence_notes": "",
        })
        fake_client = mock.Mock()
        fake_client.messages.create.return_value = fake_response

        with mock.patch.object(vision, "ANTHROPIC_API_KEY", "sk-fake"):
            with mock.patch("anthropic.Anthropic", return_value=fake_client):
                result = vision.analyze_facial_traits_with_vision(_fake_image(), _fake_image(), _fake_image())

        self.assertEqual(result.facial_features_profile, {"ears_projection": "flat"})
        self.assertTrue(any("no pudo juzgar con confianza" in w for w in result.warnings))
        # No hay nota de confianza vacía como aviso.
        self.assertFalse(any(w.startswith("Nota de la IA") for w in result.warnings))

    def test_unrecognized_value_is_discarded_with_warning(self):
        fake_response = _fake_tool_use({
            "profile_type": "algo_raro_que_el_modelo_inventó",
            "ears_projection": "flat",
            "neck_proportions": "proportional",
            "eyebrow_type": "arched",
            "chin_projection": "balanced",
            "jawline_definition": "defined",
            "confidence_notes": "",
        })
        fake_client = mock.Mock()
        fake_client.messages.create.return_value = fake_response

        with mock.patch.object(vision, "ANTHROPIC_API_KEY", "sk-fake"):
            with mock.patch("anthropic.Anthropic", return_value=fake_client):
                result = vision.analyze_facial_traits_with_vision(_fake_image(), _fake_image(), _fake_image())

        self.assertNotIn("profile_type", result.facial_features_profile)
        self.assertTrue(any("valor no reconocido" in w and "perfil de nariz" in w for w in result.warnings))

    def test_api_error_wrapped(self):
        import anthropic as anthropic_module

        fake_client = mock.Mock()
        fake_client.messages.create.side_effect = anthropic_module.APIConnectionError(request=mock.Mock())

        with mock.patch.object(vision, "ANTHROPIC_API_KEY", "sk-fake"):
            with mock.patch("anthropic.Anthropic", return_value=fake_client):
                with self.assertRaises(vision.VisionAnalysisError):
                    vision.analyze_facial_traits_with_vision(_fake_image(), _fake_image(), _fake_image())

    def test_no_tool_use_block_raises(self):
        text_block = mock.Mock()
        text_block.type = "text"
        response = mock.Mock()
        response.content = [text_block]
        fake_client = mock.Mock()
        fake_client.messages.create.return_value = response

        with mock.patch.object(vision, "ANTHROPIC_API_KEY", "sk-fake"):
            with mock.patch("anthropic.Anthropic", return_value=fake_client):
                with self.assertRaises(vision.VisionAnalysisError):
                    vision.analyze_facial_traits_with_vision(_fake_image(), _fake_image(), _fake_image())


if __name__ == "__main__":
    unittest.main()
