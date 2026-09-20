"""The host can inspect upstream registries without executing model code."""
import unittest

from scripts.export_transformers_support import export_support


class TransformersCatalogTests(unittest.TestCase):
    def test_static_composition_preserves_native_heads_without_executing_source(self):
        source = b'''
raise RuntimeError("upstream module must not execute")
MODEL_MAPPING_NAMES = OrderedDict([("encoder", "EncoderModel")])
MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict([("decoder", "DecoderForCausalLM")])
MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES = OrderedDict([
    *list(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.items()),
    ("vision", ("VisionModel", "VisionVariant")),
])
'''
        catalog = export_support(source, "test")
        mapping = catalog["mappings"]["MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES"]
        self.assertEqual(mapping["decoder"], "DecoderForCausalLM")
        self.assertEqual(mapping["vision"], ("VisionModel", "VisionVariant"))
        self.assertEqual(len(catalog["source_sha256"]), 64)

    def test_changed_registry_shape_is_rejected_instead_of_publishing_partial_data(self):
        source = '''
MODEL_MAPPING_NAMES = OrderedDict([("encoder", "EncoderModel")])
MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict([("decoder", "DecoderForCausalLM")])
MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES = REPLACEMENT
'''
        for expression in ("OrderedDict(build_dynamic_mapping())", "build_dynamic_mapping()", '{"vision": "Model"}'):
            with self.subTest(expression=expression), self.assertRaisesRegex(ValueError, "unsupported Auto registry"):
                export_support(source.replace("REPLACEMENT", expression).encode(), "test")

    def test_calls_inside_entries_are_rejected_without_execution(self):
        with self.assertRaises(ValueError):
            export_support(b'MODEL_MAPPING_NAMES = OrderedDict([("a", load_remote_code())])', "test")


if __name__ == "__main__":
    unittest.main()
