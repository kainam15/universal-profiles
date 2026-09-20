"""Offline native audio generation: snapshots, Auto loading, processor and output.

Tiny random weights test execution contracts, not checkpoint accuracy. The
Tekken fixture uses the public mistral-common v7 audio format (Apache-2.0).
Run in either locked multimodal environment; no Hub downloads are needed.
"""

import base64
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from acprof.container.handlers.multimodal import MultimodalHandler
from acprof.container.validation import validate_output
import test_multimodal_generation_runtime as fixtures


_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ('torch', 'transformers', 'mistral_common'))


@unittest.skipUnless(_AVAILABLE, 'requires the locked native audio generation environment')
class AudioGenerationRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, threads)

    def setUp(self):
        import torch
        self.addCleanup(torch.set_rng_state, torch.get_rng_state().clone())
        torch.manual_seed(0)
        self.media = fixtures.MultimodalGenerationRuntimeTests()

    def assert_snapshot_generates(self, snapshot, expected_model, *, prefix=''):
        import torch

        handler = MultimodalHandler()
        devices = ['cpu', 'cuda:0'] if torch.cuda.is_available() else ['cpu']
        for device in devices:
            with self.subTest(device=device):
                ctx = handler.load(str(snapshot), 'audio-text-to-text', 'transformers_model', device,
                                   load_options={'attention_implementation': 'eager'})
                self.assertIsInstance(ctx['model'], type(expected_model))
                for key, value in expected_model.state_dict().items():
                    loaded = ctx['model'].state_dict()[key]
                    self.assertTrue(torch.equal(loaded.cpu(), value.to(dtype=loaded.dtype).cpu()), prefix + key)
                payload = {'samples': [{'text': 'Describe media.', 'audio_base64': self.media.audio(), 'sampling_rate': 16000}],
                           'params': {'max_new_tokens': 2}, 'input_scale_type': 'duration_s', 'input_scale': 1}
                processed = handler.preprocess(ctx, payload)
                self.assertIn('input_features', processed['inputs'])
                self.assertEqual(processed['inputs']['input_features'].device.type, device.split(':')[0])
                self.assertTrue(torch.isfinite(processed['inputs']['input_features']).all())
                raw = handler.predict(ctx, processed)
                response = handler.postprocess(ctx, raw)
                validate_output(ctx, payload, processed, raw, response)
                self.assertEqual(response['output_type'], 'text')
                self.assertGreater(response['actual_output_tokens'], 0)
                self.assertLessEqual(response['actual_output_tokens'], 2)
                again = handler.postprocess(ctx, handler.predict(ctx, processed))
                self.assertEqual(response, again)
                del ctx

    def test_qwen_audio_snapshot_uses_auto_loader_and_native_audio_messages(self):
        from transformers import Qwen2AudioConfig, Qwen2AudioForConditionalGeneration, Qwen2AudioProcessor, WhisperFeatureExtractor

        config = Qwen2AudioConfig(
            audio_config={'d_model': 32, 'encoder_layers': 1, 'encoder_attention_heads': 4, 'encoder_ffn_dim': 64,
                          'num_mel_bins': 128, 'max_source_positions': 1500},
            text_config={'model_type': 'qwen2', 'vocab_size': 32, 'hidden_size': 32, 'intermediate_size': 64,
                         'num_hidden_layers': 1, 'num_attention_heads': 4, 'num_key_value_heads': 2,
                         'pad_token_id': 1, 'eos_token_id': 2}, audio_token_index=7,
        )
        model = Qwen2AudioForConditionalGeneration(config).eval()
        model.generation_config.suppress_tokens = [i for i in range(32) if i not in (10, 11, 12)]
        processor = Qwen2AudioProcessor(feature_extractor=WhisperFeatureExtractor(feature_size=128),
                                       tokenizer=self.media.tokenizer(), chat_template=self.media.template())
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            processor.save_pretrained(directory)
            self.assert_snapshot_generates(Path(directory), model)

    def test_voxtral_snapshot_uses_native_mistral_audio_tokenizer_without_jinja(self):
        from mistral_common.tokens.tokenizers.tekken import Tekkenizer
        from transformers import AutoTokenizer, VoxtralConfig, VoxtralForConditionalGeneration, VoxtralProcessor, WhisperFeatureExtractor

        config = VoxtralConfig(
            audio_config={'model_type': 'voxtral_encoder', 'hidden_size': 32, 'num_hidden_layers': 1,
                          'num_attention_heads': 4, 'num_key_value_heads': 4, 'head_dim': 8, 'intermediate_size': 64,
                          'num_mel_bins': 128, 'max_source_positions': 1500},
            text_config={'model_type': 'llama', 'vocab_size': 320, 'hidden_size': 32, 'intermediate_size': 64,
                         'num_hidden_layers': 1, 'num_attention_heads': 4, 'num_key_value_heads': 2,
                         'head_dim': 8, 'pad_token_id': 11, 'eos_token_id': 2}, audio_token_id=24,
        )
        model = VoxtralForConditionalGeneration(config).eval()
        model.generation_config.suppress_tokens = [i for i in range(320) if not 129 <= i < 155]
        specials = list(Tekkenizer.DEPRECATED_SPECIAL_TOKENS)
        extras = {24: '[AUDIO]', 25: '[BEGIN_AUDIO]', 34: '[TRANSCRIBE]'}
        specials.extend({'rank': i, 'token_str': extras.get(i, f'<SPECIAL_{i}>'), 'is_control': True} for i in range(20, 35))
        tokenizer_data = {
            'config': {'pattern': r'[\s\S]', 'default_vocab_size': 320, 'default_num_special_tokens': 64, 'version': 'v7'},
            'vocab': [{'rank': i, 'token_bytes': base64.b64encode(bytes([i])).decode(),
                       'token_str': bytes([i]).decode(errors='replace')} for i in range(256)],
            'special_tokens': specials,
            'audio': {'sampling_rate': 16000, 'frame_rate': 12.5, 'chunk_length_s': 30.,
                      'audio_encoding_config': {'num_mel_bins': 128, 'hop_length': 160, 'window_size': 400}},
        }
        with tempfile.TemporaryDirectory() as directory:
            tokenizer_file = Path(directory) / 'tekken.json'
            tokenizer_file.write_text(json.dumps(tokenizer_data))
            config.save_pretrained(directory)
            processor = VoxtralProcessor(feature_extractor=WhisperFeatureExtractor(feature_size=128),
                                         tokenizer=AutoTokenizer.from_pretrained(directory, local_files_only=True))
            self.assertFalse(getattr(processor, 'chat_template', None))
            snapshot = Path(directory) / 'snapshot'
            model.save_pretrained(snapshot)
            processor.save_pretrained(snapshot)
            self.assert_snapshot_generates(snapshot, model)

    def test_composite_omni_snapshot_loads_only_registered_multimodal_text_head(self):
        import transformers
        from acprof.model_resolution import audio_text_loader

        if not audio_text_loader(transformers.__version__, {'model_type': 'qwen2_5_omni_thinker'}):
            # The default version cannot expose this head through Auto. Host
            # preflight must select the newer environment instead of direct classes.
            from acprof.model_resolution import supports_transformers_task
            self.assertFalse(supports_transformers_task(transformers.__version__, 'audio-text-to-text', 'qwen2_5_omni'))
            return
        from safetensors.torch import save_file
        from transformers import Qwen2_5OmniConfig, Qwen2_5OmniThinkerConfig, Qwen2_5OmniThinkerForConditionalGeneration
        from transformers import Qwen2_5OmniProcessor, Qwen2VLImageProcessor, Qwen2VLVideoProcessor, WhisperFeatureExtractor

        config = Qwen2_5OmniThinkerConfig(
            audio_config={'d_model': 32, 'encoder_layers': 1, 'encoder_attention_heads': 4, 'encoder_ffn_dim': 64,
                          'num_mel_bins': 128, 'max_source_positions': 1500, 'output_dim': 32},
            text_config={'vocab_size': 32, 'hidden_size': 32, 'intermediate_size': 64, 'num_hidden_layers': 1,
                         'num_attention_heads': 4, 'num_key_value_heads': 2, 'pad_token_id': 1, 'eos_token_id': 2,
                         'rope_parameters': {'rope_type': 'default', 'mrope_section': [2, 1, 1]}},
            vision_config={'depth': 1, 'hidden_size': 32, 'intermediate_size': 64, 'out_hidden_size': 32, 'num_heads': 4},
            audio_token_index=13, audio_start_token_id=14, audio_end_token_id=15,
            vision_start_token_id=3, vision_end_token_id=4, image_token_id=5, video_token_id=6,
        )
        model = Qwen2_5OmniThinkerForConditionalGeneration(config).eval()
        model.generation_config.suppress_tokens = [i for i in range(32) if i not in (10, 11, 12)]
        tokenizer = self.media.tokenizer(omni=True)
        tokenizer.init_kwargs['model_specific_special_tokens'] = {
            name: getattr(tokenizer, name) for name in (
                'image_token', 'video_token', 'vision_bos_token', 'vision_eos_token',
                'audio_token', 'audio_bos_token', 'audio_eos_token',
            )
        }
        processor = Qwen2_5OmniProcessor(
            feature_extractor=WhisperFeatureExtractor(feature_size=128), tokenizer=tokenizer,
            image_processor=Qwen2VLImageProcessor(), video_processor=Qwen2VLVideoProcessor(),
            chat_template=self.media.template(omni=True),
        )
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory)
            Qwen2_5OmniConfig(thinker_config=config.to_dict(), enable_audio_output=False).save_pretrained(snapshot)
            save_file({'thinker.' + key: value for key, value in model.state_dict().items()}, str(snapshot / 'model.safetensors'))
            model.generation_config.save_pretrained(snapshot)
            processor.save_pretrained(snapshot)
            self.assert_snapshot_generates(snapshot, model, prefix='thinker.')


if __name__ == '__main__':
    unittest.main()
