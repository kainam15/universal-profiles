"""Offline tiny-model checks; run in the pinned multimodal/CV image.

The ordinary host suite skips these without Torch/Transformers. No Hub access
or pretrained weights are required, and all models execute on a single CPU thread.
"""

import base64
import importlib.util
import io
import unittest
import wave

import numpy as np
from PIL import Image

from acprof.container.handlers.multimodal import MultimodalHandler


_RUNTIME_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ('torch', 'transformers', 'tokenizers'))


@unittest.skipUnless(_RUNTIME_AVAILABLE, 'requires the Transformers multimodal container')
class MultimodalGenerationRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, threads)

    def setUp(self):
        import torch
        self.addCleanup(torch.set_rng_state, torch.get_rng_state().clone())

    def tokenizer(self, omni=False):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import Qwen2TokenizerFast
        tokens = ['[UNK]', '[PAD]', '[EOS]', '<|vision_start|>', '<|vision_end|>', '<|image_pad|>', '<|video_pad|>', '<|AUDIO|>', '<|audio_bos|>', '<|audio_eos|>', 'Describe', 'media', '.', '<|audio_pad|>', '<|audio_start|>', '<|audio_end|>']
        backend = Tokenizer(models.WordLevel({token: i for i, token in enumerate(tokens)}, unk_token='[UNK]'))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = Qwen2TokenizerFast(tokenizer_object=backend, unk_token='[UNK]', pad_token='[PAD]', eos_token='[EOS]', additional_special_tokens=tokens[3:10] + tokens[13:], model_max_length=2048)
        tokenizer.model_input_names = ['input_ids', 'attention_mask']
        tokenizer.image_token = '<|image_pad|>'
        tokenizer.video_token = '<|video_pad|>'
        tokenizer.vision_bos_token = '<|vision_start|>'
        tokenizer.vision_eos_token = '<|vision_end|>'
        if omni:
            tokenizer.audio_token = '<|audio_pad|>'
            tokenizer.audio_bos_token = '<|audio_start|>'
            tokenizer.audio_eos_token = '<|audio_end|>'
        return tokenizer

    def template(self, omni=False):
        audio = '<|audio_start|><|audio_pad|><|audio_end|>' if omni else '<|audio_bos|><|AUDIO|><|audio_eos|>'
        return "{% for message in messages %}{% for item in message.content %}{% if item.type == 'image' %}<|vision_start|><|image_pad|><|vision_end|>{% elif item.type == 'video' %}<|vision_start|><|video_pad|><|vision_end|>{% elif item.type == 'audio' %}" + audio + "{% else %} {{ item.text }} {% endif %}{% endfor %}{% endfor %}"

    def image(self):
        out = io.BytesIO()
        Image.new('RGB', (28, 28), 'white').save(out, 'PNG')
        return base64.b64encode(out.getvalue()).decode()

    def audio(self):
        out = io.BytesIO()
        with wave.open(out, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(np.zeros(16000, dtype='<i2').tobytes())
        return base64.b64encode(out.getvalue()).decode()

    def test_tiny_qwen2_vl_generates_from_image_and_video(self):
        import torch
        from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration, Qwen2VLImageProcessor, Qwen2VLProcessor, Qwen2VLVideoProcessor
        torch.manual_seed(0)
        config = Qwen2VLConfig(
            text_config={'vocab_size': 32, 'hidden_size': 32, 'intermediate_size': 64, 'num_hidden_layers': 1, 'num_attention_heads': 4, 'num_key_value_heads': 2, 'rope_scaling': {'type': 'mrope', 'mrope_section': [2, 1, 1]}, 'pad_token_id': 1, 'eos_token_id': 2},
            vision_config={'depth': 1, 'embed_dim': 32, 'hidden_size': 32, 'num_heads': 4, 'patch_size': 14, 'spatial_merge_size': 2, 'temporal_patch_size': 2},
            image_token_id=5, video_token_id=6, vision_start_token_id=3, vision_end_token_id=4,
        )
        model = Qwen2VLForConditionalGeneration(config).eval()
        processor = Qwen2VLProcessor(
            image_processor=Qwen2VLImageProcessor(min_pixels=28*28, max_pixels=28*28),
            video_processor=Qwen2VLVideoProcessor(min_pixels=28*28, max_pixels=28*28),
            tokenizer=self.tokenizer(), chat_template=self.template(),
        )
        handler = MultimodalHandler()
        for task, sample, scale_type, scale in (
            ('image-text-to-text', {'image_base64': self.image()}, 'resolution_px', 28),
            ('video-text-to-text', {'video_frames_base64': [self.image(), self.image()], 'fps': 2}, 'frame_count', 2),
        ):
            with self.subTest(task=task):
                ctx = {'model': model, 'processor': processor, 'mode': 'generate', 'model_type': 'qwen2_vl', 'task_type': task, 'device': 'cpu'}
                payload = {'samples': [{'text': 'Describe media.', **sample}], 'params': {'max_new_tokens': 2}, 'input_scale_type': scale_type, 'input_scale': scale}
                processed = handler.preprocess(ctx, payload)
                self.assertEqual(processed['_effective_input_scale'], scale)
                first = handler.postprocess(ctx, handler.predict(ctx, processed))
                second = handler.postprocess(ctx, handler.predict(ctx, processed))
                self.assertEqual(first, second)
                self.assertEqual(first['output_type'], 'text')

    def test_tiny_qwen2_audio_generates_from_audio_and_text(self):
        import torch
        from transformers import Qwen2AudioConfig, Qwen2AudioForConditionalGeneration, Qwen2AudioProcessor, WhisperFeatureExtractor
        torch.manual_seed(0)
        config = Qwen2AudioConfig(
            audio_config={'d_model': 32, 'encoder_layers': 1, 'encoder_attention_heads': 4, 'encoder_ffn_dim': 64, 'num_mel_bins': 128, 'max_source_positions': 1500},
            text_config={'model_type': 'qwen2', 'vocab_size': 32, 'hidden_size': 32, 'intermediate_size': 64, 'num_hidden_layers': 1, 'num_attention_heads': 4, 'num_key_value_heads': 2, 'pad_token_id': 1, 'eos_token_id': 2},
            audio_token_index=7,
        )
        model = Qwen2AudioForConditionalGeneration(config).eval()
        processor = Qwen2AudioProcessor(feature_extractor=WhisperFeatureExtractor(feature_size=128), tokenizer=self.tokenizer(), chat_template=self.template())
        ctx = {'model': model, 'processor': processor, 'mode': 'generate', 'model_type': 'qwen2_audio', 'task_type': 'audio-text-to-text', 'device': 'cpu'}
        payload = {'samples': [{'text': 'Describe media.', 'audio_base64': self.audio(), 'sampling_rate': 16000}], 'params': {'max_new_tokens': 2}, 'input_scale_type': 'duration_s', 'input_scale': 1}
        handler = MultimodalHandler()
        processed = handler.preprocess(ctx, payload)
        self.assertEqual(processed['_effective_input_scale'], 1)
        self.assertIn('input_features', processed['inputs'])
        first = handler.postprocess(ctx, handler.predict(ctx, processed))
        second = handler.postprocess(ctx, handler.predict(ctx, processed))
        self.assertEqual(first, second)
        self.assertEqual(first['output_type'], 'text')

    def test_omni_seed_repeats_audio_and_restores_external_rng(self):
        from types import SimpleNamespace
        import torch

        class RandomAudioModel:
            def generate(self, input_ids, **kwargs):
                return input_ids, torch.randn(16)

        handler = MultimodalHandler()
        ctx = {'model': RandomAudioModel(), 'mode': 'omni'}
        processed = {'inputs': {'input_ids': torch.tensor([[1, 2]])}, 'prompt_length': 0, 'params': {'max_new_tokens': 2, 'seed': 42}}
        torch.manual_seed(17)
        before = torch.get_rng_state().clone()
        first = handler.predict(ctx, processed)['generated'][1]
        after = torch.get_rng_state().clone()
        second = handler.predict(ctx, processed)['generated'][1]
        self.assertTrue(torch.equal(before, after))
        self.assertTrue(torch.equal(first, second))
        processed['params']['seed'] = 99
        different = handler.predict(ctx, processed)['generated'][1]
        self.assertFalse(torch.equal(first, different))

    def test_real_omni_processor_retains_audio_image_and_video(self):
        from types import SimpleNamespace
        import torch
        from transformers import Qwen2VLImageProcessor, Qwen2VLVideoProcessor, Qwen2_5OmniProcessor, WhisperFeatureExtractor
        processor = Qwen2_5OmniProcessor(
            image_processor=Qwen2VLImageProcessor(min_pixels=28*28, max_pixels=28*28),
            video_processor=Qwen2VLVideoProcessor(min_pixels=28*28, max_pixels=28*28),
            feature_extractor=WhisperFeatureExtractor(feature_size=128), tokenizer=self.tokenizer(omni=True), chat_template=self.template(omni=True),
        )
        model = SimpleNamespace(device=torch.device('cpu'), dtype=torch.float32)
        ctx = {'model': model, 'processor': processor, 'mode': 'omni', 'model_type': 'qwen2_5_omni', 'task_type': 'any-to-any', 'device': 'cpu'}
        payload = {'samples': [{'text': 'Describe media.', 'image_base64': self.image(), 'video_frames_base64': [self.image(), self.image()], 'fps': 2, 'audio_base64': self.audio(), 'sampling_rate': 16000}], 'params': {'return_audio': True}, 'input_scale_type': 'duration_s', 'input_scale': 1}
        processed = MultimodalHandler().preprocess(ctx, payload)
        self.assertTrue({'input_features', 'pixel_values', 'pixel_values_videos'} <= set(processed['inputs']))
        self.assertEqual(processed['_effective_input_scale'], 1)


if __name__ == '__main__':
    unittest.main()
