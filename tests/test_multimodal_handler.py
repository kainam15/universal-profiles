import base64
import contextlib
import io
import sys
import tempfile
import types
import unittest
import wave
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from acprof.container.handlers.multimodal import MultimodalHandler


def image_payload(size=(8, 8)):
    out = io.BytesIO()
    Image.new('RGB', size, 'white').save(out, format='PNG')
    return base64.b64encode(out.getvalue()).decode('ascii')


def audio_payload(rate=16000):
    out = io.BytesIO()
    with wave.open(out, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(np.zeros(160, dtype='<i2').tobytes())
    return base64.b64encode(out.getvalue()).decode('ascii')


class MultimodalHandlerTests(unittest.TestCase):
    def test_moss_uses_transcription_template_and_preserves_chunked_audio(self):
        from acprof.container.handlers.moss import MossTranscribeDiarizeHandler

        self.ctx.update(task_type='audio-text-to-text', model_type='moss_transcribe_diarize', audio_chunking=True)
        self.processor.feature_extractor.sampling_rate = 16000
        self.processor.feature_extractor.n_samples = 80
        self.processor.return_value['input_features'] = np.ones((2, 80, 3000))
        self.processor.return_value['audio_chunk_mapping'] = np.array([0, 0])
        self.processor.return_value['audio_feature_lengths'] = np.array([1, 1])
        request = {'samples': [{'text': 'Transcribe with speaker labels.', 'audio_base64': audio_payload(), 'sampling_rate': 16000}]}
        handler = MossTranscribeDiarizeHandler()
        processed = handler.preprocess(self.ctx, request)
        messages = self.processor.apply_chat_template.call_args.args[0]
        self.assertEqual([item['type'] for item in messages[0]['content']], ['text', 'audio'])
        self.assertEqual(self.processor.call_args.kwargs['audio'][0].shape, (160,))
        self.assertIn('audio_chunk_mapping', processed['inputs'])
        self.assertEqual(processed['_effective_input_scale'], 0.01)
        self.model.generate.assert_not_called()
        handler.predict(self.ctx, processed)
        self.assertIn('audio_feature_lengths', self.model.generate.call_args.kwargs)
        self.ctx['audio_chunking'] = False
        with self.assertRaisesRegex(ValueError, 'truncation'):
            self.handler.preprocess(self.ctx, request)

    def setUp(self):
        self.handler = MultimodalHandler()
        self.torch = types.ModuleType('torch')
        self.torch.float16, self.torch.float32 = 'fp16', 'fp32'
        self.torch.inference_mode = contextlib.nullcontext
        self.torch.device = lambda name: name
        self.torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        self.torch.random = types.SimpleNamespace(
            fork_rng=lambda devices: contextlib.nullcontext(),
            default_generator=types.SimpleNamespace(manual_seed=Mock()),
        )
        self.patch = patch.dict(sys.modules, {'torch': self.torch})
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.processor = Mock()
        self.processor.chat_template = 'template'
        self.processor.apply_chat_template.return_value = 'formatted prompt'
        self.processor.return_value = {'input_ids': np.array([[1, 2, 3]]), 'pixel_values': np.ones((1, 3, 8, 8))}
        self.processor.batch_decode.return_value = ['answer']
        self.processor.tokenizer.encode.return_value = [42]
        self.model = Mock(device='cpu')
        self.model.config = types.SimpleNamespace(is_encoder_decoder=False, model_type='qwen2_5_vl')
        self.model.generate.return_value = np.array([[1, 2, 3, 42]])
        self.ctx = {'model': self.model, 'processor': self.processor, 'task_type': 'image-text-to-text', 'device': 'cpu', 'model_type': 'qwen2_5_vl', 'mode': 'generate'}

    def request(self, **sample):
        return {'samples': [{'text': 'Describe the image.', 'image_base64': image_payload(), **sample}], 'params': {'max_new_tokens': 16}}

    def test_preprocess_consumes_image_and_prompt_outside_inference(self):
        processed = self.handler.preprocess(self.ctx, self.request())
        self.model.generate.assert_not_called()
        kwargs = self.processor.call_args.kwargs
        self.assertEqual(kwargs['images'][0].size, (8, 8))
        self.assertEqual(kwargs['text'], ['formatted prompt'])
        self.assertFalse(kwargs['add_special_tokens'])
        self.processor.reset_mock()
        raw = self.handler.predict(self.ctx, processed)
        self.processor.assert_not_called()
        self.model.generate.assert_called_once()
        self.assertEqual(self.model.generate.call_args.kwargs['max_new_tokens'], 16)
        result = self.handler.postprocess(self.ctx, raw)
        self.assertEqual(result['texts'], ['answer'])
        np.testing.assert_array_equal(self.processor.batch_decode.call_args.args[0], [[42]])
        self.assertEqual(result['output_token_count'], 1)

    def test_batch_and_missing_modality_fail_explicitly(self):
        for request, error in [({'samples': []}, 'batch_size=1'), ({'samples': [{}, {}]}, 'batch_size=1'), ({'samples': [{'text': 'x'}]}, 'image_base64'), (self.request(text=''), 'text')]:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                self.handler.preprocess(self.ctx, request)
        with self.assertRaisesRegex(ValueError, 'Base64'):
            self.handler.preprocess(self.ctx, self.request(image_base64='broken'))

    def test_params_reject_unknown_or_unbounded_generation(self):
        for params in ([], {'bogus': True}, {'max_new_tokens': 0}, {'max_new_tokens': True}, {'do_sample': True}, {'num_beams': 2}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                self.handler.preprocess(self.ctx, {**self.request(), 'params': params})

    def test_video_preserves_frame_order_and_fps(self):
        self.ctx.update(task_type='video-text-to-text')
        self.processor.return_value['pixel_values_videos'] = np.ones((2, 3, 8, 8))
        request = {'samples': [{'text': 'What changes?', 'video_frames_base64': [image_payload(), image_payload()], 'fps': 2.0}]}
        self.handler.preprocess(self.ctx, request)
        kwargs = self.processor.call_args.kwargs
        self.assertEqual(kwargs['videos'][0].shape, (2, 8, 8, 3))
        self.assertEqual(kwargs['fps'], 2.0)
        for fps in (0, -1, float('nan'), True):
            request['samples'][0]['fps'] = fps
            with self.subTest(fps=fps), self.assertRaisesRegex(ValueError, 'fps'):
                self.handler.preprocess(self.ctx, request)

    def test_audio_uses_native_audio_argument_and_validates_rate(self):
        self.ctx.update(task_type='audio-text-to-text', model_type='qwen2_audio')
        self.processor.feature_extractor.sampling_rate = 16000
        self.processor.return_value['input_features'] = np.ones((1, 8, 10))
        request = {'samples': [{'text': 'Describe the sound.', 'audio_base64': audio_payload(), 'sampling_rate': 16000}]}
        self.handler.preprocess(self.ctx, request)
        kwargs = self.processor.call_args.kwargs
        self.assertEqual(kwargs['audio'][0].shape, (160,))
        self.assertEqual(kwargs['sampling_rate'], 16000)
        request['samples'][0]['sampling_rate'] = 8000
        with self.assertRaisesRegex(ValueError, 'WAV header'):
            self.handler.preprocess(self.ctx, request)
        request['samples'][0]['sampling_rate'] = 16000
        self.processor.feature_extractor.sampling_rate = 24000
        with self.assertRaisesRegex(ValueError, 'processor'):
            self.handler.preprocess(self.ctx, request)

    def test_any_to_any_generates_and_summarizes_real_audio(self):
        self.ctx.update(task_type='any-to-any', model_type='qwen2_5_omni', mode='omni')
        self.model.generate.return_value = (np.array([[1, 2, 3, 42]]), np.zeros(24000))
        processed = self.handler.preprocess(self.ctx, self.request())
        messages = self.processor.apply_chat_template.call_args.args[0]
        self.assertEqual(messages[0]['role'], 'system')
        self.assertIn('generating text and speech', messages[0]['content'][0]['text'])
        result = self.handler.postprocess(self.ctx, self.handler.predict(self.ctx, processed))
        kwargs = self.model.generate.call_args.kwargs
        self.assertTrue(kwargs['return_audio'])
        self.torch.random.default_generator.manual_seed.assert_called_once_with(12345)
        self.assertEqual(kwargs['thinker_max_new_tokens'], 16)
        self.assertFalse(kwargs['talker_do_sample'])
        self.assertEqual(result['output_type'], 'text_audio')
        self.assertEqual(result['audio_num_samples'], 24000)
        self.assertEqual(result['audio_sample_rate'], 24000)
        self.assertEqual(result['audio_duration_s'], 1.0)
        self.assertNotIn('audio', result)

    def test_any_to_any_rejects_text_only_result(self):
        self.ctx.update(task_type='any-to-any', mode='omni')
        with self.assertRaisesRegex(ValueError, 'audio'):
            self.handler.postprocess(self.ctx, {'generated': np.array([[1, 2]]), 'prompt_length': 0})

    def test_retrieval_encodes_both_sides_and_scores_every_prediction(self):
        self.ctx.update(task_type='visual-document-retrieval', model_type='colpali', mode='retrieval')
        self.processor.process_images.return_value = {'pixel_values': np.ones((1, 3, 8, 8))}
        self.processor.process_queries.return_value = {'input_ids': np.array([[1, 2]])}
        self.model.side_effect = [types.SimpleNamespace(embeddings=np.ones((1, 2, 4))), types.SimpleNamespace(embeddings=np.ones((1, 3, 4)))] * 2
        self.processor.score_retrieval.return_value = np.array([[2.5]])
        processed = self.handler.preprocess(self.ctx, {**self.request(), 'params': {}})
        self.assertEqual(self.model.call_count, 0)
        raw = self.handler.predict(self.ctx, processed)
        self.handler.predict(self.ctx, processed)
        self.assertEqual(self.model.call_count, 4)
        self.assertEqual(self.processor.score_retrieval.call_count, 2)
        result = self.handler.postprocess(self.ctx, raw)
        self.assertEqual(result['scores'], [[2.5]])
        self.assertEqual(result['n_queries'], 1)
        self.assertEqual(result['n_documents'], 1)
        self.assertEqual(result['retrieval_scope'], 'query_and_document_encoding_plus_scoring')

    def test_qa_pipeline_is_split_and_reusable_without_mutating_chunks(self):
        pipe = Mock()
        pipe.model.can_generate.return_value = False
        pipe.preprocess.return_value = iter([{'input_ids': np.array([[1]]), 'words': ['hello'], 'is_last': True}])
        pipe._ensure_tensor_on_device.side_effect = lambda value, device: value
        def forward(chunk, **kwargs):
            return {'words': chunk.pop('words'), 'logits': np.ones((1, 2))}
        pipe._forward.side_effect = forward
        pipe.postprocess.return_value = [{'answer': 'hello', 'score': 0.5}]
        self.ctx.update(task_type='document-question-answering', mode='qa', pipeline=pipe)
        request = {**self.request(words=['hello'], boxes=[[0, 0, 1000, 1000]]), 'params': {}}
        processed = self.handler.preprocess(self.ctx, request)
        pipe._forward.assert_not_called()
        raw = self.handler.predict(self.ctx, processed)
        self.handler.predict(self.ctx, processed)
        self.assertEqual(pipe.preprocess.call_count, 1)
        self.assertEqual(pipe._forward.call_count, 2)
        self.assertEqual(processed['chunks'][0]['words'], ['hello'])
        result = self.handler.postprocess(self.ctx, raw)
        self.assertEqual(result['answers'][0]['answer'], 'hello')
        pipe.assert_not_called()

    def test_classification_qa_rejects_inapplicable_generation_params(self):
        pipe = Mock()
        pipe.model.can_generate.return_value = False
        self.ctx.update(task_type='visual-question-answering', mode='qa', pipeline=pipe)
        pipe.preprocess.return_value = {'input_ids': np.array([[1]])}
        with self.assertRaisesRegex(ValueError, 'generation'):
            self.handler.preprocess(self.ctx, self.request())

    def test_document_boxes_must_match_words_and_be_normalized(self):
        self.ctx.update(task_type='document-question-answering', mode='qa', pipeline=Mock())
        for sample in ({'words': ['a'], 'boxes': []}, {'words': ['a'], 'boxes': [[0, 0, 2000, 1]]}, {'words': ['a'], 'boxes': [[100, 0, 50, 1]]}):
            with self.subTest(sample=sample), self.assertRaisesRegex(ValueError, 'boxes'):
                self.handler.preprocess(self.ctx, self.request(**sample))

    def test_load_uses_builtin_classes_revision_and_eager_options(self):
        transformers = types.ModuleType('transformers')
        transformers.AutoConfig = Mock()
        transformers.AutoConfig.from_pretrained.return_value = types.SimpleNamespace(model_type='qwen2_5_vl')
        transformers.AutoProcessor = Mock()
        transformers.AutoProcessor.from_pretrained.return_value = self.processor
        transformers.AutoModelForImageTextToText = Mock()
        transformers.AutoModelForImageTextToText.from_pretrained.return_value = self.model
        with patch.dict(sys.modules, {'transformers': transformers}):
            ctx = self.handler.load('example/model', 'image-text-to-text', 'transformers_model', 'cpu', 'abc123', {'attention_implementation': 'eager'})
            self.assertIs(ctx['model'], self.model)
            kwargs = transformers.AutoModelForImageTextToText.from_pretrained.call_args.kwargs
            self.assertEqual(kwargs['revision'], 'abc123')
            self.assertEqual(kwargs['attn_implementation'], 'eager')
            self.assertFalse(kwargs['trust_remote_code'])
            with tempfile.TemporaryDirectory() as source:
                self.handler.load(source, 'image-text-to-text', 'transformers_model', 'cpu', 'abc123')
                self.assertNotIn('revision', transformers.AutoModelForImageTextToText.from_pretrained.call_args.kwargs)

    def test_effective_scale_is_verified_against_decoded_media(self):
        request = {**self.request(), 'input_scale': 8, 'input_scale_type': 'resolution_px'}
        processed = self.handler.preprocess(self.ctx, request)
        self.assertEqual(processed['_effective_input_scale'], 8)
        self.assertFalse(processed['_truncated_by_limit'])
        for scale_type, scale in [('resolution_px', 12), ('duration_s', 8)]:
            with self.subTest(scale_type=scale_type), self.assertRaisesRegex(ValueError, 'input_scale'):
                self.handler.preprocess(self.ctx, {**request, 'input_scale': scale, 'input_scale_type': scale_type})

    def test_unknown_fields_and_inapplicable_modalities_are_rejected(self):
        for request in [self.request(image_url='https://example.test/i.png'), self.request(audio_base64=audio_payload()), {**self.request(), 'bogus': True}]:
            with self.subTest(request=list(request)), self.assertRaises(ValueError):
                self.handler.preprocess(self.ctx, request)

    def test_processor_cannot_silently_drop_the_requested_modality(self):
        self.processor.return_value = {'input_ids': np.array([[1, 2]])}
        with self.assertRaisesRegex(ValueError, 'image'):
            self.handler.preprocess(self.ctx, self.request())

    def test_any_to_any_requires_return_audio_true_and_rejects_false(self):
        self.ctx.update(task_type='any-to-any', mode='omni', model_type='qwen2_5_omni')
        request = {**self.request(), 'params': {'return_audio': True}}
        self.handler.preprocess(self.ctx, request)
        request['params']['return_audio'] = False
        with self.assertRaisesRegex(ValueError, 'return_audio'):
            self.handler.preprocess(self.ctx, request)

    def test_omni_eager_probe_is_explicitly_unsupported(self):
        with patch.dict(sys.modules, {'transformers': types.ModuleType('transformers')}):
            with self.assertRaisesRegex(RuntimeError, 'token2wav.*SDPA'):
                self.handler.load('example/omni', 'any-to-any', 'transformers_model', 'cpu', load_options={'attention_implementation': 'eager'})

    def test_unimplemented_architecture_never_falls_back_to_text_generation(self):
        transformers = types.ModuleType('transformers')
        transformers.AutoConfig = Mock()
        transformers.AutoConfig.from_pretrained.return_value = types.SimpleNamespace(model_type='some_custom_model')
        transformers.AutoProcessor = Mock()
        with patch.dict(sys.modules, {'transformers': transformers}):
            for task in ('any-to-any', 'audio-text-to-text', 'visual-document-retrieval'):
                with self.subTest(task=task), self.assertRaisesRegex(ValueError, 'architecture'):
                    self.handler.load('example/model', task, 'transformers_model', 'cpu')
            transformers.AutoProcessor.from_pretrained.assert_not_called()


if __name__ == '__main__':
    unittest.main()
