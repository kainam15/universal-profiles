"""Audio generation uses native interfaces, independent of checkpoint names."""

import contextlib
import sys
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from acprof.container.handlers.multimodal import MultimodalHandler
from acprof.host.detect import TaskInfo
from acprof.host.task_support import TaskSupportError, require_task_support
from acprof.runtime_profiles import select_runtime_profile
from tests.test_multimodal_handler import audio_payload


def task(model_type, **kwargs):
    values = dict(model_id='unseen/audio-checkpoint', pipeline_tag='audio-text-to-text',
                  task_family='multimodal', runtime_backend='transformers_model',
                  library_name='transformers', model_revision='a' * 40,
                  detection_method='hub_api', model_config={'model_type': model_type})
    values.update(kwargs)
    return TaskInfo(**values)


class AudioGenerationResolutionTests(unittest.TestCase):
    def test_native_audio_architectures_share_preflight_without_id_rules(self):
        for architecture, library in (('voxtral', 'vllm'), ('qwen2_audio', 'transformers'),
                                      ('granite_speech', 'transformers')):
            with self.subTest(architecture=architecture):
                info = task(architecture, library_name=library)
                require_task_support(info)
                self.assertEqual(select_runtime_profile(info).profile_id, 'multimodal-transformers4576')
                self.assertEqual(info.model_adapter, 'family-default')
                self.assertEqual(info.model_resolution['status'], 'candidate')

    def test_new_auto_registration_selects_new_environment(self):
        info = task('audioflamingo3')
        require_task_support(info)
        self.assertEqual(select_runtime_profile(info).profile_id, 'multimodal-transformers560-cu128')

    def test_text_generation_subconfig_uses_registered_auto_not_full_speech_output(self):
        from acprof import model_resolution

        config = {'model_type': 'qwen2_5_omni', 'thinker_config': {
            'model_type': 'qwen2_5_omni_thinker', 'audio_config': {'model_type': 'qwen2_5_omni_audio_encoder'},
        }}
        info = task('qwen2_5_omni', model_config=config)
        require_task_support(info)
        self.assertEqual(select_runtime_profile(info).profile_id, 'multimodal-transformers560-cu128')
        self.assertEqual(model_resolution.audio_text_loader('5.6.0', config),
                         ('AutoModelForImageTextToText', 'thinker_config'))

    def test_unknown_or_custom_architectures_still_fail_before_build(self):
        for config in ({'model_type': 'not_registered'}, {'model_type': 'custom', 'auto_map': {'AutoConfig': 'custom.Config'}}):
            with self.subTest(config=config), self.assertRaises(TaskSupportError):
                require_task_support(task(config['model_type'], model_config=config))

    def test_submodels_cannot_hide_an_unknown_parent_or_discard_audio(self):
        for config in (
            {'model_type': 'not_registered', 'head': {'model_type': 'qwen2_5_omni_thinker'}},
            {'model_type': 'qwen2_5_omni', 'text_config': {'model_type': 'qwen2'}},
            {'model_type': 'qwen2_5_omni', 'head_a': {'model_type': 'qwen2_5_omni_thinker'},
             'head_b': {'model_type': 'qwen3_omni_moe_thinker'}},
        ):
            with self.subTest(config=config), self.assertRaises(TaskSupportError):
                require_task_support(task(config['model_type'], model_config=config))


class AudioGenerationHandlerTests(unittest.TestCase):
    def setUp(self):
        self.handler = MultimodalHandler()
        self.torch = types.SimpleNamespace(float16='fp16', float32='fp32', inference_mode=contextlib.nullcontext)
        self.processor = Mock()
        self.processor.chat_template = None
        self.processor.feature_extractor = types.SimpleNamespace(sampling_rate=16000, n_samples=480000)
        self.processor.batch_decode.return_value = ['sound']
        self.processor.tokenizer.encode.return_value = [42]
        self.model = Mock(device='cpu')
        self.model.config = types.SimpleNamespace(is_encoder_decoder=False)
        self.model.generate.return_value = np.array([[1, 2, 42]])
        self.ctx = dict(model=self.model, processor=self.processor, task_type='audio-text-to-text',
                        mode='generate', model_type='voxtral', device='cpu')
        self.request = {'samples': [{'text': 'Describe the sound.', 'audio_base64': audio_payload(), 'sampling_rate': 16000}],
                        'params': {'max_new_tokens': 2}}

    def test_load_uses_the_same_auto_registry_as_preflight(self):
        for architecture in ('voxtral', 'qwen2_audio', 'granite_speech'):
            with self.subTest(architecture=architecture):
                config = types.SimpleNamespace(model_type=architecture, to_dict=lambda: {'model_type': architecture})
                transformers = types.SimpleNamespace(__version__='4.57.6', AutoConfig=Mock(), AutoProcessor=Mock(),
                                                       AutoModelForSeq2SeqLM=Mock())
                transformers.AutoConfig.from_pretrained.return_value = config
                transformers.AutoProcessor.from_pretrained.return_value = self.processor
                transformers.AutoModelForSeq2SeqLM.from_pretrained.return_value = self.model
                with patch.dict(sys.modules, torch=self.torch, transformers=transformers):
                    ctx = self.handler.load('unseen/audio-checkpoint', 'audio-text-to-text', 'transformers_model',
                                            'cpu', 'a' * 40, {'attention_implementation': 'eager'})
                kwargs = transformers.AutoModelForSeq2SeqLM.from_pretrained.call_args.kwargs
                self.assertEqual(kwargs['revision'], 'a' * 40)
                self.assertEqual(kwargs['attn_implementation'], 'eager')
                self.assertFalse(kwargs['trust_remote_code'])
                self.assertIs(ctx['model'], self.model)

    def test_native_chat_processor_receives_audio_and_text_without_jinja_requirement(self):
        paths = []

        def native(messages, **kwargs):
            self.assertEqual(set(kwargs), {'tokenize', 'return_dict', 'return_tensors'})
            self.assertEqual(len(messages), 1)
            content = messages[0]['content']
            audio = next(item for item in content if item['type'] == 'audio')
            self.assertEqual(next(item['text'] for item in content if item['type'] == 'text'), 'Describe the sound.')
            paths.append(Path(audio['path']))
            with wave.open(str(paths[-1]), 'rb') as wav:
                self.assertEqual((wav.getframerate(), wav.getnframes()), (16000, 160))
            self.assertTrue(kwargs['tokenize'])
            self.assertTrue(kwargs['return_dict'])
            self.assertEqual(kwargs['return_tensors'], 'pt')
            return {'input_ids': np.array([[1, 2]]), 'input_features': np.ones((1, 8, 10))}

        self.processor.apply_chat_template.side_effect = native
        processed = self.handler.preprocess(self.ctx, self.request)
        self.assertFalse(paths[0].exists())
        self.processor.assert_not_called()
        self.model.generate.assert_not_called()
        with patch.dict(sys.modules, torch=self.torch):
            result = self.handler.postprocess(self.ctx, self.handler.predict(self.ctx, processed))
        self.assertEqual(result['texts'], ['sound'])
        self.assertEqual(result['output_token_count'], 1)
        self.assertEqual(processed['_workload']['input']['audio']['audio_seconds'], 0.01)

    def test_native_processor_cannot_drop_audio_or_return_empty_features(self):
        for audio in ({}, {'input_features': None}, {'input_features': np.empty((0, 8))}):
            with self.subTest(keys=list(audio)), self.assertRaisesRegex(ValueError, 'audio'):
                self.processor.apply_chat_template.return_value = {'input_ids': np.array([[1, 2]]), **audio}
                self.handler.preprocess(self.ctx, self.request)

    def test_native_processor_failure_removes_temporary_audio(self):
        paths = []

        def native(messages, **kwargs):
            paths.append(Path(messages[0]['content'][0]['path']))
            raise ValueError('invalid audio template')

        self.processor.apply_chat_template.side_effect = native
        with self.assertRaisesRegex(ValueError, 'invalid audio template'):
            self.handler.preprocess(self.ctx, self.request)
        self.assertFalse(paths[0].exists())

    def test_processor_kwargs_are_routed_through_the_native_api_signature(self):
        def native(messages, *, processor_kwargs, **kwargs):
            self.assertEqual(processor_kwargs['audio_kwargs']['sampling_rate'], 16000)
            self.assertTrue(processor_kwargs['padding'])
            self.assertNotIn('sampling_rate', kwargs)
            return {'input_ids': np.array([[1, 2]]), 'input_features': np.ones((1, 8, 10))}

        self.processor.apply_chat_template = native
        self.handler.preprocess(self.ctx, self.request)


if __name__ == '__main__':
    unittest.main()
