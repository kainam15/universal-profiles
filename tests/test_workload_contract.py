import unittest
import csv
import io
import json

from acprof.workloads.contract import workload_contract, summarize_workload_contracts


class WorkloadContractTests(unittest.TestCase):
    def test_fast_window_contracts_roundtrip_csv_without_losing_work_counts(self):
        small = {'schema_version': 1, 'generation': {'max_output_tokens': 256, 'actual_output_tokens': 3}}
        large = {'schema_version': 1, 'generation': {'max_output_tokens': 256, 'actual_output_tokens': 7}}
        summary = summarize_workload_contracts([small] * 9000 + [large] * 1000 + [None])
        stream = io.StringIO()
        csv.writer(stream).writerow([json.dumps(summary)])
        stream.seek(0)
        recovered = json.loads(next(csv.reader(stream))[0])
        self.assertEqual(recovered['request_count'], 10001)
        self.assertEqual(recovered['variants'], [
            {'count': 9000, 'contract': small}, {'count': 1000, 'contract': large},
            {'count': 1, 'contract': None}])
        actual = sum(v['count'] * v['contract']['generation']['actual_output_tokens']
                     for v in recovered['variants'] if v['contract'] is not None)
        self.assertEqual(actual, 34000)
        self.assertLess(len(json.dumps(summary)), 1000)

    def test_nested_multimodal_samples_preserve_separate_observed_modalities(self):
        result = workload_contract(
            {'task_type': 'any-to-any'},
            {'samples': [{'text': 'describe', 'image_base64': 'unused', 'audio_base64': 'unused'}],
             'input_scale': 224, 'params': {'max_new_tokens': 256}},
            {'_effective_input_scale': 224, '_workload': {'input': {
                'images': {'count': 1, 'original_resolution': [224, 224], 'processed_shape': [1, 3, 448, 448]},
                'audio': {'audio_seconds': 2., 'sample_rate': 16000, 'channels': 1, 'processed_duration': 2.},
                'text': {'tokens': 19, 'token_scope': 'model_input_including_modality_and_special_tokens'},
            }}},
            {'task': 'any-to-any', 'actual_input_tokens': 19, 'actual_output_tokens': 3,
             'actual_output_tokens_per_sequence': [3], 'stop_reason': 'eos'},
        )
        self.assertEqual(set(result['input']) - {'actual_scale', 'planned_scale'}, {'text', 'audio', 'images'})
        self.assertEqual(result['input']['text']['tokens'], 19)
        self.assertEqual(result['input']['audio']['processed_duration'], 2.)
        self.assertEqual(result['generation']['actual_output_tokens'], 3)
        self.assertEqual(result['generation']['max_output_tokens'], 256)

    def test_unobserved_processor_geometry_is_explicitly_unknown(self):
        result = workload_contract({'task_type': 'image-classification'},
                                   {'image_base64': 'not_decoded_here'}, {'_effective_input_scale': 1},
                                   {'task': 'image-classification', 'n_results': 5})
        self.assertIsNone(result['input']['images']['processed_resolution'])
        self.assertEqual(result['input']['images']['processed_resolution_status'], 'unavailable')

    def test_resource_limits_are_not_mixed_into_workload(self):
        result = workload_contract({'task_type': 'tabular-regression'},
                                   {'input_scale': 2, 'features': [[1., 2.], [3., 4.]], 'batch_size': 1},
                                   {'_effective_input_scale': 2}, {'output_shape': [2, 1], 'n_results': 2})
        self.assertEqual(result['input']['rows'], 2)
        self.assertNotIn('cpus', result)
        self.assertNotIn('memory', result)
        self.assertEqual(result['scenario'], {'type': 'serial'})

    def test_audio_waveform_duration_does_not_claim_processor_padding_duration(self):
        result = workload_contract({'task_type': 'automatic-speech-recognition'},
                                   {'audio_base64': 'unused', 'input_scale': 0.01},
                                   {'audio': [0.] * 160, 'sample_rate': 16000,
                                    '_duration_s': 0.01, '_effective_input_scale': 0.01},
                                   {'text': 'hello'})
        self.assertEqual(result['input']['audio']['processor_input_duration'], 0.01)
        self.assertIsNone(result['input']['audio']['processed_duration'])
        self.assertEqual(result['input']['audio']['processed_duration_status'], 'unavailable')


if __name__ == '__main__':
    unittest.main()
