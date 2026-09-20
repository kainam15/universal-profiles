"""basic 容器验收器的负向协议测试；数值为测试输入，不是采集证据。"""
import json
import unittest

from acprof.capabilities import apply_collection_result, apply_runtime_validation, measurement_report
from scripts.check_onnx_basic import audit_basic_capabilities, audit_basic_rows


def audit_rows():
    contract = {
        'schema_version': 1, 'request_count': 2,
        'variants': [{'count': 2, 'contract': {
            'task': 'tabular-regression', 'batch_size': 2,
            'input': {'rows': 4, 'feature_dim': 8}, 'output': {'shape': [4, 1]},
        }}],
    }
    return [{
        'warmup': warmup, 'repeat_idx': repeat, 'status': 'ok', 'error': '',
        'latency_app_s': '0.5', 'throughput_samples_per_s': '4',
        'container_cpu_util_avg_pct': '0', 'container_mem_usage_avg_bytes': '33554432',
        'resource_usage_iters': '2', 'repeat_in_window': '2',
        'workload_contract': json.dumps(contract),
    } for warmup, repeat in [('1', '0'), ('0', '0'), ('0', '1')]]


class ONNXBasicAuditTests(unittest.TestCase):
    def test_image_scenario_checks_processed_geometry_separately_from_source(self):
        rows = audit_rows()
        for row in rows:
            row['throughput_samples_per_s'] = '2'
            contract = json.loads(row['workload_contract'])
            contract['variants'][0]['contract'] = {
                'task': 'image-classification', 'batch_size': 1,
                'input': {'images': {'count': 1, 'original_resolution': [28, 28],
                                     'processed_resolution': [14, 14], 'processed_shape': [1, 3, 14, 14]}},
                'output': {'shape': [1, 2]},
            }
            row['workload_contract'] = json.dumps(contract)
        audit_basic_rows(rows, scenario='image')
        contract['variants'][0]['contract']['input']['images']['processed_resolution'] = [28, 28]
        rows[1]['workload_contract'] = json.dumps(contract)
        with self.assertRaisesRegex(ValueError, 'actual workload.*processed_resolution'):
            audit_basic_rows(rows, scenario='image')

    def test_text_scenario_uses_actual_tokens_and_its_own_batch_throughput(self):
        rows = audit_rows()
        for row in rows:
            row['throughput_samples_per_s'] = '2'
            contract = json.loads(row['workload_contract'])
            contract['variants'][0]['contract'] = {
                'task': 'text-classification', 'batch_size': 1,
                'input': {'text': {'tokens': 4, 'actual_tokens_per_sample': [4],
                                  'content_tokens_per_sample': [2], 'padding': 'none', 'truncation': 'reject'}},
                'output': {'shape': [1, 2]},
            }
            row['workload_contract'] = json.dumps(contract)
        audit_basic_rows(rows, scenario='text')
        contract['variants'][0]['contract']['input']['text']['tokens'] = 2
        rows[1]['workload_contract'] = json.dumps(contract)
        with self.assertRaisesRegex(ValueError, 'actual workload'):
            audit_basic_rows(rows, scenario='text')

    def test_available_execution_without_protocol_or_task_evidence_is_not_success(self):
        for absent in ('protocol', 'task'):
            with self.subTest(absent=absent):
                capability = measurement_report('basic', gpu_modes=['off'])
                layers = {name: {'status': 'verified'} for name in ('protocol', 'task') if name != absent}
                apply_runtime_validation(capability, {'devices': {'off': {'status': 'ok', 'validation': layers}}})
                apply_collection_result(capability, audit_rows())
                self.assertTrue(capability.to_dict()['requested_measurements_complete'])
                with self.assertRaisesRegex(ValueError, 'CPU execution.*verified'):
                    audit_basic_capabilities(capability)

    def test_complete_output_validation_and_metrics_allow_basic_success(self):
        capability = measurement_report('basic', gpu_modes=['off'])
        apply_runtime_validation(capability, {'devices': {'off': {'status': 'ok', 'validation': {
            'protocol': {'status': 'verified'}, 'task': {'status': 'verified'},
        }}}})
        apply_collection_result(capability, audit_rows())
        audit_basic_capabilities(capability)

    def test_measured_zero_cpu_is_valid_and_missing_cpu_is_not(self):
        rows = audit_rows()
        audit_basic_rows(rows)
        rows[1]['container_cpu_util_avg_pct'] = 'nan'
        with self.assertRaisesRegex(ValueError, 'required finite basic metric.*container_cpu'):
            audit_basic_rows(rows)

    def test_throughput_must_match_existing_batch_latency_definition(self):
        rows = audit_rows()
        rows[1]['throughput_samples_per_s'] = '8'
        with self.assertRaisesRegex(ValueError, 'batch_size / application latency'):
            audit_basic_rows(rows)

    def test_actual_workload_cannot_drop_measured_requests(self):
        rows = audit_rows()
        rows[2]['repeat_in_window'] = '3'
        with self.assertRaisesRegex(ValueError, 'actual workload request count'):
            audit_basic_rows(rows)

    def test_wrong_output_shape_fails_even_with_valid_performance_metrics(self):
        rows = audit_rows()
        contract = json.loads(rows[1]['workload_contract'])
        contract['variants'][0]['contract']['output']['shape'] = [3, 1]
        rows[1]['workload_contract'] = json.dumps(contract)
        with self.assertRaisesRegex(ValueError, 'actual workload differs'):
            audit_basic_rows(rows)

    def test_unrequested_metrics_cannot_be_filled_with_zero(self):
        rows = audit_rows()
        rows[0]['cpu_energy_total_j'] = '0'
        with self.assertRaisesRegex(ValueError, 'unrequested full metric'):
            audit_basic_rows(rows)


if __name__ == '__main__':
    unittest.main()
