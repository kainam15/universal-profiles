import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from acprof.host import orchestrator
from acprof.host.detect import TaskInfo
from acprof.host.docker_runtime import ImageInfo


class MatrixPlanTests(unittest.TestCase):
    def setUp(self):
        self.task = TaskInfo('org/model', 'fill-mask', 'nlp', 'transformers_pipeline',
                             'transformers', 'a' * 40, 'manual')
        self.image = ImageInfo('sha256:' + 'b' * 64)

    def run_matrix(self, directory, **options):
        return orchestrator.run_matrix(
            self.task, self.image, [4, 1, 2], [8, 2, 4], ['off'],
            str(directory), str(directory), input_scales='16,32,64,128',
            warmup=0, repeat=1, profiling_mode='basic', **options)

    def test_matrix_freezes_actual_case_and_scale_order(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            orchestrator, 'run_single_case', return_value=''
        ) as case:
            self.run_matrix(tmp)
            path = Path(tmp) / 'matrix_plan.json'
            self.assertTrue(path.is_file(), 'formal collection needs a frozen plan')
            plan = json.loads(path.read_text())
            self.assertEqual(len(plan['cases']), 9)
            self.assertEqual(len(plan['plan_sha256']), 64)
            self.assertTrue(plan['algorithm_version'])
            self.assertEqual(
                [(c['cpu_cores'], c['mem_cap_gb']) for c in plan['cases']],
                [(c.kwargs['cpu'], c.kwargs['mem']) for c in case.call_args_list])
            for item, call in zip(plan['cases'], case.call_args_list):
                self.assertEqual(item['input_scales'], call.kwargs['input_scale_order'])

    def test_same_seed_produces_byte_identical_plan(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            orchestrator, 'run_single_case', return_value=''
        ):
            paths = [Path(tmp) / name for name in ('first', 'second')]
            for path in paths:
                self.run_matrix(path, matrix_order='seeded', matrix_seed=37)
            self.assertEqual((paths[0] / 'matrix_plan.json').read_bytes(),
                             (paths[1] / 'matrix_plan.json').read_bytes())

    def test_changed_options_do_not_overwrite_a_frozen_plan(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            orchestrator, 'run_single_case', return_value=''
        ):
            self.run_matrix(tmp, matrix_seed=3)
            original = (Path(tmp) / 'matrix_plan.json').read_bytes()
            with self.assertRaisesRegex(ValueError, 'matrix.*(identity|options)'):
                self.run_matrix(tmp, matrix_seed=4)
            self.assertEqual((Path(tmp) / 'matrix_plan.json').read_bytes(), original)

    def test_reuse_does_not_rebuild_or_repeat_probes(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            orchestrator, 'run_single_case', return_value=''
        ) as case:
            self.run_matrix(tmp, matrix_seed=37)
            original = case.call_args_list[:]
            case.reset_mock()
            with patch('acprof.host.matrix_plan.build_matrix_plan', side_effect=AssertionError('reshuffled')), \
                 patch('acprof.host.startup_probe.run_startup_probes', side_effect=AssertionError('reprobed')):
                self.run_matrix(tmp, matrix_seed=37)
            self.assertEqual(case.call_args_list, original)

    def test_scale_seed_does_not_depend_on_number_or_order_of_cases(self):
        from acprof.host.matrix_plan import matrix_identity, build_matrix_plan
        plans = []
        for cpus in ([1, 2], [8, 2, 1]):
            identity = matrix_identity(self.task, self.image, cpus, [4], ['off'],
                                       [16., 32., 64., 128.], order='seeded', seed=37, prune=False)
            plans.append(build_matrix_plan(identity, {}))
        def common(plan):
            return {c['cpu_cores']: (c['input_scales'], c['input_scale_seed'])
                    for c in plan['cases'] if c['cpu_cores'] in (1, 2)}
        self.assertEqual(common(plans[0]), common(plans[1]))
        self.assertNotEqual(plans[0]['cases'][0]['input_scale_seed'], '37')

    def test_corrupted_plan_is_rejected_before_formal_collection(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            orchestrator, 'run_single_case', return_value=''
        ) as case:
            self.run_matrix(tmp)
            path = Path(tmp) / 'matrix_plan.json'
            plan = json.loads(path.read_text())
            plan['cases'].reverse()
            path.write_text(json.dumps(plan))
            case.reset_mock()
            with self.assertRaisesRegex(ValueError, 'hash'):
                self.run_matrix(tmp)
            case.assert_not_called()


if __name__ == '__main__':
    unittest.main()
