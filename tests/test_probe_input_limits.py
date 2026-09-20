"""Optional probe limit evidence remains separate from actual truncation."""
import unittest

from acprof.host.input_plan import _parse_probe_response, _probe_exceeds_limit


class ProbeInputLimitTests(unittest.TestCase):
    def test_legacy_probe_response_remains_usable(self):
        response = _parse_probe_response({'effective_input_scale': 8, 'truncated_by_limit': False,
                                          'reason': 'within model limit'}, 'legacy')
        self.assertFalse(response['limit_exceeded'])
        self.assertFalse(_probe_exceeds_limit(response))

    def test_rejected_input_and_actual_truncation_are_both_unusable_but_distinct(self):
        for truncated, exceeded in ((True, False), (False, True)):
            with self.subTest(truncated=truncated, exceeded=exceeded):
                response = _parse_probe_response({'effective_input_scale': 20,
                    'truncated_by_limit': truncated, 'limit_exceeded': exceeded,
                    'reason': 'model input constraint'}, 'candidate')
                self.assertEqual(response['truncated_by_limit'], truncated)
                self.assertEqual(response['limit_exceeded'], exceeded)
                self.assertTrue(_probe_exceeds_limit(response))

    def test_malformed_optional_limit_flag_is_not_coerced_to_success(self):
        for value in ('false', 0, None):
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, 'non-boolean limit_exceeded'):
                _parse_probe_response({'effective_input_scale': 20, 'truncated_by_limit': False,
                                       'limit_exceeded': value, 'reason': 'bad flag'}, 'candidate')


if __name__ == '__main__':
    unittest.main()
