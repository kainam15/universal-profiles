import unittest

import client


class EffectiveEnergyWarningTests(unittest.TestCase):
    def test_negative_effective_metrics_are_reported_per_field(self) -> None:
        warnings = client._eff_negative_warnings(
            avg_power_eff_w=-0.1,
            peak_power_eff_w=0.0,
            energy_eff_j=-0.001,
        )

        self.assertEqual(warnings, ["avg_power_eff_w<0", "energy_eff_j<0"])


if __name__ == "__main__":
    unittest.main()
