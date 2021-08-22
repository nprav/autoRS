"""Unit tests for autoRS.spectrum."""

# Standard library imports
import unittest
import os

# Third party imports
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured

# Local Application Imports
from context import autoRS
from autoRS.spectrum import (
    get_asme_frequencies,
    get_default_frequencies,
    response_spectrum,
)
from autoRS.rw import read_csv_multi
from utility import low_pass_filter


class TestRsUtilityFuncs(unittest.TestCase):
    def test_asme_frequencies(self):
        frqs = get_asme_frequencies()
        print(len(frqs), frqs)

        self.assertEqual(frqs[0], 0.1)
        self.assertEqual(frqs[-1], 34)
        self.assertGreaterEqual(len(frqs), 73)

    def test_default_frequencies(self):
        frqs1 = get_default_frequencies()
        print(len(frqs1), frqs1)

        frqs2 = get_default_frequencies(high_frequency=True)
        print(len(frqs2), frqs2)

        self.assertEqual(frqs1[0], 0.1)
        self.assertEqual(frqs1[-1], 100)
        self.assertGreaterEqual(len(frqs2), 100)

        self.assertEqual(frqs2[0], 0.1)
        self.assertEqual(frqs2[-1], 1000)
        self.assertGreaterEqual(len(frqs2), 112)


class TestRS(unittest.TestCase):
    lp_freq = 25
    th_path = os.path.join("../tests", "test_resources", "multi_col.csv",)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Extract THs
        self._extract_eg_ths()

        # Choose range of damping values
        self.zeta_arr = np.linspace(0.01, 0.8, num=self.num_ths)

        # Generate sample RS
        self._generate_lp_sample_spectra()

    def _extract_eg_ths(self):
        th_raw = np.genfromtxt(
            self.th_path,
            delimiter=",",
            skip_header=1,
            names=True,
            deletechars=" !#$%&'()*+,-./:;<=>?[\\]^{|}~",
        )

        # Convert to unstructured ndarray and remove nan values (last column)
        th_raw = structured_to_unstructured(th_raw)
        self.time = th_raw[:, 0]
        self.th_arr = th_raw[:, 1:-1]

        # Get number of THs
        self.num_ths = self.th_arr.shape[1]
        self.setnums = list(range(1, self.num_ths + 1))

    def _generate_lp_sample_spectra(self):
        self.lp_rs_dict = {}
        for method in autoRS.RS_METHODS:
            self.lp_rs_dict[method] = [
                response_spectrum(
                    low_pass_filter(self.th_arr[:, i], self.lp_freq, time=self.time),
                    self.time,
                    zeta=self.zeta_arr[i],
                    method=method,
                    high_frequency=False,
                )[0]
                for i in range(self.num_ths)
            ]

    def test_step_fft_comparison(self):
        # Absolute ratio of differences in response values
        diff_ratios = [
            np.absolute(
                (
                    self.lp_rs_dict["fft"][setnum - 1]
                    / self.lp_rs_dict["shake"][setnum - 1]
                )
                - 1
            )
            for setnum in self.setnums
        ]

        # Verify that 95% of values vary by under 5% with the step and fft methods
        max_diffs = [np.quantile(ratios, 0.95) for ratios in diff_ratios]
        print(
            "\n95th percentile difference percentage for each TH set:\n",
            [f"{x*100:.1f}%" for x in max_diffs],
        )
        self.assertTrue(all([diff < 0.05 for diff in max_diffs]))

        # Verify that the mean difference ratio between step and fft is < 1%
        mean_diffs = [np.mean(ratios) for ratios in diff_ratios]
        print(
            "\nMean % difference ratio for each TH set:\n",
            [f"{x*100:.1f}%" for x in mean_diffs],
        )
        self.assertTrue(all([diff < 0.01 for diff in mean_diffs]))
