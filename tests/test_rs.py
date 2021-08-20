"""Unit tests for autoRS.resp_spect."""

# Standard library imports
import unittest
import os
import shutil
import re

# Third party imports
import numpy as np
from numpy.typing import ArrayLike

# Local Application Imports
from context import autoRS
from autoRS.resp_spect import get_asme_frequencies, get_default_frequencies
from autoRS.rw import read_csv_multi


class TestRS(unittest.TestCase):
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
