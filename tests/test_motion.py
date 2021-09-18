"""Unit tests for autoRS.core and autoRS.timehistory."""

# Standard library imports
import unittest
import os

# Third party imports
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured

# Local Application Imports
from context import autoRS
from autoRS.core import MotionHistory, MotionSpectra

# %% Tests
motion_cols = ("acceleration", "velocity", "displacement")


class TestMotionHistory(unittest.TestCase):

    th_path = os.path.join("../tests", "test_resources", "multi_col.csv")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Extract example time histories
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

    def test_init(self):
        """A Motion History can be initialized without any input,
        with a time or with any of the acceleraiton/ displacments/
        velocities."""

        mh = MotionHistory()
        self.assertEqual(mh.table.column_names, motion_cols)
        self.assertTrue(mh.table.shape == (0, 3))
        self.assertTrue(mh.table.index_name == "time")

        acc = self.th_arr[:, 0]
        mh = MotionHistory(acceleration=acc)
        self.assertTrue(all(mh.acceleration == acc))


class TestMotionSpectrum(unittest.TestCase):
    def test_init(self):
        """Table can be defined with data as a list of lists or a single list."""
        pass
