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
from autoRS.core.table import EmptyInputError, SizeMismatchError
from autoRS.core.utils import cumulative_integral

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
        self.acc = self.th_arr[:, 0]
        self.dt = 0.005
        self.t = np.arange(0, len(self.acc) * self.dt, self.dt)
        self.vel = cumulative_integral(self.acc, self.t, method="trapezoidal")
        self.disp = cumulative_integral(self.vel, self.t, method="trapezoidal")

    def test_init(self):
        """A Motion History can be initialized without any input,
        with a time or with any one of the acceleraiton/ displacments/
        velocities. If not input, time will default to a dt = 1."""

        mh = MotionHistory()
        self.assertEqual(mh.table.column_names, motion_cols)
        self.assertTrue(mh.table.shape == (0, 3))
        self.assertTrue(mh.table.index_name == "time")

        mh = MotionHistory(acceleration=self.acc)
        self.assertTrue(all(mh.acceleration == self.acc))

        mh = MotionHistory(acceleration=self.acc)
        self.assertTrue(all(mh.acceleration == self.acc))
        self.assertTrue(all(mh.time == np.arange(0, len(self.acc), 1)))

        mh = MotionHistory(time=self.t, acceleration=self.acc)
        self.assertTrue(all(mh.time == self.t))
        self.assertTrue(all(mh.table.index == self.t))

        mh = MotionHistory(dt=self.dt, acceleration=self.acc)
        self.assertTrue(all(mh.time == self.t))
        self.assertTrue(mh.dt == self.dt)
        self.assertTrue(all(mh.table.index == self.t))

    def test_reset(self):
        """The private reset method enables redefinition of the MotionHistory"""
        mh = MotionHistory(dt=self.dt, acceleration=self.acc)
        mh.reset(skip_columns=("time",))
        self.assertTrue(all(np.isnan(mh.table.data[0, :])))
        self.assertTrue(mh.table.shape[0] == len(self.acc))

        mh.reset()
        self.assertTrue(mh.table.shape[0] == 0)

    def test_init_multiple(self):
        """Even if multiple columns are defined in the MotionTable, only
        one of acc/vel/disp can be explicitly defined. The other columns
        will instead be implicitly calculated."""
        mh = MotionHistory(acceleration=self.acc, displacement=self.acc)
        self.assertTrue(all(np.isnan(mh.table["displacement"])))

    def test_invalid_inputs(self):
        """The MotionHistory class does not allow invalid data inputs.
        Invalid inputs are motion arrays with size mis-matches. Invalid inputs
        raise an ValueErrors, SizeMismatchErrors, or EmptyInputErrors."""
        t_inv = [1, 2, 3]
        with self.assertRaises(SizeMismatchError):
            mh = MotionHistory(time=t_inv, acceleration=self.acc)
        with self.assertRaises(SizeMismatchError):
            mh = MotionHistory(acceleration=self.acc)
            mh.velocity = t_inv
        with self.assertRaises(EmptyInputError):
            mh = MotionHistory(acceleration=[])
        with self.assertRaises(SizeMismatchError):
            mh = MotionHistory(acceleration=self.acc)
            mh.velocity = []
        with self.assertRaises(ValueError):
            mh = MotionHistory(dt=self.dt)
        with self.assertRaises(ValueError):
            mh = MotionHistory()
            mh.time_from_dt(self.dt)

    def test_time_from_dt(self):
        mh = MotionHistory(acceleration=self.acc)
        mh.time_from_dt(self.dt)
        self.assertTrue(all(mh.time == self.t))

        mh = MotionHistory()
        mh.time_from_dt(self.dt, len(self.acc))
        self.assertTrue(all(mh.time == self.t))

    def test_motion_getters(self):
        """Motion histories can be accessed through the acceleration,
        velocity, or displacement properties. The input of a single history will
        enable the automatic generation of the other two cases."""
        mh = MotionHistory(dt=0.005, displacement=self.disp)
        acc_range = max(self.acc) - min(self.acc)
        vel_range = max(self.vel) - min(self.vel)
        print(mh.acceleration - self.acc)
        print(mh.velocity - self.vel)
        pct_diff_acc = np.abs(mh.acceleration - self.acc) / acc_range * 100
        pct_diff_vel = np.abs(mh.velocity - self.vel) / vel_range * 100
        print(max(pct_diff_acc))
        print(max(pct_diff_vel))
        self.assertTrue(all(pct_diff_acc < 2))
        self.assertTrue(all(pct_diff_vel < 0.5))


class TestMotionSpectrum(unittest.TestCase):
    def test_init(self):
        """Table can be defined with data as a list of lists or a single list."""
        pass
