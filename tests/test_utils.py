"""Unit tests for autoRS.core.utils."""

# Standard library imports
import unittest

# Third party imports
import numpy as np

# Local Application Imports
from context import autoRS
from autoRS.core.utils import (
    cumulative_integral,
    integration_methods,
)

# %% Tests


class TestCumulativeIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.alpha = 2 * np.pi
        self.t_max = 10

    def y(self, x):
        return np.sin(self.alpha * x)

    def int_y(self, x):
        return (-np.cos(self.alpha * x) + 1) / self.alpha

    def test_integrals_trapz(self):
        """The cumulative trapezoidal integration function provides reasonable
        accuracy for integrating vibratory signals. If there are 10 pts within an
        oscillation (5 pts in a single peak), the error in the peak integrated value
        is under 5%. If there are 30 points in an oscillation (15 pts in a single peak),
        the error in the peak integrated value is under 0.5%."""
        pts_to_acc = {
            10: 5,
            30: 0.5,
        }
        for pts_per_cycle, accuracy in pts_to_acc.items():
            int_t = np.arange(0, self.t_max, 1 / pts_per_cycle)
            theory_res = self.int_y(int_t)
            res = cumulative_integral(
                self.y(int_t), int_t, initial=0, method="trapezoidal"
            )
            peak_error_pct = np.abs(max(res) / max(theory_res) - 1) * 100
            print(peak_error_pct)
            self.assertTrue(peak_error_pct < accuracy)

    def test_integrals_simps_quad(self):
        """The cumulative 1/3 composite simpson's rule integration function provides
        good accuracy for integrating vibratory signals. If there are 4 pts
        within an oscillation (2 pts in a single peak), the error in the peak
        integrated value is under 5%. If there are 8 points in an oscillation
        (4 pts in a single peak), the error in the peak integrated value is under
        0.5%."""
        pts_to_acc = {
            4: 5,
            8: 0.5,
        }
        for pts_per_cycle, accuracy in pts_to_acc.items():
            int_t = np.arange(0, self.t_max, 1 / pts_per_cycle)
            theory_res = self.int_y(int_t)
            theory_range = max(theory_res) - min(theory_res)
            res = cumulative_integral(
                self.y(int_t), int_t, initial=0, method="simpson 1/3",
            )
            mean_error_pct = np.mean(np.abs(res - theory_res) / theory_range) * 100
            print(mean_error_pct)
            self.assertTrue(mean_error_pct < accuracy)

    def test_integrals_simps_cubic(self):
        """The cumulative 3/8 composite simpson's rule integration function provides
        good accuracy for integrating vibratory signals. If there are 4 pts
        within an oscillation (2 pts in a single peak), the error in the peak
        integrated value is under 5%. If there are 8 points in an oscillation
        (4 pts in a single peak), the error in the peak integrated value is under
        0.5%."""
        pts_to_acc = {
            4: 5,
            8: 0.5,
        }
        for pts_per_cycle, accuracy in pts_to_acc.items():
            int_t = np.arange(0, self.t_max, 1 / pts_per_cycle)
            theory_res = self.int_y(int_t)
            theory_range = max(theory_res) - min(theory_res)
            res = cumulative_integral(
                self.y(int_t), int_t, initial=0, method="simpson 3/8",
            )
            mean_error_pct = np.mean(np.abs(res - theory_res) / theory_range) * 100
            print(mean_error_pct)
            self.assertTrue(mean_error_pct < accuracy)

    def test_initial(self):
        """The integration methods allow specification of an initial value that is the
        first point in the output integration array."""
        int_t = np.arange(0, self.t_max, 1 / 500)
        initial = 10
        theory_res = self.int_y(int_t) + initial
        for method in integration_methods.keys():
            res = cumulative_integral(
                self.y(int_t), int_t, initial=initial, method=method
            )
            self.assertEqual(res[0], initial)
            self.assertAlmostEqual(max(res), max(theory_res), places=3)

    def test_dx(self):
        """The x-values are optional in the integration funcitons. The values
        default to a range of x-values spaced in equal intervals of 1. The x-values
        can also be specified with the `dx` parameter. This defines a set of x-values
        that are equally-spaced by `dx`."""
        dt = 1 / 10
        int_t = np.arange(0, self.t_max, dt)
        linear_y = int_t
        int_y = 1 / 2 * linear_y ** 2
        res = cumulative_integral(linear_y, method="trapezoidal")
        self.assertEqual(res[-1], int_y[-1] / dt)
        res = cumulative_integral(linear_y, dx=2, method="trapezoidal")
        self.assertEqual(res[-1], 2 * int_y[-1] / dt)

    def test_len_y(self):
        """An input error is raised if the input sequence is only a single element.
        The method is switched to the trapezoidal method if only 2 values
        are provided. The method is switched to the simpson's 1/3 method if only 3
        elements are provided with the simpson's 3/8 method."""
        with self.assertRaises(ValueError):
            res = cumulative_integral([1])

        res_trapz = cumulative_integral([1, 2], method="trapezoidal")
        res_simps = cumulative_integral([1, 2], method="simpson 1/3")
        self.assertTrue(all(res_trapz == res_simps))
        self.assertEqual(res_trapz[-1], 1.5)

        res_simps3 = cumulative_integral([1, 2, 3], method="simpson 1/3")
        res_simps8 = cumulative_integral([1, 2, 3], method="simpson 3/8")
        self.assertTrue(all(res_simps3 == res_simps8))

    def test_invalid_xy(self):
        """y and x (if provided) must have the same number of elements.
            Otherwise an exception is raised."""
        with self.assertRaises(ValueError):
            cumulative_integral([1, 2, 3], [1, 2])

    def test_unequally_spaced_x(self):
        """If the x-values are not equally spaced, the integration method
        is automatically switched to the trapezoidal method."""
        res_trapz = cumulative_integral([1, 2, 3], [1, 5, 6], method="trapezoidal")
        res_simps3 = cumulative_integral([1, 2, 3], [1, 5, 6], method="simpson 1/3")
        res_simps8 = cumulative_integral([1, 2, 3], [1, 5, 6], method="simpson 3/8")
        self.assertTrue(all(res_trapz == res_simps3))
        self.assertTrue(all(res_trapz == res_simps8))
