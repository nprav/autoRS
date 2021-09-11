"""Unit tests for autoRS.core and autoRS.timehistory."""

# Standard library imports
import unittest
import os

# Third party imports
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured

# Local Application Imports
from context import autoRS
from autoRS.core import Table


class TestTable(unittest.TestCase):
    def test_init_list(self):
        """Table can be defined with data as a list of lists or a single list. Data
        should be homogenous and numeric."""
        t1 = Table(data=[[1, 2, 3], [3, 4, 5]], column_names=("a", "b", "c"))
        print(t1["b"])
        self.assertTrue(all(t1["b"] == np.array([2, 4])))

        t1 = Table(data=[[1, 2, 3], [3, 4, 5]], column_names=("a", "b", "c"))
        print(t1["b"])
        self.assertTrue(all(t1["b"] == np.array([2, 4])))
        self.assertTrue(t1.columns == ("a", "b", "c"))

        t1 = Table(data=[1, 2, 3], column_names=["a"])
        print(t1["a"])
        self.assertTrue(all(t1["a"] == np.array([1, 2, 3])))
        self.assertTrue(t1.columns == ("a",))

    def test_init_dict(self):
        """Table can be defined with data as a dictionary."""
        t2 = Table(data={"a": [1, 3], "b": [2, 4], "c": [3, 5]})
        print(t2["a"])
        self.assertTrue(all(t2["a"] == np.array([1, 3])))
        self.assertTrue(t2.columns == ("a", "b", "c"))

    def test_invalid_data(self):
        """Inputs to the Table class must be homogenous, numeric, and all columns must
        have the same number of rows/records."""
        with self.assertRaises(ValueError):
            t3 = Table(data=[[1, 2, 3], [1, 2]])
        with self.assertRaises(ValueError):
            t3 = Table(data={"a": [1, 3], "b": [2, 4], "c": []})
        with self.assertRaises(ValueError):
            t3 = Table(data={"a": ["a", 3], "b": [2, 4]})
        with self.assertRaises(ValueError):
            t3 = Table(data={"a": ["a", 3], "b": [2, 4]})

    def test_index_input(self):
        """Tables contain an index that can be specified. The name of the index
        can be modified. The index can also be extracted by using the index name
        as a key."""
        t = Table(
            data={"a": [1, 3], "b": [2, 4], "c": [3, 5]},
            index=[0, 10],
            index_name="Row",
        )
        self.assertTrue(all(t.index == [0, 10]))
        self.assertTrue(t.index_name == "Row")
        self.assertTrue(all(t["Row"] == [0, 10]))
