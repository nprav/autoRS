"""Unit tests for autoRS.core and autoRS.timehistory."""

# Standard library imports
import unittest
import os

# Third party imports
import numpy as np

# Local Application Imports
from context import autoRS
from autoRS.core import Table


class TestTable(unittest.TestCase):
    def test_init_list(self):
        """Table can be defined with data as a list of lists or a single list. Data
        should be homogenous and numeric."""
        t1 = Table(raw_data=[[1, 2, 3], [3, 4, 5]], column_names=("a", "b", "c"))
        print(t1["b"])
        self.assertTrue(all(t1["b"] == np.array([2, 4])))
        print(t1.index)
        self.assertTrue(all(t1.index == np.array([0, 1])))

        t1 = Table(
            raw_data=np.asarray([[1, 2, 3], [3, 4, 5]]), column_names=("a", "b", "c")
        )
        print(t1["b"])
        self.assertTrue(all(t1["b"] == np.array([2, 4])))
        self.assertTrue(t1.column_names == ("a", "b", "c"))

        t1 = Table(raw_data=[1, 2, 3], column_names=["a"])
        print(t1["a"])
        self.assertTrue(all(t1["a"] == np.array([1, 2, 3])))
        self.assertTrue(t1.column_names == ("a",))

    def test_init_dict(self):
        """Table can be defined with data as a dictionary."""
        t2 = Table(raw_data={"a": [1, 3], "b": [2, 4], "c": [3, 5]})
        print(t2["a"])
        self.assertTrue(all(t2["a"] == np.array([1, 3])))
        self.assertTrue(t2.column_names == ("a", "b", "c"))

        t1 = Table(raw_data={"a": [1, 2, 3]})
        print(t1["a"])
        self.assertTrue(all(t1["a"] == np.array([1, 2, 3])))
        self.assertTrue(t1.column_names == ("a",))

    def test_invalid_data(self):
        """Inputs to the Table class must be homogenous, numeric, and all column_names must
        have the same number of rows/records."""
        with self.assertRaises(ValueError):
            t3 = Table(raw_data=[[1, 2, 3], [1, 2]])
        with self.assertRaises(ValueError):
            t3 = Table(raw_data={"a": [1, 3], "b": [2, 4], "c": []})
        with self.assertRaises(ValueError):
            t3 = Table(raw_data={"a": ["a", 3], "b": [2, 4]})
        with self.assertRaises(ValueError):
            t3 = Table(raw_data={"a": ["a", 3], "b": [2, 4]})

    def test_index_input(self):
        """Tables contain an index that can be specified. The name of the index
        can be modified. The index can also be extracted by using the index name
        as a key."""
        t = Table(
            raw_data={"a": [1, 3], "b": [2, 4], "c": [3, 5]},
            index=[0, 10],
            index_name="Row",
        )
        self.assertTrue(all(t.index == [0, 10]))
        self.assertTrue(t.index_name == "Row")
        self.assertTrue(all(t["Row"] == [0, 10]))

    def test_empty_init(self):
        """Tables can be defined as empty, with just an index, just column_names,
        or with just data."""
        t = Table()
        t.data = [[1, 2, 3], [3, 4, 5]]
        t.column_names = ["a", "b", "c"]
        self.assertTrue(all(t["a"] == np.array([1, 3])))

        t = Table(index=[0, 10], index_name="Row")
        self.assertTrue(t.shape == (2, 0))

        t = Table(column_names=["a", "b"])
        self.assertTrue(t.shape == (0, 2))
        print(t.data)

    def test_set_index_or_data_post_init(self):
        """Data, Index, or column names in Table can be defined after instantiation.
        The new set values must match the shape of any already defined attributes."""
        t = Table()
        t.data = [[1, 2, 3], [3, 4, 5]]
        t.column_names = ["a", "b", "c"]
        self.assertTrue(all(t.index == np.array([0, 1])))

        t = Table(index=[0, 10], index_name="Row")
        t.data = [[1, 2, 3], [3, 4, 5]]
        self.assertTrue(all(t.data[0] == np.array([1, 2, 3])))

        t = Table(index=[0, 10], column_names=["Column1"])
        self.assertTrue(t.shape == (2, 1))
        t.data = [1, 2]
        self.assertTrue(all(t["Column1"] == np.array([1, 2])))

        t = Table(index=[0, 10])
        self.assertTrue(t.shape == (2, 0))
        t.column_names = ["a", "b"]
        self.assertTrue(t.shape == (2, 2))
        print(t.data)

        with self.assertRaises(ValueError):
            t = Table()
            t.data = [[1, 2, 3], [3, 4, 5]]
            t.index = [1, 2, 3]
