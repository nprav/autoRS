"""Unit tests for autoRS.core and autoRS.timehistory."""

# Standard library imports
import unittest

# Third party imports
import numpy as np

# Local Application Imports
from context import autoRS
from autoRS.core import FloatTable


class TestTable(unittest.TestCase):
    def test_init_list(self):
        """Table can be defined with data as a list of lists or a single list. Data
        should be homogenous and numeric."""
        t1 = FloatTable(raw_data=[[1, 2, 3], [3, 4, 5]], column_names=("a", "b", "c"))
        print(t1["b"])
        self.assertTrue(all(t1["b"] == np.array([2, 4])))
        print(t1.index)
        self.assertTrue(all(t1.index == np.array([0, 1])))

        t1 = FloatTable(
            raw_data=np.asarray([[1, 2, 3], [3, 4, 5]]), column_names=("a", "b", "c")
        )
        print(t1["b"])
        self.assertTrue(all(t1["b"] == np.array([2, 4])))
        self.assertTrue(t1.column_names == ("a", "b", "c"))

        t1 = FloatTable(raw_data=[1, 2, 3], column_names=["a"])
        print(t1["a"])
        self.assertTrue(all(t1["a"] == np.array([1, 2, 3])))
        self.assertTrue(t1.column_names == ("a",))

    def test_init_dict(self):
        """Table can be defined with data as a dictionary."""
        t2 = FloatTable(raw_data={"a": [1, 3], "b": [2, 4], "c": [3, 5]})
        print(t2["a"])
        self.assertTrue(all(t2["a"] == np.array([1, 3])))
        self.assertTrue(t2.column_names == ("a", "b", "c"))

        t1 = FloatTable(raw_data={"a": [1, 2, 3]})
        print(t1["a"])
        self.assertTrue(all(t1["a"] == np.array([1, 2, 3])))
        self.assertTrue(t1.column_names == ("a",))

    def test_invalid_data(self):
        """Inputs to the Table class must be homogenous, numeric, and all column_names must
        have the same number of rows/records."""
        with self.assertRaises(ValueError):
            t3 = FloatTable(raw_data=[[1, 2, 3], [1, 2]])
        with self.assertRaises(ValueError):
            t3 = FloatTable(raw_data={"a": [1, 3], "b": [2, 4], "c": []})
        with self.assertRaises(ValueError):
            t3 = FloatTable(raw_data={"a": ["a", 3], "b": [2, 4]})
        with self.assertRaises(ValueError):
            t3 = FloatTable(raw_data={"a": ["a", 3], "b": [2, 4]})

    def test_index_input(self):
        """Tables contain an index that can be specified. The name of the index
        can be modified. The index can also be extracted by using the index name
        as a key."""
        t = FloatTable(
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
        t = FloatTable()
        t.data = [[1, 2, 3], [3, 4, 5]]
        t.column_names = ["a", "b", "c"]
        self.assertTrue(all(t["a"] == np.array([1, 3])))

        t = FloatTable()
        t.column_names = ["a", "b", "c"]
        t["a"] = [1, 3]
        self.assertTrue(all(t["a"] == np.array([1, 3])))
        print(t)

        t = FloatTable(index=[0, 10], index_name="Row")
        self.assertTrue(t.shape == (2, 0))

        t = FloatTable(column_names=["a", "b"])
        self.assertTrue(t.shape == (0, 2))
        print(t.data)

    def test_set_index_or_data_post_init(self):
        """Data, Index, or column names in Table can be defined after instantiation.
        The new set values must match the shape of any already defined attributes."""
        t = FloatTable()
        t.data = [[1, 2, 3], [3, 4, 5]]
        t.column_names = ["a", "b", "c"]
        self.assertTrue(all(t.index == np.array([0, 1])))

        t = FloatTable(index=[0, 10], index_name="Row")
        t.data = [[1, 2, 3], [3, 4, 5]]
        self.assertTrue(all(t.data[0] == np.array([1, 2, 3])))

        t = FloatTable(index=[0, 10], column_names=["Column1"])
        self.assertTrue(t.shape == (2, 1))
        t.data = [1, 2]
        self.assertTrue(all(t["Column1"] == np.array([1, 2])))

        t = FloatTable(index=[0, 10])
        self.assertTrue(t.shape == (2, 0))
        t.column_names = ["a", "b"]
        self.assertTrue(t.shape == (2, 2))
        print(t.data)

        t = FloatTable(column_names=["Col1"])
        self.assertTrue(t.shape == (0, 1))
        t.data = [1, 2, 3]
        self.assertTrue(t.shape == (3, 1))
        self.assertTrue(all(t["Col1"] == np.array([1, 2, 3])))

        with self.assertRaises(ValueError):
            t = FloatTable()
            t.data = [[1, 2, 3], [3, 4, 5]]
            t.index = [1, 2, 3]

        with self.assertRaises(ValueError):
            t = FloatTable(column_names=["Col1"])
            t.data = [[1, 2]]

    def test_del_column(self):
        """Table columns can be deleted after instantiation. Indexes cannot be
        deleted."""
        t = FloatTable()
        t.data = [[1, 2, 3], [3, 4, 5]]
        t.column_names = ["a", "b", "c"]
        del t["a"]
        self.assertTrue(t.column_names == ("b", "c"))
        self.assertTrue((np.array([[2, 3], [4, 5]]) == t.data).all())

        with self.assertRaises(ValueError):
            del t["Index"]

    def test_reset_index(self):
        """Table indices cannot be deleted, but they can be reset."""
        t = FloatTable(index=[0, 10], index_name="Row")
        t.reset_index()
        self.assertTrue(t.index is None)

    def test_key_error(self):
        """Tables will raise KeyErrors if columns that do not exist are called."""
        t = FloatTable()
        t.data = [[1, 2, 3], [3, 4, 5]]
        t.column_names = ["a", "b", "c"]
        with self.assertRaises(KeyError):
            print(t["d"])
        with self.assertRaises(KeyError):
            del t["d"]

    def test_set_column(self):
        """Table columns can be reset or new columns can be added.
        New columns must adhere to any shape if data has already been defined"""
        t = FloatTable()
        t.data = [[1, 2, 3], [3, 4, 5]]
        t.column_names = ["a", "b", "c"]
        t["c"] = [5, 5]
        self.assertTrue(all(t["c"] == np.array([5, 5])))

        with self.assertRaises(ValueError):
            t["c"] = [5, 5, 5]
        with self.assertRaises(ValueError):
            t["c"] = [5, 5, [5]]

        t["d"] = [6, 6]
        self.assertTrue((t.shape == (2, 4)))
        self.assertTrue(all(t["d"] == np.array([6, 6])))

        t2 = FloatTable()
        t2["a"] = [10, 10]
        self.assertTrue(t2.shape == (2, 1))

    def test_iter(self):
        """Table is a subclass of MutableMapping. It can be iterated over with Python
        dictionary-like methods such as .items(), .keys(), .values()."""
        t = FloatTable()
        t.data = [[1, 2, 3], [3, 4, 5]]
        t.column_names = ["a", "b", "c"]
        raw_data = {"a": [1, 3], "b": [2, 4], "c": [3, 5]}
        for k, v in t.items():
            self.assertTrue(all(np.array(raw_data[k]) == v))

    def test_pop(self):
        """Since Table is a MutableMapping it can also pop."""
        t = FloatTable(raw_data={"a": [1, 3], "b": [2, 4], "c": [3, 5]})
        self.assertTrue(all(t.pop("b") == np.array([2, 4])))
        key, value = t.popitem()
        self.assertTrue(all(value == np.array([3, 5])))
        self.assertTrue(key == "c")

    def test_str(self):
        """The Table can be converted to a string and printed in a minimalist
        form with indexes and column names showing."""
        pass

    def test_TableLoc(self):
        """The Table values can be accessed by indexing using the
        index and column values. This is similar to Pandas
        DataFrame.loc[...]"""
        pass
