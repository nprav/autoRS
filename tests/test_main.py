"""Unit tests for autoRS.__init__."""

# Standard library imports
import unittest
import os
import shutil
import re

# Local Application Imports
from context import autoRS
from autoRS.rw import read_csv_multi


class TestMain(unittest.TestCase):
    def setUp(self):
        with open("test_settings1.txt", "w") as file:
            file.write("folder=   C:\\Users\n" "sdf=a\n" "zeta=0.07\n" "ext=  n\n")
        with open("test_settings2.txt", "w") as file:
            file.write("  folder   =  hello\n" "method= shake")
        if not os.path.isdir(os.path.join("test_resources", "RS")):
            os.mkdir(os.path.join("test_resources", "RS"))

        shutil.copyfile(
            os.path.join("test_resources", "single_col_w_comma.csv"),
            os.path.join("single_col_w_comma.csv"),
        )

        try:
            os.remove(autoRS.SETTINGS_FNAME)
        except FileNotFoundError:
            pass

    def test_get_settings1(self):
        autoRS.get_settings("test_settings1.txt")
        print(autoRS.settings)
        self.assertEqual(autoRS.settings["folder"], r"C:\Users")
        self.assertEqual(autoRS.settings["zeta"], 0.07)
        self.assertEqual(autoRS.settings["ext"], False)
        self.assertEqual(set(autoRS.settings.keys()), set(autoRS.ALLOWED_SETTING_KEYS))

    def test_get_settings2(self):
        autoRS.get_settings("test_settings2.txt")
        self.assertEqual(autoRS.settings["folder"], "hello")
        self.assertEqual(autoRS.settings["method"], "shake")

    def test_get_settings3(self):
        autoRS.write_default_settings("test_settings3.txt")
        autoRS.get_settings("test_settings3.txt")
        self.assertEqual(autoRS.DEFAULT_SETTINGS, autoRS.settings)

    def test_TH_file_list(self):
        files = autoRS.get_TH_file_list("test_resources")
        self.assertEqual(
            set(files),
            {
                "single_col_w_comma.csv",
                "shake_acc_eg.ahl",
                "multi_col.csv",
                "single_col_wo_comma.csv",
                "nan_eg.csv",
            },
        )

    def test_rs_from_ahl(self):
        th_path = os.path.join("test_resources", "shake_acc_eg.ahl",)
        rs_path = os.path.join("test_resources", "RS", "test.csv",)
        autoRS.generate_rs_from_ahl(th_path, rs_path)
        self.assertTrue(os.path.isfile(rs_path))

    def test_rs_from_csv(self):
        th_path = os.path.join("test_resources", "multi_col.csv",)
        rs_path = os.path.join("test_resources", "RS", "test2.csv",)
        autoRS.generate_rs_from_csv(th_path, rs_path)
        th_list = read_csv_multi(th_path, header=2)
        rs_list = read_csv_multi(rs_path, header=8)
        self.assertEqual(len(th_list[1]), len(rs_list[1]))

        th_path = os.path.join("test_resources", "single_col_w_comma.csv",)
        rs_path = os.path.join("test_resources", "RS", "test3.csv",)
        autoRS.generate_rs_from_csv(th_path, rs_path)

        th_path = os.path.join("test_resources", "single_col_wo_comma.csv",)
        rs_path = os.path.join("test_resources", "RS", "test4.csv",)
        autoRS.generate_rs_from_csv(th_path, rs_path)
        self.assertTrue(os.path.isfile(rs_path))

        th_path = os.path.join("test_resources", "nan_eg.csv",)
        rs_path = os.path.join("test_resources", "RS", "test5.csv",)
        autoRS.generate_rs_from_csv(th_path, rs_path)
        # All columns in `nan_eg.csv` are invalid. No file should
        # be generated.
        self.assertFalse(os.path.isfile(rs_path))

    def test_make_RS_folder(self):
        rs_dir = autoRS.make_RS_folder("test_resources")
        self.assertTrue(os.path.isdir(os.path.join("test_resources", "RS")))
        self.assertEqual(rs_dir, os.path.join("test_resources", "RS"))

    def test_get_data_paths(self):
        th_paths, rs_paths = autoRS.get_data_paths("test_resources")
        th_fnames = [os.path.split(x)[-1] for x in th_paths]
        rs_fnames = [os.path.split(x)[-1] for x in rs_paths]
        check = [
            (re.split(x[:-4], y)[-1] == "_RS.csv") for x, y in zip(th_fnames, rs_fnames)
        ]
        self.assertTrue(all(check))

    def test_generate_rs(self):
        autoRS.generate_rs()
        self.assertFalse(os.path.isdir(os.path.join("RS")))
        autoRS.generate_rs()
        self.assertTrue(len(os.listdir(os.path.join("RS"))) == 1)

    def tearDown(self):
        tear_down_functions = [
            lambda: os.remove("test_settings1.txt"),
            lambda: os.remove("test_settings2.txt"),
            lambda: os.remove("test_settings3.txt"),
            lambda: os.remove("single_col_w_comma.csv"),
            lambda: shutil.rmtree(os.path.join("test_resources", "RS")),
            lambda: shutil.rmtree(os.path.join("RS")),
        ]
        for function in tear_down_functions:
            try:
                function()
            except FileNotFoundError:
                continue


if __name__ == "__main__":
    unittest.main()
