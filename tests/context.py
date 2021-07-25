"""
File is used to import the autoRS package so it can be tested by the test
suite python files.
"""

import os
import sys
sys.path.insert(
    0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')   # '..' implies the
    )                                                   # parent folder
)

import autoRS
