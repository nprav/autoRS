.. autoRS documentation master file, created by
   sphinx-quickstart on Thu Aug 12 21:17:03 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to autoRS's documentation!
==================================

autoRs is a simple Response Spectrum (RS) generator application (.exe) written in
Python. It performs the following:

* Searches for all .ahl or .csv files in a target folder.
* Reads input settings from a text file called 'RS_settings.txt'.
* Generates acceleration RS for all acceleration time histories in all valid files
  in the folder.
* Saves the RS in .csv files in a new 'RS' folder within the target folder.

The autoRS repository includes the Python module for the underlying behaviour. The
`Pyinstaller` library is used to generate a .exe file from the module.

.. note::
   .csv files are assumed to have one time column at the beginning, followed by one or
   more acceleration columns. The .csv files are also assumed to have 2 header lines at
   the top of the file. The second header line is assumed to designate the title of
   each column in the data.

**New capabilities to be added:**

* Generation of velocity and displacement response spectra.
* Additional settings including:
   * Specific frequencies to be included in the RS.
   * Input the number of header lines in input .csv files.
   * Additional input file types.
   * Generation of RS plots.


Contents
--------

.. automodule:: autoRS
   :members:
   :undoc-members:

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
