# autoRS
autoRs is a simple Response Spectrum (RS) generator application (.exe) written in
Python. It performs the following:

* Searches for all .ahl or .csv files in a target folder.
* Reads input settings from a text file called 'RS_settings.txt'.
* Generates acceleration RS for all acceleration time histories in all valid files
  in the folder.
* Saves the RS in .csv files in a new 'RS' folder within the target folder.

The autoRS repository includes the Python module for the underlying behaviour. The
`Pyinstaller` library is used to generate a .exe file from the module.

### New capabilities to be added
* Generation of velocity and displacement response spectra.
* Additional settings including:
    * Specific frequencies to be included in the RS.
    * Input the number of header lines in input .csv files.
    * Additional input file types.
    * Generation of RS plots.

