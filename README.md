# autoRS
 Response spectra generator application (.exe file)
 
06-10-2020
pnidamaluri


## Acceleration response spectra generator exe
Aim: Make an exe file that:
  - searches for all .ahl or .csv files in a specified folder
      - assumes that .csv files are in LS-DYNA single x-axis output format
      - .csv files are assumed to have just one time column at the beginning
      - .csv files have 1 header line, and can have multiple subsequent columns
      - RS will be generated for all the columns after the first time column
  - for each file, extracts the acceleration time histories
  - generates acceleration response spectra for each time history per specified settings
  - writes acceleration response spectra in .csv files
  - uses input settings (including specified folder) from a text file
