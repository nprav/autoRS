"""
07-27-2021
pnidamaluri
Acceleration response spectra generator exe scratch file
Aim: Make an exe file that:
  - searches for all .ahl or .csv files in a specified folder
      - assumes that .csv files are in LS-DYNA single x axis output format
      - .csv files are assumed to have just one time column at the beginning
      - .csv files have 1 header line, and can have multiple subsequent
          columns
      - RS will be generated for all the columns after the first time column
  - for each file, extracts the acceleration time histories
  - generates acceleration response spectra for each time history
    per specified settings
  - writes acceleration response spectra in .csv files
  - uses input settings (including specified folder) from a text file
"""

# %% Import required libraries
# Standard library imports
import os
import re
import sys
import traceback
from typing import Tuple, Callable, Dict, List

# Third party imports
import numpy as np

# Local application imports
from autoRS.resp_spect import step_resp_spect, fft_resp_spect
from autoRS.rw import read_shk_ahl

# %% Define any global/default variables
SETTINGS_FNAME: str = "RS_settings.txt"
DATE: str = "July 26 2021"
ALLOWED_SETTING_KEYS: Tuple[str, ...] = ("folder", "zeta", "ext", "method")
AVAILABLE_METHODS: Tuple[str, ...] = ("shake", "fft")
rs_function: Callable = step_resp_spect
DEFAULT_SETTINGS = {
    "folder": ".",
    "zeta": 0.05,
    "ext": False,
    "method": "fft",
}
settings = DEFAULT_SETTINGS.copy()


# %% Define required functions

# Update the RS settings
def get_settings(fname: str = SETTINGS_FNAME) -> None:
    global settings, rs_function

    with open(fname, "r") as file:
        raw_strings = [line for line in file]
        regex = "^(?P<key>.+)= *(?P<value>.*)$"
        raw_settings = {}
        for string in raw_strings:
            match = re.search(regex, string)
            if match:
                key, value = match.groups()
                key = key.strip()
                if key in ALLOWED_SETTING_KEYS:
                    raw_settings[key] = value

    settings = process_settings(raw_settings)

    if settings["method"] == "fft":
        rs_function = fft_resp_spect
    else:
        rs_function = step_resp_spect


# Convert raw settings to acceptable settings
def process_settings(raw_settings: Dict[str, str]) -> Dict[str, str]:
    clean_settings = DEFAULT_SETTINGS.copy()
    clean_settings.update(raw_settings)

    # Clean critical damping ratio
    clean_settings["zeta"] = float(clean_settings["zeta"])
    if not (0 < clean_settings["zeta"] < 1):
        clean_settings["zeta"] = DEFAULT_SETTINGS["zeta"]

    # Clean ext
    if clean_settings["ext"] == "y":
        clean_settings["ext"] = True
    elif clean_settings["ext"] == "n":
        clean_settings["ext"] = False
    else:
        clean_settings["ext"] = DEFAULT_SETTINGS["ext"]

    # clean method
    if clean_settings["method"] not in AVAILABLE_METHODS:
        clean_settings["method"] = DEFAULT_SETTINGS["method"]

    return clean_settings


# Get list of csv/ ahl files
def get_TH_file_list(path: str) -> List[str]:
    regex = r"\.(ahl)|(csv)$"
    files = list(
        filter(
            lambda file: re.search(regex, file),
            os.listdir(path),
        )
    )
    return files


# Standard header details for Response Spectra output files
def get_output_header_string() -> str:
    string = (
            "RS Settings:\n"
            + "zeta = ,{}\n".format(settings["zeta"])
            + "ext = ,{}\n".format(settings["ext"])
            + "method = ,{}\n".format(settings["method"])
            + "Note: Acceleration units will match the input TH.\n\n"
    )
    return string


# Generate RS from ahl or csv files, and save
def generate_rs_from_ahl(th_path: str, rs_path: str) -> None:
    acc, dt = read_shk_ahl(th_path, get_dt=True)
    time = np.arange(0, dt * len(acc), dt)
    rs, frq = rs_function(
        acc,
        time,
        zeta=settings["zeta"],
        ext=settings["ext"],
    )

    df = np.vstack((frq, rs)).T
    with open(rs_path, "w", newline="") as file:
        file.write(os.path.split(rs_path)[-1])
        file.write("\n" + get_output_header_string())
        np.savetxt(
            file,
            df,
            fmt="%.5f",
            delimiter=",",
            comments="",
            header=",".join(["Frequency (Hz)", "S_a"]),
        )


def generate_rs_from_csv(th_path: str, rs_path: str) -> None:
    df_th = np.genfromtxt(
        th_path,
        delimiter=",",
        skip_header=1,
        names=True,
        deletechars=" !#$%&'()*+,-./:;<=>?[\\]^{|}~",
    )
    rs = {}
    time_col = df_th.dtype.names[0]
    acc_cols = df_th.dtype.names[1:]
    for column in acc_cols:
        print(column)
        if any(np.isnan(df_th[column])):
            print("Nan detected; column skipped.")
            continue
        rs_column = column + "_S_a"
        rs[rs_column], frq = rs_function(
            df_th[column],
            df_th[time_col],
            zeta=settings["zeta"],
            ext=settings["ext"],
        )

    df_rs = np.vstack((frq, *rs.values())).T
    with open(rs_path, "w", newline="") as file:
        file.write(os.path.split(rs_path)[-1])
        file.write("\n" + get_output_header_string())
        np.savetxt(
            file,
            df_rs,
            fmt="%.5f",
            delimiter=",",
            comments="",
            header=",".join(["Frequency (Hz)", *rs.keys()]),
        )


# Write settings file
def write_default_settings(fname=SETTINGS_FNAME) -> None:
    with open(fname, "x") as file:
        file.write("Response Spectrum Generator settings file\n")
        file.write("\n")
        file.write("Folder with input time " "histories in .ahl or .csv format:\n")
        file.write("folder = {}\n".format(DEFAULT_SETTINGS["folder"]))
        file.write("\n")
        file.write("Critical damping ratio:\n")
        file.write("zeta = {}\n".format(DEFAULT_SETTINGS["zeta"]))
        file.write("\n")
        file.write("Generate RS up to 1000Hz (y) " "or just 100Hz (n)?\n")
        file.write("ext = {}\n".format("y" if DEFAULT_SETTINGS["ext"] else "n"))
        file.write("\n")
        file.write("Choose RS generation method (fft, or shake):\n")
        file.write("method = {}".format(DEFAULT_SETTINGS["method"]))


# Prepare RS folder
def make_RS_folder(path: str) -> str:
    rs_dir = os.path.join(os.path.join(path, "RS"))
    try:
        os.mkdir(rs_dir)
    except FileExistsError:
        pass
    finally:
        return rs_dir


# Get paths of all THs and RSs
def get_data_paths(th_folder: str) -> [str, str]:
    rs_folder = make_RS_folder(th_folder)
    th_fnames = get_TH_file_list(th_folder)
    th_paths = [os.path.join(th_folder, fname) for fname in th_fnames]
    rs_paths = [os.path.join(rs_folder, fname[:-4] + "_RS.csv") for fname in th_fnames]
    return th_paths, rs_paths


#  Function with overall logic
def generate_rs() -> None:
    print("AutoRS", f"{DATE}\n", sep="\n")
    if SETTINGS_FNAME not in os.listdir("."):
        write_default_settings()
        print("Settings file not detected. Rerun to " "use default settings.")
        return

    get_settings()
    print("Detected settings:")
    for key, value in settings.items():
        print("{} = {}".format(key, value))
    print("")

    th_paths, rs_paths = get_data_paths(settings["folder"])

    for th_path, rs_path in zip(th_paths, rs_paths):
        print(os.path.split(th_path)[-1])
        if th_path[-3:] == "ahl":
            generate_rs_from_ahl(th_path, rs_path)
            print("")
        elif th_path[-3:] == "csv":
            generate_rs_from_csv(th_path, rs_path)
            print("")
        else:
            continue
    print("RS Generation complete.")


# Main function to run upon opening module or exe file
def main() -> None:
    try:
        generate_rs()
    except BaseException:
        print("Error encountered.")
        print(sys.exc_info()[0])
        print(traceback.format_exc())
    finally:
        print("\nPress enter to exit.")
        input()


# %% Main body
if __name__ == "__main__":
    main()
