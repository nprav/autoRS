"""Main controller and entry point for the autoRS package."""

# %% Import required libraries
# Standard library imports
import os
import re
import sys
import traceback
from typing import Tuple, Dict, List

# Third party imports
import numpy as np

# Local application imports
from autoRS.spectrum import response_spectrum, RS_METHODS, DEFAULT_METHOD
from autoRS.rw import read_shk_ahl

# %% Define any global/default variables
SETTINGS_FNAME: str = "RS_settings.txt"
DATE: str = "July 26 2021"
ALLOWED_SETTING_KEYS: Tuple[str, ...] = ("folder", "zeta", "ext", "method")
AVAILABLE_METHODS: Tuple[str, ...] = tuple(RS_METHODS)
DEFAULT_SETTINGS = {
    "folder": ".",
    "zeta": 0.05,
    "ext": False,
    "method": DEFAULT_METHOD,
}
settings = DEFAULT_SETTINGS.copy()


# %% Define required functions

# Update the RS settings
def get_settings(fname: str = SETTINGS_FNAME) -> None:
    """Read the settings file (assuming it exists) and set up the settings
    dictionary."""
    global settings

    # Parse the settings file and feed all valid key-value pairs into a raw dictionary
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


# Convert raw settings to acceptable settings
def process_settings(raw_settings: Dict[str, str]) -> Dict[str, str]:
    """Given a raw settings dictionary, update the program settings. Use default values
    in case of invalid raw setting definitions."""
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


def get_TH_file_list(path: str) -> List[str]:
    """Get the list of .csv and .ahl time history files to process."""
    regex = r"\.(ahl)|(csv)$"
    files = list(filter(lambda file: re.search(regex, file), os.listdir(path),))
    return files


def get_output_header_string() -> str:
    """Generate header string for output Response Spectra output files."""
    string = (
        "RS Settings:\n"
        + "zeta = ,{}\n".format(settings["zeta"])
        + "ext = ,{}\n".format(settings["ext"])
        + "method = ,{}\n".format(settings["method"])
        + "Note: Acceleration units will match the input TH.\n\n"
    )
    return string


# Generate RS from all valid files (currently just .ahl and .csv) and save
# TODO: Add additional file extensions (eg. .ot2, peer record, etc.)


def generate_rs_from_ahl(th_path: str, rs_path: str) -> None:
    """Read .ahl time history from `th_path`. Generate the RS. Write to `rs_path`."""
    acc, dt = read_shk_ahl(th_path)
    time = np.arange(0, dt * len(acc), dt)
    rs, frq = response_spectrum(
        acc, time, zeta=settings["zeta"], high_frequency=settings["ext"],
    )

    # Reformat `rs`, `frq` arrays as a combined array for use in np.savetxt.
    df = np.vstack((frq, rs)).T

    # Open file and write informative header lines + RS data
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
    """Read .csv time history(s) from `th_path`. Generate the RS.
    Write to `rs_path`."""
    df_th = np.genfromtxt(
        th_path,
        delimiter=",",
        skip_header=1,
        names=True,
        deletechars=" !#$%&'()*+,-./:;<=>?[\\]^{|}~",
    )

    # .csv file may have multiple time history columns. Hence, define RS
    # as a dictionary and generate separately for each column.
    rs = {}
    time_col = df_th.dtype.names[0]
    acc_cols = df_th.dtype.names[1:]
    for column in acc_cols:
        print(column)
        if any(np.isnan(df_th[column])):
            print("Nan detected; column skipped.")
            continue
        rs_column = column + "_S_a"
        rs[rs_column], frq = response_spectrum(
            df_th[column],
            df_th[time_col],
            zeta=settings["zeta"],
            high_frequency=settings["ext"],
        )

    # If no valid THs and Nans detected in all cases, rs dictionary will be
    # empty. Exit from function
    if not rs:
        print("No valid columns in file. No RS generated. File skipped.")
        return

    # Reformat `rs`, `frq` arrays as a combined array for use in np.savetxt.
    df_rs = np.vstack((frq, *rs.values())).T

    # Open file and write informative header lines + RS data
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


def write_default_settings(fname=SETTINGS_FNAME) -> None:
    """Write default settings file."""
    with open(fname, "x") as file:
        file.write("Response Spectrum Generator settings file\n")
        file.write("\n")
        file.write("Folder with input time histories in .ahl or .csv format:\n")
        file.write("folder = {}\n".format(DEFAULT_SETTINGS["folder"]))
        file.write("\n")
        file.write("Critical damping ratio:\n")
        file.write("zeta = {}\n".format(DEFAULT_SETTINGS["zeta"]))
        file.write("\n")
        file.write("Generate RS up to 1000Hz (y) or just 100Hz (n)?\n")
        file.write("ext = {}\n".format("y" if DEFAULT_SETTINGS["ext"] else "n"))
        file.write("\n")
        file.write("Choose RS generation method (fft, or shake):\n")
        file.write("method = {}".format(DEFAULT_SETTINGS["method"]))


def make_RS_folder(path: str) -> str:
    """Make RS folder at given path (if it does not exist already).
    Returns the final RS directory."""
    rs_dir = os.path.join(os.path.join(path, "RS"))
    try:
        os.mkdir(rs_dir)
    except FileExistsError:
        pass
    finally:
        return rs_dir


# Get paths of all THs and RSs
def get_data_paths(th_folder: str) -> Tuple[List[str], List[str]]:
    """For a given folder path with time histories, return all time history files and
    generate prospective RS files."""

    # Make/get RS folder in the given time history directory.
    rs_folder = make_RS_folder(th_folder)

    # Get all the valid TH files from the time history folder
    th_fnames = get_TH_file_list(th_folder)

    # Create and return the absolute TH/RS paths of each file
    th_paths = [os.path.join(th_folder, fname) for fname in th_fnames]
    rs_paths = [os.path.join(rs_folder, fname[:-4] + "_RS.csv") for fname in th_fnames]
    return th_paths, rs_paths


def generate_rs() -> None:
    """Overall program logic:

    - Detects settings in the default settings file (`SETTINGS_FNAME`).
    - Generate RS for all valid TH files in the target directory listed in the settings
      file.
    """

    print("AutoRS", f"{DATE}\n", sep="\n")

    # Generate the default settings file if the settings file is not detected.
    if SETTINGS_FNAME not in os.listdir("."):
        write_default_settings()
        print("Settings file not detected. Rerun to " "use default settings.")
        return

    # Parse the settings in the settings text file. Print the detected settings.
    get_settings()
    print("Detected settings:")
    for key, value in settings.items():
        print("{} = {}".format(key, value))
    print("")

    th_paths, rs_paths = get_data_paths(settings["folder"])

    # Generate spectra for each valid time history file.
    # TODO: Convert if-else chain to dictionary as additional TH extension
    #   options are added.
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


def main() -> None:
    """Main function to run upon opening module or exe file."""
    try:
        generate_rs()
    except BaseException:
        print("Error encountered.")
        print(sys.exc_info()[0])
        print(traceback.format_exc())
    finally:
        # When using pyinstaller to generate a .exe file, the program runs in
        # a command prompt/terminal. As soon as the last command is completed, the
        # terminal closes. This makes it difficult to view/debug the printed text.
        # --> Use input() to enforce terminal to wait for user input to exit.
        print("\nPress enter to exit.")
        input()


# %% Main body
if __name__ == "__main__":
    main()
