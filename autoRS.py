# 06-10-2020
# pnidamaluri
# Acceleration response spectra generator exe scratch file
# Aim: Make an exe file that:
#   - searches for all .ahl or .csv files in a specified folder
#       - assumes that .csv files are in LS-DYNA single x axis output format
#       - .csv files are assumed to have just one time column at the beginning
#       - .csv files have 1 header line, and can have multiple subsequent columns
#       - RS will be generated for all the columns after the first time column
#   - for each file, extracts the acceleration time histories
#   - generates acceleration response spectra for each time history per specified settings
#   - writes acceleration response spectra in .csv files
#   - uses input settings (including specified folder) from a text file

# %% Import required libraries
import numpy as np
import pandas as pd
from structpy.resp_spect import step_resp_spect, fft_resp_spect
from structpy.rw import read_shk_ahl
import os
import re

# %% Define any global/default variables
settings_fname = "RS_settings.txt"
allowed_setting_keys = ('folder', 'zeta', 'ext', 'method')
available_methods = ('shake', 'fft')
rs_function = step_resp_spect
default_settings = {
    'folder': '.',
    'zeta': 0.05,
    'ext': True,
    'method': 'shake',
}
settings = default_settings.copy()


# %% Define required functions

# Update the RS settings
def get_settings(fname=settings_fname):
    global settings, rs_function

    with open(fname, 'r') as file:
        raw_strings = [line for line in file]
        regex = '^(?P<key>.+)= *(?P<value>.*)$'
        raw_settings = {}
        for string in raw_strings:
            match = re.search(regex, string)
            if match:
                key, value = match.groups()
                key = key.strip()
                if key in allowed_setting_keys:
                    raw_settings[key] = value

    settings = process_settings(raw_settings)

    if settings['method'] == 'fft':
        rs_function = fft_resp_spect
    else:
        rs_function = step_resp_spect


# Convert raw settings to acceptable settings
def process_settings(raw_settings):
    clean_settings = default_settings.copy()
    clean_settings.update(raw_settings)

    # Clean critical damping ratio
    clean_settings['zeta'] = float(clean_settings['zeta'])
    if not (0 < clean_settings['zeta'] < 1):
        clean_settings['zeta'] = default_settings['zeta']

    # Clean ext
    if clean_settings['ext'] == 'y':
        clean_settings['ext'] = True
    elif clean_settings['ext'] == 'n':
        clean_settings['ext'] = False
    else:
        clean_settings['ext'] = default_settings['ext']

    # clean method
    if clean_settings['method'] not in available_methods:
        clean_settings['method'] = default_settings['method']

    return clean_settings


# Get list of csv/ ahl files
def get_TH_file_list(path):
    regex = r"\.(ahl)|(csv)$"
    files = list(filter(
        lambda file: re.search(regex, file),
        os.listdir(path),
    ))
    return files


# Standard header details for Response Spectra output files
def get_output_header_string():
    string = "RS Settings:\n" + \
             "zeta = ,{}\n".format(settings['zeta']) + \
             "ext = ,{}\n".format(settings['ext']) + \
             "method = ,{}\n".format(settings['method']) + \
             "Note: Acceleration units will match the input TH.\n\n"
    return string


# Generate RS from ahl or csv files, and save
def generate_rs_from_ahl(th_path, rs_path):
    acc, dt = read_shk_ahl(th_path, get_dt=True)
    time = np.arange(0, dt * len(acc), dt)
    rs, frq = rs_function(
        acc, time, zeta=settings['zeta'],
        ext=settings['ext'],
    )

    df = pd.DataFrame(data=rs, index=frq,
                      columns=['S_a'])

    with open(rs_path, 'w', newline='') as file:
        file.write(os.path.split(rs_path)[-1])
        file.write("\n" + get_output_header_string())
        df.to_csv(file, index_label="Frequency (Hz)",
                  float_format='%.5f')


def generate_rs_from_csv(th_path, rs_path):
    df_th = pd.read_csv(th_path, skiprows=1, index_col=0)
    df_th.dropna(axis=1, inplace=True)
    rs = {}
    for column in df_th.columns:
        rs_column = column + "_S_a"
        rs[rs_column], frq = rs_function(
            df_th[column].values, df_th.index,
            zeta=settings['zeta'],
            ext=settings['ext'],
        )
    df_rs = pd.DataFrame(data=rs, index=frq)
    with open(rs_path, 'w', newline='') as file:
        file.write(os.path.split(rs_path)[-1])
        file.write("\n" + get_output_header_string())
        df_rs.to_csv(file, index_label="Frequency (Hz)",
                     float_format='%.5f')


# Write settings file
def write_default_settings(fname=settings_fname):
    with open(fname, 'x') as file:
        file.write("Response Spectrum Generator settings file\n")
        file.write("\n")
        file.write("Folder with input time "
                   "histories in .ahl or .csv format:\n")
        file.write("folder = {}\n".format(default_settings['folder']))
        file.write("\n")
        file.write("Critical damping ratio:\n")
        file.write("zeta = {}\n".format(default_settings['zeta']))
        file.write("\n")
        file.write("Generate RS up to 1000Hz (y) "
                   "or just 100Hz (n)?\n")
        file.write("ext = {}\n".format(
            'y' if default_settings['ext'] else 'n'))
        file.write("\n")
        file.write("Choose RS generation method (fft, or shake):\n")
        file.write("method = {}".format(default_settings['method']))


# Prepare RS folder
def make_RS_folder(path):
    rs_dir = os.path.join(os.path.join(path, "RS"))
    try:
        os.mkdir(rs_dir)
    except FileExistsError:
        pass
    finally:
        return rs_dir


# Get paths of all THs and RSs
def get_data_paths(th_folder):
    rs_folder = make_RS_folder(th_folder)
    th_fnames = get_TH_file_list(th_folder)
    th_paths = [os.path.join(th_folder, fname) for
                fname in th_fnames]
    rs_paths = [os.path.join(rs_folder, fname[:-4] + "_RS.csv") for
                fname in th_fnames]
    return th_paths, rs_paths


# Main function with overall logic
def generate_rs():
    if settings_fname not in os.listdir('.'):
        write_default_settings()
        print("Settings file not detected. Rerun to "
              "use default settings.")
        return

    get_settings()

    th_paths, rs_paths = get_data_paths(settings['folder'])

    for th_path, rs_path in zip(th_paths, rs_paths):
        if th_path[-3:] == 'ahl':
            generate_rs_from_ahl(th_path, rs_path)
        elif th_path[-3:] == 'csv':
            generate_rs_from_csv(th_path, rs_path)
        else:
            continue

    print("RS Generation complete.")


# %% Main body
if __name__ == '__main__':
    generate_rs()
