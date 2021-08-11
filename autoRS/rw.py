"""
Created: Nov 2017
Latest update:  Aug 2019
@author: Praveer Nidamaluri

Module that provides file read/write functions for
shake/dmod/peer inputs and outputs.
"""

# %% Import required modules
from typing import List, Tuple
import re


# %% Read functions


def read_shk_ahl(filename: str, header: int = 3) -> Tuple[List[float], float]:
    """Read SHAKE .ahl time history output files.

    Parameters
    ----------
    filename : str
        Address of .ahl file with filename.

    header : int, optional
        Number of lines to skip at the start of the file.
        Should be an integer greater than 0. Defaults to 3.

    Returns
    -------
    ahl : List[float]
        List of acceleration values from the .ahl file.

    dt : float
        Timestep value
    """

    with open(filename) as file:
        ahl = []
        file.seek(0, 0)
        s = False
        for i, line in enumerate(file):
            if i == 1:
                dt = float(line.split()[2])
            if i >= header:
                s = True
            if s:
                ahl += [float(k) for k in line.split()]

    return ahl, dt


def read_dmd_acc(filename: str) -> Tuple[List[float], List[float], float]:
    """Read surface and base time histories from D-MOD .acc file.

    Parameters
    ----------
    filename : str
        Address of .acc file with filename.

    Returns
    -------
    acc_surf : List[float]
        List of acceleration values of the surface (1st layer) time history.

    acc_base : List[float]
        List of acceleration values of the base (last layer) time history.

    tstep : float
        Timestep value
    """

    file = open(filename, "r")
    acc_surf = []
    acc_base = []
    file.seek(0, 0)
    s = False
    tstep = 100
    for i, line in enumerate(file):
        if not s and i == 5:
            tstep = line.split()[-1]
        if not s and i > 5:
            if line.split()[0] == tstep:
                s = True
        if s:
            if line.split()[0] == "0":
                file.close()
                break
            acc_surf.append(float(line.split()[1]))
            acc_base.append(float(line.split()[-1]))

    return acc_surf, acc_base, tstep


def read_fort_txt(filename, header=8, cols=8, dgts=9) -> List[float]:
    """Extract time histories from text files in fortran format.

    Defaults to '8F9.6' format.

    Parameters
    ----------
    filename : str
        Address of file with filename and extension.

    header : int, optional
        Number of lines to skip at the start of the file.
        Should be an integer greater than 0. Defaults to 8.

    cols : int, optional
        Number of columns in fortran format.
        Should be an integer greater than 0. Defaults to 8.

    dgts : int, optional
        Number of digits per value in fortran format.
        Should be an integer greater than 0. Defaults to 9.

    Returns
    -------
    acc : List[float]
        List of values from the file.
    """

    file = open(filename)
    acc = []
    file.seek(0, 0)
    s = False
    for i, line in enumerate(file):
        if i >= header:
            s = True
        if s:
            while line != "\n":
                acc.append(float(line[:dgts]))
                line = line[dgts:]

    file.close()

    return acc


def read_csv_multi(filename: str, header: int = 1) -> [List[float], List[List[float]]]:
    """Reads .csv files with multiple columns.

    Parameters
    ----------
    filename : str
        Address of .csv file with filename.

    header : int, optional
        Number of lines to skip at the start of the file.
        Should be an integer greater than 0. Defaults to 2.

    Returns
    -------
    tm : 1D List[float]
        List of abscissa values from the file.

    acc : List[List[float]]
        2D List of ordinate values from the file. Values from each
        abscissa column of the csv file are contained in a separate list
        within `acc`.
    """

    file = open(filename, "r")

    file.seek(0, 0)
    s = False
    for i, line in enumerate(file):
        if i >= header:
            s = True
        if s:
            num_cols = len(re.split("[,\n]+", line)) - 2
            break

    acc = []
    for i in range(0, num_cols):
        acc.append([])

    tm = []
    file.seek(0, 0)
    s = False
    for i, line in enumerate(file):
        if i >= header:
            s = True
        if s:
            tm.append(float(line.split(",")[0]))
            for j in range(0, num_cols):
                acc[j].append(float(line.split(",")[j + 1]))

    file.close()

    return tm, acc


def read_peer_record(filename: str) -> dict:
    """Read time history file from PEER Strong Motion Record Database.

    Parameters
    ----------
    filename : str
        Address of file with filename and extension.

    Returns
    -------
    eq_record : dict
        Dictionary with the Peer TH properties.
    """

    eq_record = {}

    with open(filename, "r") as file:
        # Skip the first header line
        file.readline()

        line2 = file.readline()
        props2 = line2.split(",")
        eq_record.update(dict(zip(["name", "date", "array", "direction"], props2)))
        # remove '\n' from end of direction string
        eq_record["direction"] = eq_record["direction"].split()[0]

        line3 = file.readline()
        props3 = line3.split()
        eq_record["type"] = props3[0].lower()
        eq_record["units"] = props3[-1].lower()

        line4 = file.readline()
        eq_record["npts"] = int(re.findall(r"(?<=NPTS=)[\d ]+(?=,)", line4)[0])
        eq_record["dt"] = float(re.findall(r"(?<=DT=).+(?=SEC)", line4)[0])

        # Read in time history
        th = []
        for line in file:
            th += list(map(float, line.split()))
        eq_record["th"] = th

    return eq_record
