"""
Helper functions for loading and processing simulation data.
"""
import sys
import torch as pt
from glob import glob

from os import environ
from os.path import join
from typing import Tuple, Optional, List
from pandas import read_csv, concat, DataFrame, read_table
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, environ.get("FLOWTORCH_INSTALL_DIR"))
from flowtorch.data import CSVDataloader


def load_residuals(load_path: str, cols: Optional[List[int]] = None, names: Optional[List[str]] = None) -> DataFrame:
    """
    Load and merge solver residuals from simulation output files.

    :param load_path: Path to the top-level simulation directory containing ``postProcessing/residuals``.
    :type load_path: str
    :param cols: Column indices to extract from ``solverInfo.dat`` (default includes Ux, Uy, Uz, and p fields).
    :type cols: list[int] | None
    :param names: Column names for the resulting DataFrame (must match the number of selected columns).
    :type names: list[str] | None
    :return: DataFrame containing merged residual histories for all available time steps.
    :rtype: DataFrame
    :raises AssertionError: If ``len(names)`` does not match ``len(cols)``.
    """
    dirs = sorted(glob(join(load_path, "postProcessing", "residuals", "*")), key=lambda x: float(x.split("/")[-1]))

    if cols is None:
        # we don't care about the solver, since it's defined in the setup
        cols = [0] + list(range(2, 12)) + [13, 14, 15, 16]
    if names is None:
        names = ["time", "Ux_initial", "Ux_final", "Ux_iters", "Uy_initial", "Uy_final", "Uy_iters", "Uz_initial",
                 "Uz_final", "Uz_iters", "U_converged", "p_initial", "p_final", "p_iters", "p_converged",]
    assert len(names) == len(cols), "len(names) must be the same as len(cols)."

    _solverInfo = [read_csv(join(p, "solverInfo.dat"), sep=r"\s+", comment="#", header=None, skiprows=2, usecols=cols,
                            names=names) for p in dirs]

    # merge to single DF and remove all duplicates
    if len(_solverInfo) > 1:
        _solverInfo = concat(_solverInfo)
    _solverInfo.drop_duplicates(["time"], inplace=True)
    _solverInfo.reset_index(inplace=True, drop=True)

    return _solverInfo


def compute_friction_velocity(load_path: str, filename: str, t_start: float, nu: float, dtype: pt.dtype = pt.float64) -> tuple:
    """
    Compute the wall friction velocity (:math:`u_\\tau`) from OpenFOAM surface field data.

    :param load_path: Path to the simulation case directory containing surface field data.
    :type load_path: str
    :param filename: Name of the surface field file (e.g., ``"wallGradU"`` or similar).
    :type filename: str
    :param t_start: Minimum simulation time from which to begin processing snapshots.
    :type t_start: float
    :param nu: Kinematic viscosity of the fluid.
    :type nu: float
    :param dtype: dtype
    :type dtype: pt.dtype
    :return: Tuple containing:
        - ``x`` (Tensor): x-coordinates of surface vertices.
        - ``y`` (Tensor): y-coordinates of surface vertices.
        - ``z`` (Tensor): z-coordinates of surface vertices.
        - ``area`` (Tensor): Surface cell areas.
        - ``u_tau`` (list[Tensor]): Computed friction velocities for each time step.
        - ``grad_u`` (list[Tensor]): Velocity gradient tensors for each time step.
    :rtype: tuple
    """
    loader = CSVDataloader.from_foam_surface(load_path, filename, dtype=dtype)
    times = loader.write_times
    times = [t for t in times if float(t) >= t_start]
    vertices = loader.vertices
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    area_normal = -pt.vstack(loader.load_snapshot(["area_x", "area_y", "area_z"], times[0])).T
    area = area_normal.norm(dim=1)
    normal = area_normal / area.unsqueeze(-1)
    u_tau, grad_u = [], []
    for t in times:
        grad = pt.vstack(
            loader.load_snapshot(["gradU_xx", "gradU_xy", "gradU_xz", "gradU_yx", "gradU_yy", "gradU_yz",
                                  "gradU_zx", "gradU_zy", "gradU_zz"], t)
        ).T
        grad_u.append(grad.reshape((grad.shape[0], 3, 3)))
        shear = nu * (grad_u[-1] + grad_u[-1].transpose(2, 1))
        projection_norm = (shear @ normal.unsqueeze(-1)).squeeze().norm(dim=1)
        u_tau.append(projection_norm.sqrt())
    return x, y, z, area, u_tau, grad_u


def load_force_coefficients(load_path: str, use_cols: Optional[List[int]] = None,
                            names: Optional[List[str]] = None) -> DataFrame:
    """
    Load and merge aerodynamic force coefficients from simulation output files.

    :param load_path: Path to the top-level simulation directory containing ``postProcessing/forces``.
    :type load_path: str
    :param use_cols: Column indices to read from ``coefficient.dat`` (default: [0, 1, 4]).
    :type use_cols: list[int] | None
    :param names: Column names for the resulting DataFrame (default: ["time", "cx", "cy"]).
    :type names: list[str] | None
    :return: DataFrame with combined force coefficients for all available time steps.
    :rtype: DataFrame
    """
    if names is None:
        names = ["time", "cx", "cy"]
    if use_cols is None:
        use_cols = [0, 1, 4]
    dirs = sorted(glob(join(load_path, "postProcessing", "forces", "*")), key=lambda x: float(x.split("/")[-1]))
    coeffs = [read_csv(join(p, "coefficient.dat"), sep=r"\s+", comment="#", header=None, usecols=use_cols, names=names)
              for p in dirs]

    # merge to single DF and remove all duplicates
    if len(coeffs) > 1:
        coeffs = concat(coeffs)
    coeffs.drop_duplicates(["time"], inplace=True)
    coeffs.reset_index(inplace=True, drop=True)

    return coeffs


def load_probes(load_path: str, num_probes: int, filename: str = "p", skip_n_points: int = 0) -> DataFrame:
    """
    Reads the probe output files located in the ``postProcessing/probes`` directory of an OpenFOAM case
    and combines them into a single pandas DataFrame.

    Depending on the requested field (scalar or vector), the function
    automatically handles parsing and column naming:

    - For scalar fields (e.g., ``p``, ``p_rgh``): columns are named
      ``p_probe_0``, ``p_probe_1``, etc.
    - For vector fields (e.g., ``U``): columns are named
      ``ux_probe_0``, ``uy_probe_0``, ``uz_probe_0``, etc., and parentheses
      surrounding the vector components are removed automatically.

    :param load_path: Path to the top-level directory of the simulation case
                      (containing the ``postProcessing`` folder).
    :type load_path: str
    :param num_probes: Number of probes placed in the simulation domain.
    :type num_probes: int
    :param filename: Name of the field written in the ``probes`` directory
                     (e.g., ``'p'``, ``'p_rgh'``, or ``'U'``).
    :type filename: str
    :param skip_n_points: Number of initial time steps to skip when reading the files, e.g. a transient phase.
    :type skip_n_points: int
    :return: DataFrame containing probe data for all time steps and probes, merged into a single structure without
            duplicate time entries.
    :rtype: DataFrame
    """
    dirs = sorted(glob(join(load_path, "postProcessing", "probes", "*")), key=lambda x: float(x.split("/")[-1]))
    _probes = []

    for d in dirs:
        # skip header, header = n_probes + 2 lines containing probe no. and time header
        if filename.startswith("p"):
            names = ["time"] + [f"{filename}_probe_{pb}" for pb in range(num_probes)]
            probe = read_table(join(d, filename), sep=r"\s+", skiprows=(num_probes + 1) + skip_n_points, header=None,
                                  names=names)
        else:
            names = ["time"]
            for pb in range(num_probes):
                names += [f"{k}_probe_{pb}" for k in ["ux", "uy", "uz"]]

            probe = read_table(join(d, filename), sep=r"\s+", skiprows=(num_probes + 2) + skip_n_points, header=None,
                                  names=names)

            # replace all parentheses, because (ux u_y uz) is separated since all columns are separated with white space
            # as well
            for k in names:
                if k.startswith("ux"):
                    probe[k] = probe[k].str.replace(r"\(", "", regex=True).astype(float)
                elif k.startswith("uz"):
                    probe[k] = probe[k].str.replace(r"\)", "", regex=True).astype(float)
                else:
                    continue
        _probes.append(probe)

    # merge to single DF and remove all duplicates
    if len(_probes) > 1:
        _probes = concat(_probes)
    _probes.drop_duplicates(["time"], inplace=True)
    _probes.reset_index(inplace=True, drop=True)

    return _probes


def load_line_samples(load_path: str, loc: list, coord: str = "x", start: int = 0,
                      stop:  int = 50000, times: list[str] = None) -> Tuple[list, list]:
    """
    Load the line samples from OpenFOAM.

    The function searches for CSV files corresponding to one or more sampling locations and
    collects both the numerical data and their associated time steps.

    The order of variables in the OpenFOAM `volSymmTensorField` is assumed to be:
    ``[XX, XY, XZ, YY, YZ, ZZ]`` — this ordering is used to assign column names consistently.

    :param load_path: Path to the OpenFOAM case directory (containing ``postProcessing/sample_lines``).
    :type load_path: str
    :param loc: List of sampling location names to load, has to be characteristic to the file name
    :type loc: list[str]
    :param coord: Coordinate along which the sampling line was taken (e.g., "x" or "y").
    :type coord: str
    :param start: Start index for slicing the available time directories.
    :type start: int
    :param stop: Stop index for slicing the available time directories.
    :type stop: int
    :param times: List containing the write times to load, the order of times is preserved when loading, if given
                ``start`` and ``stop`` will be ignored
    :type times: list[str]
    :return: A tuple of two lists:
             - ``all_lines``: list of lists of pandas DataFrames, one per location and time step.
             - ``all_times``: list of lists of time values (as strings) corresponding to each DataFrame.
    :rtype: Tuple[list, list]
    """
    names = [coord, "p", "p_mean", "p_prime2Mean", "Ux", "Uy", "Uz", "Ux_mean", "Uy_mean", "Uz_mean", "U_prime2Mean_xx",
             "U_prime2Mean_xy", "U_prime2Mean_xz", "U_prime2Mean_yy", "U_Prime2Mean_yz", "U_Prime2Mean_zz"]

    all_lines, all_times = [], []
    for l in loc:
        print(f"Loading location '{l}'.")
        if times is None:
            files = sorted(glob(join(load_path, "postProcessing", "sample_lines", "*", f"*_{l}_*.csv")),
                           key=lambda x: float(x.split("/")[-2]))[start:stop]
        else:
            # we assume that we have exactly one file per time step which we want to load
            files = [glob(join(load_path, "postProcessing", "sample_lines", t, f"*_{l}_*.csv"))[0] for t in times]

        line = [read_csv(f, names=names, header=None, sep=",", skiprows=1, usecols=range(len(names))) for f in files]
        all_lines.append(line)
        all_times.append([t.split("/")[-2] for t in files])

    return all_lines, all_times


if __name__ == '__main__':
    pass
