"""
This script contains post-processing and plotting functions for the cylinder_3D_Re3900 test case.
Large parts of this code are adopted from:

     https://github.com/JanisGeise/learning_of_optimized_multigrid_solver_mesh_for_CFD_applications/tree/main/post_processing
"""
import regex as re
import torch as pt
import matplotlib.pyplot as plt
from flowtorch.data import CSVDataloader

from glob import glob
from os import makedirs
from typing import Union, Tuple
from os.path import join, exists
from matplotlib.patches import Circle, Rectangle

from simutils import compute_friction_velocity, load_force_coefficients, load_probes, load_line_samples, load_residuals


def get_cfl_number(load_path: str) -> dict:
    """
    Extract the mean and maximum Courant numbers from OpenFOAM solver log files.

    :param load_path: Path to the simulation directory containing one or more ``log.pimpleFoam*`` files.
    :type load_path: str
    :return: Dictionary containing the time history of Courant numbers:
        - ``"cfl_mean"`` (list[float]): Mean Courant numbers at each recorded time step.
        - ``"cfl_max"`` (list[float]): Maximum Courant numbers at each recorded time step.
        - ``"write_time"`` (list[float]): Corresponding simulation times.
    :rtype: dict
    """
    # check if we have multiple log files, if so sort them
    try:
        logs = sorted(glob(join(load_path, f"log.pimpleFoam*")), key=lambda x: int(x.split("_")[-1]))
    except ValueError:
        logs = glob(join(load_path, f"log.pimpleFoam*"))

    data = {"cfl_mean": [], "cfl_max": [], "write_time": []}
    for log in logs:
        with open(log, "r") as file:
            logfile = file.readlines()

        """
        solver log file looks something like this:
    
            Courant Number mean: 0.00156147 max: 0.860588
        """
        start_line = False
        for line in logfile:
            # omit the initial Courant number (prior starting the time loop)
            if line.startswith("Starting time loop"):
                start_line = True
            if line.startswith("Courant Number mean") and start_line:
                data["cfl_mean"].append(float(line.split(" ")[3]))
                data["cfl_max"].append(float(line.split(" ")[-1].strip("\n")))
            elif line.startswith("Time = ") and start_line:
                data["write_time"].append(float(line.split()[-1].strip("\n")))
            else:
                continue

    return data


def get_pimple_iterations(load_path: str) -> Tuple[list, list]:
    """
    Extract the number of PIMPLE pressure–velocity coupling iterations from OpenFOAM solver log files.

    :param load_path: Path to the simulation directory containing one or more ``log.pimpleFoam*`` files.
    :type load_path: str
    :return:
        A tuple ``(times, iterations)`` where:

        - ``times`` (*list[float]*): Simulation times corresponding to each time step.
        - ``iterations`` (*list[int]*): Number of PIMPLE iterations performed at each time step.
    :rtype: Tuple[list, list]
    """
    pattern = [r"PIMPLE: not converged within ", r"PIMPLE: converged in "]

    # check if we have multiple log files, if so sort them
    try:
        logs = sorted(glob(join(load_path, f"log.pimpleFoam*")), key=lambda x: int(x.split("_")[-1]))
    except ValueError:
        logs = glob(join(load_path, f"log.pimpleFoam*"))

    data, times = [], []
    for log in logs:
        with open(log, "r") as file:
            logfile = file.readlines()

        for line in logfile:
            if line.startswith(pattern[0]) or line.startswith(pattern[1]):
                data.append(int(line.split(" ")[-2]))
            elif line.startswith("Time = "):
                times.append(float(line.split()[-1].strip("\n")))
    return times, data


def get_probe_locations(load_path: str) -> pt.Tensor:
    """
    Extract probe coordinates from the ``controlDict`` file of an OpenFOAM case.

    :param load_path: Path to the top-level directory of the simulation containing the ``system/controlDict`` file.
    :type load_path: str
    :return: Tensor containing the 3D coordinates of all probes defined in ``probeLocations``.
    :rtype: torch.Tensor
    """
    pattern = r"-?\d.\d+ -?\d.\d+ -?\d.\d+"
    with open(join(load_path, "system", "controlDict"), "r") as file:
        loc = file.readlines()

    # avoid finding other coordinate tuples, which may be present in the controlDict
    idx = [j for j, line in enumerate(loc) if "probeLocations" in line][0]

    # get coordinates of probes, omit appending empty lists and map strings to float
    coords = [re.findall(pattern, line) for j, line in enumerate(loc) if re.findall(pattern, line) and j > idx]
    return pt.tensor([list(map(float, c[0].split())) for c in coords])


def compute_phi(coords: pt.Tensor, param, cylinder_pos):
    """
    Compute the azimuthal angle :math:`\\phi` around a cylinder and average a parameter field along the angular direction.

    :param coords: Tensor containing the spatial coordinates of the sample points.
    :type coords: torch.Tensor
    :param param: Tensor of parameter values (e.g., velocity, pressure) defined at the coordinates.
    :type param: torch.Tensor
    :param cylinder_pos: Position of the cylinder center used as the angular reference.
    :type cylinder_pos: list | tuple | torch.Tensor
    :return: Tuple containing the sorted azimuthal angles :math:`\\phi` (in degrees) and the corresponding averaged parameter values.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    # account for the shift wrt cylinder's origin
    coords[:, 0] -= cylinder_pos[0]
    coords[:, 1] -= cylinder_pos[1]

    # compute phi
    angle = pt.atan2(coords[:, 1], coords[:, 0]).rad2deg()

    # sort angles and avg. over z
    phi_sorted = angle[pt.argsort(angle)].unique()
    if len(param.size()) > 1:
        _avg_over_z = pt.cat([param[pt.where(angle == j)[0], :, :].mean(0).unsqueeze(-1) for j in phi_sorted], dim=-1)
    else:
        _avg_over_z = pt.cat([param[pt.where(angle == j)].mean().unsqueeze(-1) for j in phi_sorted], dim=-1)

    return phi_sorted, _avg_over_z


def plot_probes(save_path: str, data: list, num_probes: int = 10, title: str = "", param: str = "p",
                legend_list: list = None, share_y: bool = True, xlabel: str = r"$t$ in $[s]$",
                scaling_factor: Union[int, float] = 1) -> None:
    """
    Plot probe values as a function of time.

    :param save_path: Path to the directory where plots will be saved.
    :type save_path: str
    :param data: Probe data loaded using :func:`load_probes`.
    :type data: list[pandas.DataFrame]
    :param num_probes: Number of probes in the flow field.
    :type num_probes: int, optional
    :param title: Title of the plot.
    :type title: str, optional
    :param param: Field parameter to plot, e.g. ``'p'``, ``'p_rgh'`` or velocity components ``'ux'``, ``'uy'``, ``'uz'``.
    :type param: str, optional
    :param legend_list: Custom legend labels for each probe.
    :type legend_list: list[str], optional
    :param share_y: Whether all subplots share the same y-axis scaling.
    :type share_y: bool, optional
    :param xlabel: Label for the x-axis (e.g. time :math:`t` in seconds).
    :type xlabel: str, optional
    :param scaling_factor: Scaling factor applied to time, e.g. to nondimensionalize :math:`t`.
    :type scaling_factor: int | float, optional
    :return: None
    """
    min_x = min([j["time"].min() * scaling_factor for j in data])
    max_x = max([j["time"].max() * scaling_factor for j in data])
    if share_y:
        _fig, _ax = plt.subplots(nrows=num_probes, ncols=1, figsize=(8, 8), sharex="all")
    else:
        _fig, _ax = plt.subplots(nrows=num_probes, ncols=1, figsize=(8, 8), sharex="all")

    for j in range(len(data)):
        for l in range(num_probes):
            _ax[l].plot(data[j]["time"] * scaling_factor, data[j][f"{param}_probe_{l}"])
            _ax[l].set_ylabel(r"$\mathrm{probe}$" + f" ${l + 1}$", rotation="horizontal", labelpad=35)
    _ax[-1].set_xlabel(xlabel)
    _ax[-1].set_xlim(min_x, max_x)
    _fig.suptitle(title)
    _fig.tight_layout()
    if legend_list:
        _fig.subplots_adjust(bottom=0.12)
        _fig.legend(legend_list, loc="lower center", framealpha=1.0, ncol=3)
    plt.savefig(join(save_path, f"probes_vs_time_{param}.png"))
    plt.close("all")


if __name__ == "__main__":
    # define load and save path
    load_dir = join("/media", "janis", "Elements", "Janis", "cylinder_3D_Re3900_tests")
    save_dir = join("run", "cylinder_3D_Re3900", "plots_final")

    # cases to compare
    cases = ["cylinder_3D_Re3900"]
    legend = ["cylinder 3D $Re = 3900$"]

    # flow quantities
    u_inf = 39
    rho = 1
    nu = 1.0e-3

    # cylinder properties
    d = 0.1
    cylinder_position = (0.8, 1.0)

    # create plot directory
    if not exists(save_dir):
        makedirs(save_dir)

    # use latex fonts
    plt.rcParams.update({"text.usetex": True, "figure.dpi": 360})

    # load and plot the residuals
    residuals = [load_residuals(join(load_dir, c)) for c in cases]

    # omit the loading if you are using the packed data (.pt files) and load the .pt files directly for plotting
    # residuals = [pt.load(join(load_dir, cases[0], "postProcessing", "solverInfo", "solverInfo.pt"), weights_only=False)]

    for i in range(len(cases)):
        fig, ax = plt.subplots(2, 1, sharex="col", figsize=(6, 4))
        ax[0].plot(residuals[i].time * u_inf / d, residuals[i].Ux_initial, label="$U_{x}$")
        ax[0].plot(residuals[i].time * u_inf / d, residuals[i].Uy_initial, label="$U_{y}$")
        ax[0].plot(residuals[i].time * u_inf / d, residuals[i].Uz_initial, label="$U_{z}$")
        ax[0].plot(residuals[i].time * u_inf / d, residuals[i].p_initial, label="$p$")

        ax[1].plot(residuals[i].time * u_inf / d, residuals[i].Ux_final)
        ax[1].plot(residuals[i].time * u_inf / d, residuals[i].Uy_final)
        ax[1].plot(residuals[i].time * u_inf / d, residuals[i].Uz_final)
        ax[1].plot(residuals[i].time * u_inf / d, residuals[i].p_final)

        ax[-1].set_xlabel(r"$t \frac{U_{\infty}}{d}$")
        ax[0].set_ylabel(r"$\mathrm{initial}$")
        ax[1].set_ylabel(r"$\mathrm{final}$")
        ax[0].set_yscale("log")
        ax[1].set_yscale("log")
        ax[-1].set_xlim(residuals[i].time.iloc[0], residuals[i].time.iloc[-1] * u_inf / d)

        for k in range(2):
            ax[k].grid(visible=True, which="major", linestyle="-", alpha=0.35, color="black", axis="both")
            ax[k].minorticks_on()
            ax[k].tick_params(axis="x", which="minor", bottom=False)
            ax[k].grid(visible=True, which="minor", linestyle="--", alpha=0.25, color="black", axis="both")

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        fig.legend(loc="upper center", ncols=4)
        plt.savefig(join(save_dir, f"residuals_vs_t_case_{i}.png"))
        plt.close("all")
    del residuals

    """
    # get the Courant number from solver log and plot it
    cfl = [get_cfl_number(join(load_dir, c)) for c in cases]

    # we write out the residuals every time step
    fig, ax = plt.subplots(1, 1, sharex="col", figsize=(6, 3))
    color = [f"C{i}" for i in range(len(cases))]
    for i in range(len(cases)):
        ax.plot(pt.tensor(cfl[i]["write_time"]) * u_inf / d, cfl[i]["cfl_mean"], label=legend[i], ls="-",
                color=color[i])
        ax.plot(pt.tensor(cfl[i]["write_time"]) * u_inf / d, cfl[i]["cfl_max"], ls=":", color=color[i])
        ax.set_xlabel(r"$t \frac{U_{\infty}}{d}$")
        ax.set_ylabel(r"$Co$ $(\mu, \mathrm{max})$")
    ax.set_xlim(min(cfl[0]["write_time"]), max(cfl[0]["write_time"]) * u_inf / d)
    fig.tight_layout()
    fig.subplots_adjust()
    ax.legend(loc="upper right", ncols=4)
    plt.savefig(join(save_dir, f"courant_vs_t.png"))
    plt.close("all")

    # check PIMPLE iterations
    n_iter = [get_pimple_iterations(join(load_dir, c)) for c in cases]

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    for i in range(len(cases)):
        ax.plot(pt.tensor(n_iter[i][0]) * u_inf / d, n_iter[i][1], label=legend[i])
    ax.set_xlabel(r"$t \frac{U_{\infty}}{d}$")
    ax.set_ylabel(r"$N_\mathrm{PIMPLE}$")
    ax.set_xlim(min(n_iter[0][0]) * u_inf / d, max(n_iter[0][0]) * u_inf / d)
    fig.tight_layout()
    ax.legend(ncol=len(cases), loc="upper right")
    fig.subplots_adjust()
    plt.savefig(join(save_dir, "n_pimple_iter_vs_t.png"))
    plt.close("all")
    del n_iter, cfl
    # """

    # load and plot the force coefficients
    forces = [load_force_coefficients(join(load_dir, c)) for c in cases]

    # omit the loading if you are using the packed data (.pt files) and load the .pt files directly for plotting
    # forces = [pt.load(join(load_dir, cases[0], "postProcessing", "forces", "coefficients.pt"), weights_only=False)]

    fig, ax = plt.subplots(2, 1, figsize=(6, 4), sharex="col")

    for i in range(len(cases)):
        ax[0].plot(forces[i].time * u_inf / d, forces[i].cx, label=legend[i])
        ax[1].plot(forces[i].time * u_inf / d, forces[i].cy)
        ax[i].set_ylim(0.85, 1.15)
    ax[0].set_ylabel(r"$c_d$")
    ax[1].set_ylabel(r"$c_l$")
    ax[-1].set_xlabel(r"$t \frac{U_{\infty}}{d}$")
    ax[-1].set_xlim(0, forces[0].time.iloc[-1] * u_inf / d)
    fig.tight_layout()
    fig.legend(ncol=len(cases), loc="upper center")
    fig.subplots_adjust(top=0.9)
    plt.savefig(join(save_dir, "coefficients_vs_t.png"))
    plt.close("all")
    del forces

    # load and plot the probes for p and U
    n_probes = 7
    for p in ["U", "p"]:
        # load the probe
        probes = [load_probes(join(load_dir, c), num_probes=n_probes, filename=p) for c in cases]

        # omit the loading if you are using the packed data (.pt files) and load the .pt files directly for plotting
        # probes = [pt.load(join(load_dir, cases[0], "postProcessing", "probes", "probes.pt"), weights_only=False)[p]]

        if p == "U":
            for i in ["ux", "uy", "uz"]:
                plot_probes(save_dir, probes, param=i, scaling_factor=u_inf / d, xlabel=r"$t \frac{U_{\infty}}{d}$",
                            num_probes=n_probes, legend_list=legend)
        else:
            plot_probes(save_dir, probes, param=p, scaling_factor=u_inf / d, xlabel=r"$t \frac{U_{\infty}}{d}$",
                        num_probes=n_probes, legend_list=legend)
    del probes

    # compute u_tau, since this is an instantaneous quantity we have to loop over all time steps
    # we have 22801 snapshots between t == 0.19225 ... 5.89225, so just use the last few thousand for averaging
    t_start = 2.89225
    u_tau, grad_u = [], []
    for c in cases:
        _, _, _, _, u_tau_tmp, grad_u_tmp = compute_friction_velocity(join(load_dir, c, "postProcessing",
                                                                           "sample_planes"), "gradU_cylinder.raw",
                                                                      t_start, nu, dtype=pt.float32)
        u_tau.append(u_tau_tmp)
        grad_u.append(grad_u_tmp)
    del _, u_tau_tmp, grad_u_tmp

    # load the coordinates of the cylinder surface
    loader = [CSVDataloader.from_foam_surface(join(load_dir, c, "postProcessing", "sample_planes"),
                                              "gradU_cylinder.raw", dtype=pt.float32) for c in cases]

    # compute mean and std. dev. of u_tau
    phi, u_tau_mean, u_tau_std, grad_u_mean, grad_u_std = [], [], [], [], []

    for i in range(len(cases)):
        u_tau_tmp = pt.cat([u.unsqueeze(-1) for u in u_tau[i]], dim=-1)
        grad_u_tmp = pt.cat([u.unsqueeze(-1) for u in grad_u[i]], dim=-1)

        # compute avg. u_tau and grad_u wrt z-coordinate for temporal mean and std. deviation
        phi_tmp, u_tau_mean_tmp = compute_phi(loader[i].vertices, u_tau_tmp.mean(-1), cylinder_position)
        _, u_tau_std_tmp = compute_phi(loader[i].vertices, u_tau_tmp.std(-1), cylinder_position)
        _, grad_u_mean_tmp = compute_phi(loader[i].vertices, grad_u_tmp.mean(-1), cylinder_position)
        _, grad_u_std_tmp = compute_phi(loader[i].vertices, grad_u_tmp.std(-1), cylinder_position)

        phi.append(phi_tmp)
        u_tau_mean.append(u_tau_mean_tmp)
        u_tau_std.append(u_tau_std_tmp)
        grad_u_mean.append(grad_u_mean_tmp)
        grad_u_std.append(grad_u_std_tmp)

    del _, u_tau_mean_tmp, u_tau_std_tmp, phi_tmp, grad_u_mean_tmp, grad_u_std_tmp, grad_u, u_tau, grad_u_tmp

    # plot u_tau and avg. grad_u
    fig, ax = plt.subplots(5, 2, figsize=(6, 5), sharex="col")
    for i in range(len(cases)):
        ax[0][0].plot(phi[i] + 180, u_tau_mean[i] / u_inf, label=legend[i])
        ax[0][1].plot(phi[i] + 180, u_tau_std[i] / u_inf)

        # du/dx
        ax[1][0].plot(phi[i] + 180, grad_u_mean[i][0, 0, :] * d / u_inf)
        ax[1][1].plot(phi[i] + 180, grad_u_std[i][0, 0, :] * d / u_inf)
        ax[2][0].plot(phi[i] + 180, grad_u_mean[i][1, 1, :] * d / u_inf)
        ax[2][1].plot(phi[i] + 180, grad_u_std[i][1, 1, :] * d / u_inf)
        ax[3][0].plot(phi[i] + 180, grad_u_mean[i][0, 1, :] * d / u_inf)
        ax[3][1].plot(phi[i] + 180, grad_u_std[i][0, 1, :] * d / u_inf)
        ax[4][0].plot(phi[i] + 180, grad_u_mean[i][1, 0, :] * d / u_inf)
        ax[4][1].plot(phi[i] + 180, grad_u_std[i][1, 0, :] * d / u_inf)

    # set all labels
    ax[-1][0].set_xlabel(r"$\phi$ $[^\circ]$")
    ax[-1][1].set_xlabel(r"$\phi$ $[^\circ]$")
    ax[0][0].set_ylabel(r"$\overline{u}_{\tau} / U_{\infty}$")
    ax[1][0].set_ylabel(r"$\partial_{\tilde{x}} \tilde{u}$")
    ax[2][0].set_ylabel(r"$\partial_{\tilde{y}} \tilde{v}$")
    ax[3][0].set_ylabel(r"$\partial_{\tilde{y}} \tilde{u}$")
    ax[4][0].set_ylabel(r"$\partial_{\tilde{x}} \tilde{v}$")

    # remaining case
    ax[0][0].set_title("mean")
    ax[0][1].set_title("std.")
    ax[-1][0].set_xlim(0, 360)
    ax[-1][1].set_xlim(0, 360)
    fig.tight_layout()
    fig.legend(ncol=len(cases), loc="upper center")
    fig.subplots_adjust(top=0.88)
    plt.savefig(join(save_dir, "u_tau_mean_vs_phi.png"))
    plt.close("all")
    del grad_u_std, grad_u_mean, u_tau_mean, u_tau_std

    # plot cp vs. angle phi, the cylinder coordinates remain the same (fig. 7)
    phi, cp_avg_over_z = [], []
    every = 5       # use every nth snapshot
    for i in range(len(cases)):
        loader = CSVDataloader.from_foam_surface(join(load_dir, cases[i], "postProcessing", "sample_planes"),
                                                 "p_cylinder.raw", dtype=pt.float32)
        p = loader.load_snapshot("p", loader.write_times[::every])

        # compute mean cp distribution, cp and transform cartesian coordinates to angle, compare:
        # https://www.openfoam.com/documentation/guides/latest/doc/guide-fos-field-pressure.html
        cp_tmp = 2 * p.mean(-1).squeeze() / (rho * u_inf ** 2)
        phi_tmp, cp_avg_over_z_tmp = compute_phi(loader.vertices, cp_tmp, cylinder_position)
        phi.append(phi_tmp)
        cp_avg_over_z.append(cp_avg_over_z_tmp)

    del cp_avg_over_z_tmp, phi_tmp, cp_tmp, p

    # plot avg. cp vs. cylinder angle
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    for i, case in enumerate(cases):
        ax.plot(phi[i] + 180, cp_avg_over_z[i], label=legend[i])
    ax.set_xlabel(r"$\phi$ $[^\circ]$")
    ax.set_ylabel(r"$\overline{c}_p$")
    ax.set_xlim(0, 360)
    fig.tight_layout()
    fig.legend(ncol=len(cases), loc="upper center")
    fig.subplots_adjust(top=0.88)
    plt.savefig(join(save_dir, "cp_vs_phi.png"))
    plt.close("all")
    del phi, cp_avg_over_z

    # plot UMean line for the wake in x-direction along the wake (fig. 8)
    line_samples_mean = [load_line_samples(join(load_dir, c), ["1"], times=["5.89225"])[0][0][0] for c in cases]

    fig, ax = plt.subplots(2, 1, figsize=(6, 4), sharex="col")
    for j in range(len(cases)):
        for sample in line_samples_mean:
            if j == 0:
                ax[0].plot(sample["x"] / d, sample["Ux_mean"] / u_inf, label=legend[j])
            else:
                ax[0].plot(sample["x"] / d, sample["Ux_mean"] / u_inf)
            ax[1].plot(sample["x"] / d, sample["U_prime2Mean_xx"] / u_inf ** 2)
    ax[0].set_ylabel(r"$\overline{u} / U_{\infty}$")
    ax[1].set_ylabel(r"$\overline{u^{\prime} u^{\prime}} / U_{\infty}^2$")
    ax[-1].set_xlabel(r"$x / D$")
    ax[-1].set_xlim(line_samples_mean[0]["x"].min() / d, line_samples_mean[0]["x"].max() / d)
    fig.tight_layout()
    fig.legend(ncol=len(cases), loc="upper center")
    fig.subplots_adjust(top=0.9)
    plt.savefig(join(save_dir, f"U_wake_yd1.png"))
    plt.close("all")

    # plot UMean (fig. 9)
    loc = ["5", "7", "10"]
    line_samples_mean = [load_line_samples(join(load_dir, c), loc, times=["5.89225"])[0] for c in cases]

    fig, ax = plt.subplots(1, len(loc), figsize=(6, 4), sharey="row")
    for i, sample in enumerate(line_samples_mean):
        for col in range(len(sample)):
            if col == 0:
                ax[col].plot(sample[col][0]["Ux_mean"] / u_inf, sample[col][0]["x"] / d, label=legend[i])
            else:
                ax[col].plot(sample[col][0]["Ux_mean"] / u_inf, sample[col][0]["x"] / d)
            ax[col].set_xlabel(r"$\overline{u} / U_{\infty}$")
            ax[col].set_title(rf"$x / D = {loc[col]}$")
    ax[0].set_ylabel(r"$y / D$")
    ax[0].set_ylim(0, 20)
    fig.tight_layout()
    fig.legend(ncol=1, loc="upper center")
    fig.subplots_adjust(top=0.85)
    plt.savefig(join(save_dir, f"U_xd.png"))
    plt.close("all")

    # load line samples for Re stresses
    loc = ["106", "154", "202"]
    line_samples_mean = [load_line_samples(join(load_dir, c), loc, times=["5.89225"])[0] for c in cases]

    # plot UMean and UPrime2Mean (fig. 10)
    fig, ax = plt.subplots(nrows=3, ncols=2, sharex="all", sharey="col", figsize=(6, 6))
    for i, sample in enumerate(line_samples_mean):
        for row in range(len(sample)):
            if row == 0:
                ax[row][0].plot(sample[row][0]["x"] / d, sample[row][0]["Ux_mean"] / u_inf, label=legend[i])
            else:
                ax[row][0].plot(sample[row][0]["x"] / d, sample[row][0]["Ux_mean"] / u_inf)
            ax[row][1].plot(sample[row][0]["x"] / d, sample[row][0]["U_prime2Mean_xx"] / u_inf ** 2)

            ax[row][0].set_ylabel(r"$\overline{u} / U_{\infty}$")
            ax[row][1].set_ylabel(r"$\overline{u^{\prime} u^{\prime}} / U_{\infty}^2$")

            # add second axis for the location label on the right side
            ax_twin = ax[row][1].twinx()
            ax_twin.set_ylabel(r"$x / D =$" + f"${loc[row][0]}.{loc[row][1:]}$")
            ax_twin.yaxis.set_label_position("right")
            ax_twin.set_yticks([])

    ax[-1][-1].set_xlabel(r"$y / D$")
    ax[-1][0].set_xlabel(r"$y / D$")
    fig.tight_layout()
    fig.legend(ncol=len(cases), loc="upper center")
    fig.subplots_adjust(top=0.92, wspace=0.4)
    plt.savefig(join(save_dir, "U_xd_Uprime2Mean.png"))
    plt.close("all")

    # VPrime2Mean (fig. 11)
    fig, ax = plt.subplots(nrows=3, ncols=2, sharex="all", sharey="col", figsize=(6, 6))
    for i, sample in enumerate(line_samples_mean):
        for row in range(len(sample)):
            if row == 0:
                ax[row][0].plot(sample[row][0]["x"] / d, sample[row][0]["Uy_mean"] / u_inf, label=legend[i])
            else:
                ax[row][0].plot(sample[row][0]["x"] / d, sample[row][0]["Uy_mean"] / u_inf)
            ax[row][1].plot(sample[row][0]["x"] / d, sample[row][0]["U_prime2Mean_yy"] / u_inf ** 2)

            ax[row][0].set_ylabel(r"$\overline{v} / U_{\infty}$")
            ax[row][1].set_ylabel(r"$\overline{v^{\prime} v^{\prime}} / U_{\infty}^2$")

            # add second axis for the location label on the right side
            ax_twin = ax[row][1].twinx()
            ax_twin.set_ylabel(r"$x / D =$" + f"${loc[row][0]}.{loc[row][1:]}$")
            ax_twin.yaxis.set_label_position("right")
            ax_twin.set_yticks([])

    ax[-1][-1].set_xlabel(r"$y / D$")
    ax[-1][0].set_xlabel(r"$y / D$")
    fig.tight_layout()
    fig.legend(ncol=len(cases), loc="upper center")
    fig.subplots_adjust(top=0.92, wspace=0.4)
    plt.savefig(join(save_dir, "V_xd_Vprime2Mean.png"))
    plt.close("all")
    del line_samples_mean

    # load and plot plane samplings (not present in paper)
    # adjust field, e.g., to 'p' (although plotting different components is kinda useless for a scalar field...)
    field = "U"
    every = 20       # load every nth snapshot
    for c in range(len(cases)):
        loader = CSVDataloader.from_foam_surface(join(load_dir, cases[c], "postProcessing", "sample_planes"),
                                                 f"{field}_plane_xy.raw", dtype=pt.float32)

        # since we have so many snapshots, just use every nth to avoid memory overflow
        f = pt.cat([loader.load_snapshot(f"{field}_{cmp}", loader.write_times[::every]).unsqueeze(-1) for cmp in
                    ["x", "y"]], dim=-1)

        # plot mean and std. deviation for all components
        fig, ax = plt.subplots(nrows=2, ncols=2, sharex="all", sharey="all", figsize=(6, 6))
        for row in range(2):
            ax[row][0].tricontourf(loader.vertices[:, 0] / d, loader.vertices[:, 1] / d, f[:, :, row].mean(1))
            ax[row][1].tricontourf(loader.vertices[:, 0] / d, loader.vertices[:, 1] / d, f[:, :, row].std(1))
            ax[row][0].add_patch(Circle((cylinder_position[0] / d, cylinder_position[1] / d), radius=1 / 2,
                                        color="white"))
            ax[row][1].add_patch(Circle((cylinder_position[0] / d, cylinder_position[1] / d), radius=1/2,
                                        color="white"))
            ax[row][0].set_aspect("equal")
            ax[row][1].set_aspect("equal")
            ax[row][0].set_ylabel("$y/D$")
        ax[-1][0].set_xlabel("$x/D$")
        ax[-1][1].set_xlabel("$x/D$")
        ax[0][0].set_title(r"$\overline{u}$")
        ax[0][1].set_title(r"$\sigma\left(u\right)$")
        ax[1][0].set_title(r"$\overline{v}$")
        ax[1][1].set_title(r"$\sigma\left(v\right)$")
        fig.tight_layout()
        fig.subplots_adjust()
        plt.savefig(join(save_dir, f"mean_std_{field}_xy_plane_case{c}.png"))
        plt.close("all")
        del loader, f

        # same for x-z-plane
        loader = CSVDataloader.from_foam_surface(join(load_dir, cases[c], "postProcessing", "sample_planes"),
                                                 f"{field}_plane_xz.raw", dtype=pt.float32)
        f = pt.cat([loader.load_snapshot(f"{field}_{cmp}", loader.write_times[::every]).unsqueeze(-1) for cmp in
                    ["x", "z"]], dim=-1)

        fig, ax = plt.subplots(nrows=2, ncols=2, sharex="all", sharey="all", figsize=(6, 4))
        for row in range(2):
            ax[row][0].tricontourf(loader.vertices[:, 0] / d, loader.vertices[:, 2] / d, f[:, :, row].mean(1))
            ax[row][1].tricontourf(loader.vertices[:, 0] / d, loader.vertices[:, 2] / d, f[:, :, row].std(1))
            ax[row][0].add_patch(Rectangle((cylinder_position[0] / d - 1/2, 0), width=1, height=pt.pi,
                                           color="white"))
            ax[row][1].add_patch(Rectangle((cylinder_position[0] / d - 1/2, 0), width=1, height=pt.pi,
                                           color="white"))
            ax[row][0].set_ylabel("$z/D$")
        ax[-1][0].set_xlabel("$x/D$")
        ax[-1][1].set_xlabel("$x/D$")
        ax[-1][0].set_ylim(0, pt.pi)
        ax[-1][1].set_ylim(0, pt.pi)
        ax[0][0].set_title(r"$\overline{u}$")
        ax[0][1].set_title(r"$\sigma\left(u\right)$")
        ax[1][0].set_title(r"$\overline{w}$")
        ax[1][1].set_title(r"$\sigma\left(w\right)$")
        fig.tight_layout()
        fig.subplots_adjust()
        plt.savefig(join(save_dir, f"mean_std_{field}_xz_plane_case{c}.png"))
        plt.close("all")
        del loader, f
