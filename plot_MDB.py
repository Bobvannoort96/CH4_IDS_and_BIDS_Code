"""
Description of code:
    Compute the MDBs for a given method, IDS type
    and example type
Author:
    Bob van Noort
Date:
    27 April

"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
from Functions import *
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox, TransformedBbox
import matplotlib.lines as mlines


def compute_P_CD(
    b_mag,
    cti_model,
    d_vector,
    Qtt,
    N_samples,
    compute_Pis,
    B_T,
    alpha_prime,
    type_of_testing,
    type_of_DS,
    type_of_example,
    qmax,
):
    """
    Compute P_CD for a given bias magnitude.
    """
    t_mean = (b_mag * cti_model @ d_vector).flatten()
    t_samples = scipy.stats.multivariate_normal.rvs(
        mean=t_mean, cov=Qtt, size=N_samples
    ).T

    Pis = compute_Pis(
        t_samples,
        Qtt,
        B_T,
        alpha_prime,
        type_of_testing,
        type_of_DS=type_of_DS,
        type_of_example=type_of_example,
        qmax=qmax,
    )

    # Compute the probability of correct detection, i.e. not accepting H0
    PCD_computed = np.sum(Pis != 0) / N_samples
    return PCD_computed


def find_bias_b_MDB(
    PCD_goal,
    b_min,
    b_max,
    tol,
    max_iter,
    cti_model,
    d_vector,
    Qtt,
    N_samples,
    compute_Pis,
    B_T,
    alpha_prime,
    type_of_testing,
    type_of_DS,
    type_of_example,
    qmax,
):
    """
    Bisection search for b_mag such that P_CD ≈ PCD_goal.
    """

    for _ in range(max_iter):
        b_mid = 0.5 * (b_min + b_max)

        PCD_mid = compute_P_CD(
            b_mid,
            cti_model,
            d_vector,
            Qtt,
            N_samples,
            compute_Pis,
            B_T,
            alpha_prime,
            type_of_testing,
            type_of_DS,
            type_of_example,
            qmax,
        )

        diff = PCD_mid - PCD_goal

        if abs(diff) < tol:
            return b_mid, PCD_mid

        PCD_min = compute_P_CD(
            b_min,
            cti_model,
            d_vector,
            Qtt,
            N_samples,
            compute_Pis,
            B_T,
            alpha_prime,
            type_of_testing,
            type_of_DS,
            type_of_example,
            qmax,
        )

        # Check which side to keep
        if (PCD_min - PCD_goal) * diff < 0:
            b_max = b_mid
        else:
            b_min = b_mid

        print("b_min, b_mid, b_max", b_min, b_mid, b_max)
        print("PCI_mid, PCI_min", PCD_mid, PCD_min)

    b_mag = 0.5 * (b_min + b_max)
    PCD_bmag = compute_P_CD(
        b_mag,
        cti_model,
        d_vector,
        Qtt,
        N_samples,
        compute_Pis,
        B_T,
        alpha_prime,
        type_of_testing,
        type_of_DS,
        type_of_example,
    )
    return b_mag, PCD_bmag


# Code that functions well with the supercomputer.
# It takes d_i as input, ranging from 0 to 999. So, the supercomputer
# can be run with sbatch --array=0-999
def compute_MDB_for_d_i(
    d_i,
    type_of_testing,
    type_of_DS,
    type_of_example,
    alpha_0,
    alpha_prime,
    type_of_alpha,
    for_partition,
    qmax,
    PCD_goal,
):
    """
    Computes MDB and PCD_computed for one direction index d_i.

    Parameters
    ----------
    d_i : int
        Index of direction vector to process.
    type_of_testing : str
    type_of_DS : str
    type_of_example : str
    alpha_0 : float
    alpha_prime : float
    type_of_alpha : str
    for_partition : str
    qmax: int # specifies the total / max nr of outliers to detect for.
    PCD_goal: float, specifies the desired probability of correct detection.

    Returns
    -------
    MIB_vector : ndarray
        Computed MIB (bias vector).
    PCI_computed : float
        Probability of correct identification achieved.
    """

    # Load problem setup
    if type_of_example == "simple":
        indices_partitions = {
            "P0": 0,
            "P1": 1,
            "P2": 2,
            "P3": 3,
            "P4": 4,
            "P12": 5,
            "P13": 6,
            "P14": 7,
            "P21": 5,
            "P23": 8,
            "P24": 9,
            "P31": 6,
            "P32": 8,
            "P34": 10,
            "P41": 7,
            "P42": 9,
            "P43": 10,
            "P99": -1,
        }
        m, n, r, A, alpha, sigma, Qyy, Qyy_inv, B_T, Qtt, Qtt_inv = setup(
            alpha_0=alpha_0, alpha_method=type_of_alpha, alpha_prime=alpha_prime
        )
    elif type_of_example == "SPP_GNSS":
        indices_partitions = load_indices_partitions_GNSS_ex(False)
        m, n, r, A, alpha, sigma, Qyy, Qyy_inv, B_T, Qtt, Qtt_inv = (
            setup_SPP_GNSS_example(
                alpha_0=alpha_0, alpha_method=type_of_alpha, alpha_prime=alpha_prime
            )
        )
    else:
        raise ValueError("Unknown type_of_example: {}".format(type_of_example))

    q_i = len(for_partition.replace("P", ""))
    idx_under_hypt = indices_partitions[for_partition]
    n_points = 1000
    print("idx_under_hypt", idx_under_hypt)

    if q_i == 1:
        d_vectors = np.array([[1]])
        idxes = int(for_partition.replace("P", ""))
        ci_model = np.eye(m)[:, idxes - 1].reshape(-1, 1)
    elif q_i == 2:
        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        d_vectors = np.stack((np.cos(theta), np.sin(theta)), axis=0)
        idxes = np.array([int(zeta) for zeta in for_partition.replace("P", "")])
        ci_model = np.eye(m)[:, idxes - 1]
    elif q_i == 3:
        _, _, _, d_vectors = generate_t_grid(31, 32)
        idxes = np.array([int(zeta) for zeta in for_partition.replace("P", "")])
        ci_model = np.eye(m)[:, idxes - 1]
    else:
        raise ValueError("Partition dimension not supported.")

    if d_i >= d_vectors.shape[1]:
        raise ValueError(
            f"Requested d_i={d_i}, but only {d_vectors.shape[1]} directions available!"
        )

    d_vector = d_vectors[:, d_i].reshape(-1, 1)

    # Start simulation
    cti_model = B_T @ ci_model
    N_samples = int(1e6)

    # Initialize
    a_, b_, c_ = 0, 10, 3  # update these parameter if we cannot find a good mdb
    MDB_range = np.arange(a_, b_) * c_
    PCD_initialized = np.zeros(len(MDB_range))
    while True:
        for i_mdb, Mdb in enumerate(MDB_range):
            t_mean = Mdb * cti_model @ d_vector
            t_samples = scipy.stats.multivariate_normal.rvs(
                mean=t_mean.flatten(), cov=Qtt, size=(N_samples,)
            ).T
            Pis = compute_Pis(
                t_samples,
                Qtt,
                B_T,
                alpha_prime,
                type_of_testing,
                type_of_DS=type_of_DS,
                type_of_example=type_of_example,
                qmax=qmax,
            )
            PCD_initialized[i_mdb] = np.sum(Pis != 0) / N_samples
        print("PCD_initialized", PCD_initialized)
        where_larger_than_PCD_goal = PCD_initialized > PCD_goal
        if np.sum(where_larger_than_PCD_goal) == 0:
            # Cannot find good MDB-range
            a_ = np.copy(b_)
            b_ += 10
        else:
            break

    b_max = MDB_range[where_larger_than_PCD_goal][0]
    b_min = MDB_range[~where_larger_than_PCD_goal][-1]
    print("Where_larger_PCD", where_larger_than_PCD_goal)
    print("bmin, bmax", b_min, b_max)
    # if b_mag > 150:
    #     d_b_mag = 0.5

    b_mag, PCD_computed = find_bias_b_MDB(
        PCD_goal,
        b_min=b_min,
        b_max=b_max,
        tol=1e-3,
        max_iter=50,
        cti_model=cti_model,
        d_vector=d_vector,
        Qtt=Qtt,
        N_samples=N_samples,
        compute_Pis=compute_Pis,
        B_T=B_T,
        alpha_prime=alpha_prime,
        type_of_testing=type_of_testing,
        type_of_DS=type_of_DS,
        type_of_example=type_of_example,
        qmax=qmax,
    )

    MDB_vector = b_mag * d_vector.flatten()
    return MDB_vector, PCD_computed


def plot_MDB_surface(
    type_of_testing,
    type_of_DS,
    type_of_example,
    alpha_0,
    alpha_prime,
    type_of_alpha,
    for_partition,
    PCD_goal,
    q_i,
    ax=None,
    linestyle="-",
    legend_label=None,
    color_lines=None,
):

    print(
        "Type_of_testing, type_of_DS, PCD_goal", type_of_testing, type_of_DS, PCD_goal
    )
    ## Load the original d_vectors array

    if type_of_testing == "classical DIA":
        resDir = os.path.join(
            r"C:\Users\bgvannoort\Documents\IDS\Results\MDBs",
            type_of_example,
            for_partition.replace("P", "H"),
            type_of_testing,
            "PCD_goal={}".format(PCD_goal),
        )
        if legend_label is None:
            legend_label = "{}".format(type_of_testing)
    else:
        resDir = os.path.join(
            r"C:\Users\bgvannoort\Documents\IDS\Results\MDBs",
            type_of_example,
            for_partition.replace("P", "H"),
            "PCD_goal={}".format(PCD_goal),
            type_of_testing,
            type_of_DS,
        )
        if legend_label is None:
            legend_label = "{} {}".format(type_of_testing, type_of_DS)

    print("resdir", resDir)
    # Load MIB data
    MDBs = np.loadtxt(os.path.join(resDir, "MDB_vectors.txt"))
    print("Mdbs.shape", MDBs.shape)
    # Load PCI_computed_data
    PCD_computed = np.loadtxt(os.path.join(resDir, "PCD_computed.txt"))
    print("PCD_computed.shape", PCD_computed.shape)

    for idx_PCD in range(PCD_computed.shape[0]):
        di, PCD_comp = PCD_computed[idx_PCD, :]
        # print(PCI_comp)
        if PCD_comp is not np.nan:
            if np.abs(PCD_comp - PCD_goal) > 0.01:
                print("At di = {}, the computed PCI = {}".format(di, PCD_comp))

    orders_PCD = PCD_computed[:, 0].astype(int)
    order_MDB = MDBs[:, 0].astype(int)
    # sys.exit()

    argsorted_PCD = np.argsort(orders_PCD)
    argsorted_MDB = np.argsort(order_MDB)

    PCD_computed = PCD_computed[
        argsorted_PCD, 1
    ]  # only last column really corresponds to PCI

    MDBs_ordered = MDBs[argsorted_MDB, 1:]

    if len(MDBs_ordered) != 1000:
        raise Exception("Length of Array is not 1000")

    if q_i == 2:
        if ax is not None:
            pass
        else:
            fig, ax = plt.subplots()

        # if type_of_DS == 'B' and PCI_goal == 0.8 and type_of_testing == 'R_IDS':
        #     ## For positive b1 comps
        #     c_MIBs = np.copy(MIBs_ordered)
        #     bool_tot = np.logical_and(c_MIBs[:,0] > 50, c_MIBs[:, 0] < 60 )
        #     bool_tot2 = np.logical_and(bool_tot, c_MIBs[:, 1] > 50)
        #     arrays_to_sort = np.linalg.norm(c_MIBs[bool_tot2, :], axis=1)
        #     argsort_typeB = np.argsort(arrays_to_sort)
        #     c_MIBs[bool_tot2, :] = c_MIBs[bool_tot2, :][argsort_typeB]
        #     # print('In this loop')
        #     MIBs_ordered = c_MIBs

        #     # For negative b1
        #     c_MIBs = np.copy(MIBs_ordered)
        #     bool_tot = np.logical_and(c_MIBs[:,0] < - 50, c_MIBs[:, 0] > -60 )
        #     bool_tot2 = np.logical_and(bool_tot, c_MIBs[:, 1] < -50)
        #     arrays_to_sort = np.linalg.norm(c_MIBs[bool_tot2, :], axis=1)
        #     argsort_typeB = np.argsort(arrays_to_sort)
        #     c_MIBs[bool_tot2, :] = c_MIBs[bool_tot2, :][argsort_typeB]
        #     MIBs_ordered = c_MIBs

        if color_lines is not None:
            ax.plot(
                MDBs_ordered[:, 0],
                MDBs_ordered[:, 1],
                label=legend_label,
                linestyle=linestyle,
                color=color_lines,
                linewidth=2.0,
            )
        else:
            ax.plot(
                MDBs_ordered[:, 0],
                MDBs_ordered[:, 1],
                label=legend_label,
                linestyle=linestyle,
                linewidth=2.0,
            )
        # ax.scatter(MIBs_ordered[:, 0], MIBs_ordered[:,1], label=legend_label.replace('_', ''), marker='.')

    return ax, MDBs_ordered, PCD_computed


# %%
if __name__ == "__main__":
    type_of_testing = "R_IDS"
    type_of_DS = "A"
    # type_of_example='simple'
    type_of_example = "SPP_GNSS"
    alpha_0 = 0.01
    alpha_prime = 0.38
    type_of_alpha = "Kok_IDS"
    qmax = 2

    plt.close("all")

    # for_partition = 'P24'
    for_partition = "P13"

    qi = len(for_partition.replace("P", ""))

    plt.rcParams["font.size"] = 16

    colors_default = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # colors_default = ['red', 'green']

    colors_default = dict(zip(["A", "B", "C"], colors_default))
    # colors_default = dict(zip(['A',   'C'], colors_default))

    linestyles_list = ["-.", "-", "--"]
    # legend_labels = [r'$\mathcal{P}_0^{\text{OMT}}$', r'$\mathcal{P}_0^{\text{DS/IDS C}}$']
    legend_labels = [
        r"$\mathcal{P}_0^{\text{A}}$",
        r"$\mathcal{P}_0^{\text{B}}$",
        r"$\mathcal{P}_0^{\text{C}}$",
    ]

    res_MDBS = []
    res_PCD = []

    PCD_goals = [0.7, 0.8, 0.9]

    if type_of_testing != "classical DIA":
        # fig, ax = plt.subplots(1, 2,  figsize=(11, 6.63))
        fig, ax = plt.subplots(1, 3, figsize=(19, 6.63))  # if three plots are necessary
        # fig2, ax2 = plt.subplots(figsize=(10,10))
        fig2, (ax2_main, ax2_zoom) = plt.subplots(1, 2, figsize=(12, 6))
        for idx_PCD_goal, PCD_goal in enumerate(PCD_goals):
            for i_type, types in enumerate(["A", "B", "C"]):
                ax[i_type], MDBs, PCD_comp = plot_MDB_surface(
                    type_of_testing,
                    types,
                    type_of_example,
                    alpha_0,
                    alpha_prime,
                    type_of_alpha,
                    for_partition,
                    PCD_goal,
                    q_i=2,
                    ax=ax[i_type],
                    linestyle=linestyles_list[idx_PCD_goal],
                    legend_label=r" $P_{\text{CD}}=" + rf"{PCD_goal}$ ",
                    color_lines=colors_default[types],
                )

                ax2_main, MDBs, PCD_comp = plot_MDB_surface(
                    type_of_testing,
                    types,
                    type_of_example,
                    alpha_0,
                    alpha_prime,
                    type_of_alpha,
                    for_partition,
                    PCD_goal,
                    q_i=2,
                    ax=ax2_main,
                    linestyle=linestyles_list[idx_PCD_goal],
                    legend_label=legend_labels[i_type]
                    + r" $P_{\text{CD}}="
                    + rf"{PCD_goal}$ ",
                    color_lines=colors_default[types],
                )
                ax2_zoom, MDBs, PCD_comp = plot_MDB_surface(
                    type_of_testing,
                    types,
                    type_of_example,
                    alpha_0,
                    alpha_prime,
                    type_of_alpha,
                    for_partition,
                    PCD_goal,
                    q_i=2,
                    ax=ax2_zoom,
                    linestyle=linestyles_list[idx_PCD_goal],
                    legend_label=legend_labels[i_type]
                    + r" $P_{\text{CD}}="
                    + rf"{PCD_goal}$ ",
                    color_lines=colors_default[types],
                )

                lines = ax[i_type].get_lines()
                for line in lines:
                    line.set_color(colors_default[types])

                res_MDBS.append(MDBs)
                res_PCD.append(PCD_comp)
                # ax[i_type].set_xlim(-50, 50)
                # ax[i_type].set_ylim(-50, 50)
                # ax[i_type].set_xlim(-400, 400)
                # ax[i_type].set_ylim(-400, 400)
                ax[i_type].set_aspect(1)

                # ax[i_type].set_xlabel(r'$b_2$ [m]')
                ax[i_type].set_xlabel(r"$b_1$ [m]")

                # ax[i_type].set_xlabel(r'$b_1$ [m]')
                ax[i_type].set_title(
                    "MDBs for {} {}".format(
                        type_of_testing.replace("R_", "Reverse "), types
                    )
                )
                ax[i_type].grid("on")
    # ax[0].set_ylabel(r'$b_4$ [m]')
    ax[0].set_ylabel(r"$b_3$ [m]")
    ax[0].legend()
    # ax2.set_xlabel(r'$b_2$')
    # ax2.set_ylabel(r'$b_4$')
    # ax2.set_title('MDBs under {} for {}'.format(for_partition.replace('P', 'H'), type_of_testing.replace('R_', 'Reverse ')))
    # ax2.set_xlim(-6, 6)
    # ax2.set_ylim(-6, 6)
    # ax2.grid('on')
    # ax2.set_aspect(1)
    # ax2.legend()
    # Set labels, titles etc. on the left (main) subplot
    ax2_main.set_xlabel(r"$b_1$")
    ax2_main.set_ylabel(r"$b_3$")
    # ax2_main.set_xlabel(r'$b_2$')
    # ax2_main.set_ylabel(r'$b_4$')
    ax2_main.set_title(
        "MDBs under {} for {}, {}".format(
            for_partition.replace("P", "H"),
            type_of_testing.replace("R_", "Reverse "),
            type_of_example,
        )
    )
    # ax2_main.set_xlim(-6, 6)
    # ax2_main.set_ylim(-6, 6)
    ax2_main.grid(True)
    # ax2_main.set_aspect(1)
    plt.tight_layout()
    # ---- Define zoom region coordinates ----
    x1, x2 = 5, 11
    y1, y2 = -4, 0
    # x1, x2 = 2,5
    # y1, y2 = 0, 5

    # ---- Set limits for zoomed-in plot ----
    ax2_zoom.set_xlim(x1, x2)
    ax2_zoom.set_ylim(y1, y2)

    # ---- Draw rectangle on main plot ----
    rect = Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        linewidth=1.5,
        edgecolor="gray",
        linestyle="--",
        facecolor="none",
    )
    ax2_main.add_patch(rect)

    # ---- Connect corners with lines ----

    fig2.canvas.draw()  # Ensure all transforms are updated

    # Define rectangle in data coords
    corners_main_data = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    # Corresponding corners in zoom plot (normalized axes coordinates)
    corners_zoom_axes = [(0, 0), (1, 0), (1, 1), (0, 1)]

    # Convert corners to figure coordinates
    main_to_fig = ax2_main.transData + fig2.transFigure.inverted()
    zoom_to_fig = ax2_zoom.transAxes + fig2.transFigure.inverted()

    # for (xm, ym), (xz, yz) in zip(corners_main_data, corners_zoom_axes):
    #     x_fig_main, y_fig_main = main_to_fig.transform((xm, ym))
    #     x_fig_zoom, y_fig_zoom = zoom_to_fig.transform((xz, yz))

    #     line = mlines.Line2D(
    #         [x_fig_main, x_fig_zoom],
    #         [y_fig_main, y_fig_zoom],
    #         transform=fig2.transFigure,
    #         color='gray', linewidth=1.0
    #     )
    #     fig2.lines.append(line)

    # Titles and labels

    ax2_zoom.set_title(f"Zoomed in on [{x1}, {x2}] × [{y1}, {y2}]")
    ax2_zoom.grid("on")
    # ax2_main.legend()
    # ## Plot P0-OMT in the RIDS plot.

    # ax2_zoom, MDBs, PCD_comp = plot_MDB_surface('DS',
    #                                       'A',
    #                                       type_of_example,
    #                                       alpha_0, alpha_prime,
    #                                       type_of_alpha,
    #                                       for_partition,
    #                                       PCD_goal=0.7,
    #                                       q_i=2,
    #                                       ax=ax2_zoom,
    #                                       linestyle='-',
    #                                       legend_label= r'$\mathcal{P}_0$ OMT ' + r' $P_{\text{CD}}=' + rf'{0.7}$ ',
    #                                       color_lines = 'red' )
    # ax2_main, MDBs, PCD_comp = plot_MDB_surface('DS',
    #                                       'A',
    #                                       type_of_example,
    #                                       alpha_0, alpha_prime,
    #                                       type_of_alpha,
    #                                       for_partition,
    #                                       PCD_goal=0.7,
    #                                       q_i=2,
    #                                       ax=ax2_main,
    #                                       linestyle='-',
    #                                       legend_label= r'$\mathcal{P}_0$ OMT ' + r' $P_{\text{CD}}=' + rf'{0.7}$ ',
    #                                       color_lines = 'red' )
    ax2_zoom.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

    sys.exit()
    # %% Load data here

    type_of_example = "simple"
    # type_of_example='SPP_GNSS'
    alpha_0 = 0.01
    alpha_prime = 0.038
    type_of_alpha = "Kok_IDS"
    qmax = 2

    combs = []
    types_of_testing = ["DS", "IDS", "R_IDS", "classical DIA"]
    types_of_testing = ["DS", "R_IDS"]  # IDS has same P0 as DS types
    types_of_DS = ["A", "B", "C", "D"]
    pcd_goals = [0.7, 0.8, 0.9]
    for type_of_testing in types_of_testing:
        for type_of_DS in types_of_DS:
            if "IDS" in type_of_testing and type_of_DS == "D":
                pass
            elif type_of_testing == "DS" and type_of_DS in ["B", "D"]:
                pass
            else:
                combs.append([type_of_testing, type_of_DS])
            if type_of_testing == "classical DIA":
                break
        if type_of_testing == "classical DIA":
            break

    m = 7
    MDBs_res = np.zeros((3 * m, len(combs) + 1))
    mdir = r"C:\Users\bgvannoort\Documents\IDS\Results\MDBs\simple"
    for i in range(m):
        if i != 1:
            continue
        for i_PCD, pcd in enumerate(pcd_goals):
            hypt = "H" + str(i + 1)
            for i_col, combination in enumerate(combs):
                type_of_testing, type_of_DS = combination
                if type_of_testing != "classical DIA":
                    full_dir = os.path.join(
                        mdir,
                        hypt,
                        "PCD_goal={}".format(pcd),
                        type_of_testing,
                        type_of_DS,
                    )
                else:
                    full_dir = os.path.join(
                        mdir, hypt, "PCD_goal={}".format(pcd), type_of_testing
                    )
                fname = os.path.join(full_dir, "MDB_vectors_di_0.txt")
                data = np.loadtxt(fname)
                MDB = data[1]

                idx_to_fil = int(i) * 3 + i_PCD
                try:
                    MDBs_res[idx_to_fil, i_col + 1] = MDB
                except:
                    MDB = data[0, 1]
                    MDBs_res[idx_to_fil, i_col + 1] = MDB

            # determine the entry in the MIBS_res array.
    MDBs_res = np.round(MDBs_res, 2)
    sys.exit()
