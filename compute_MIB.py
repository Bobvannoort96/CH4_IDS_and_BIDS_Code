"""
Description of code:
    Compute the MIB for a given type of testing, type of data snooping
    and a given hypothesis.
    And a particular example type

Author:
    Bob van Noort
Date:
    25 April
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt

from Functions import *

import argparse
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.lines import Line2D

from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox, TransformedBbox
import matplotlib.lines as mlines


def compute_P_CI(
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
    idx_under_hypt,
):
    """
    Compute P_CI for a given bias magnitude.
    """
    # print('b_mag', b_mag)
    t_mean = (b_mag * cti_model @ d_vector).flatten()
    # print('t_mean', t_mean)
    # print('d_vector', d_vector)
    t_samples = scipy.stats.multivariate_normal.rvs(
        mean=t_mean, cov=Qtt, size=N_samples
    ).T

    # print('alpha_prime', alpha_prime)
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

    PCI_computed = np.sum(Pis == idx_under_hypt) / N_samples
    # print('computed PCI', PCI_computed, 'for b_mag', b_mag)

    return PCI_computed


def find_bias_b(
    PCI_goal,
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
    idx_under_hypt,
):
    """
    Bisection search for b_mag such that P_CI ≈ PCI_goal.
    """
    bmax0 = b_max
    bmin0 = b_min
    # print('find_bias_b: before PCI_min compute')
    PCI_min = compute_P_CI(
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
        idx_under_hypt,
    )
    # print('find_bias_b: after PCI min compute')
    PCI_max = compute_P_CI(
        b_max,
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
        idx_under_hypt,
    )

    for _ in range(max_iter):
        if PCI_min > PCI_goal:
            b_min = bmin0
            print("in loop PCImin > PCIgoal")
        if PCI_max < PCI_goal:
            print("in this loop")
            b_max = bmax0
        b_mid = 0.5 * (b_min + b_max)

        PCI_mid = compute_P_CI(
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
            idx_under_hypt,
        )

        diff = PCI_mid - PCI_goal

        if abs(diff) < tol:
            return b_mid, PCI_mid

        if diff > 0:  # PCI_mid is larger than the goal
            b_max = b_mid
            PCI_max = PCI_mid
        else:
            b_min = b_mid
            PCI_min = PCI_mid

        print("b_min, b_max", b_min, b_max)
        print("PCI_max, PCI_min", PCI_max, PCI_min)
        # print('PCI_mid, b_mid', PCI_mid, b_mid)
        # print('diff', diff)
        # print ('------------------------------------------------------------------------')
        # print('What goes into the PCI_mid function?')
        # for i_el, element in enumerate([b_mid, cti_model, d_vector, Qtt, N_samples,
        #                         compute_Pis, B_T, alpha_prime, type_of_testing,
        #                         type_of_DS, type_of_example, qmax, idx_under_hypt]):
        #     print('i_el, element', i_el, element)

    b_mag = 0.5 * (b_min + b_max)
    PCI_bmag = compute_P_CI(
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
        idx_under_hypt,
    )
    return b_mag, PCI_bmag


def compute_MIB_for_d_i(
    d_i,
    type_of_testing,
    type_of_DS,
    type_of_example,
    alpha_0,
    alpha_prime,
    type_of_alpha,
    for_partition,
    qmax,
    PCI_goal,
):
    """
    Computes MIB and PCI_computed for one direction index d_i.

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

    if type_of_testing == "DS" and q_i > 1:
        print("Requested is a hypothesis with q_i>1, while for type_of_DS=DS")
        print("We can never identify this hypothesis with DS. Return nans")
        return np.array([np.nan]).reshape(-1, 1), np.nan  # Cannot find good MIB

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
    # N_samples = int(1e5)

    # Initialize
    MIB_range = np.arange(0, 10) * 3
    # MIB_range = [6, 9, 12, 15]
    PCI_initialized = np.zeros(len(MIB_range))
    for i_mib, Mib in enumerate(MIB_range):
        t_mean = Mib * cti_model @ d_vector
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
        PCI_zeta = np.sum(Pis == idx_under_hypt) / N_samples
        if np.sum(Pis == idx_under_hypt) == 0:
            PCI_zeta = 1
        PCI_initialized[i_mib] = PCI_zeta
    print("PCI_initialized", PCI_initialized)

    where_larger_than_PCI_goal = PCI_initialized > PCI_goal
    if np.sum(where_larger_than_PCI_goal) == 0:
        return np.full(d_vector.shape, np.nan), np.nan  # Cannot find good MIB
        # pass

    b_max = MIB_range[where_larger_than_PCI_goal][0]
    b_min = MIB_range[~where_larger_than_PCI_goal][-1]

    print("Where_larger_PCI", where_larger_than_PCI_goal)
    print("bmin, bmax", b_min, b_max)

    # if b_mag > 150:
    #     d_b_mag = 0.5

    b_mag, PCI_computed = find_bias_b(
        PCI_goal,
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
        alpha_prime=alpha,
        type_of_testing=type_of_testing,
        type_of_DS=type_of_DS,
        type_of_example=type_of_example,
        qmax=qmax,
        idx_under_hypt=idx_under_hypt,
    )

    MIB_vector = b_mag * d_vector.flatten()
    return MIB_vector, PCI_computed


def plot_MIB_surface(
    type_of_testing,
    type_of_DS,
    type_of_example,
    alpha_0,
    alpha_prime,
    type_of_alpha,
    for_partition,
    PCI_goal,
    q_i,
    ax=None,
    linestyle="-",
    legend_label=None,
    color_lines=None,
):

    print(
        "Type_of_testing, type_of_DS, PCI_goal", type_of_testing, type_of_DS, PCI_goal
    )
    ## Load the original d_vectors array
    d_vectors = os.path.join(
        r"C:\Users\bgvannoort\Documents\IDS\Results\MIBs", "d_vectors_qi_2.txt"
    )

    if type_of_testing == "classical DIA":
        resDir = os.path.join(
            r"C:\Users\bgvannoort\Documents\IDS\Results\MIBs",
            type_of_example,
            for_partition.replace("P", "H"),
            "PCI_goal={}".format(PCI_goal),
            type_of_testing,
            alpha_string,
        )
        if legend_label is None:
            legend_label = "{}".format(type_of_testing)
    else:
        resDir = os.path.join(
            r"C:\Users\bgvannoort\Documents\IDS\Results\MIBs",
            type_of_example,
            for_partition.replace("P", "H"),
            "PCI_goal={}".format(PCI_goal),
            type_of_testing,
            type_of_DS,
            alpha_string,
        )
        if legend_label is None:
            legend_label = "{} {}".format(type_of_testing, type_of_DS)

    # Load MIB data
    MIBs = np.loadtxt(os.path.join(resDir, "MIB_vectors.txt"))
    print("Mibs.shape", MIBs.shape)
    # Load PCI_computed_data
    PCI_computed = np.loadtxt(os.path.join(resDir, "PCI_computed.txt"))
    print("PCI_computed.shape", PCI_computed.shape)

    for idx_PCI in range(PCI_computed.shape[0]):
        di, PCI_comp = PCI_computed[idx_PCI, :]
        # print(PCI_comp)
        if PCI_comp is not np.nan:
            if np.abs(PCI_comp - PCI_goal) > 0.01:
                print("At di = {}, the computed PCI = {}".format(di, PCI_comp))

    orders_PCI = PCI_computed[:, 0].astype(int)
    order_MIB = MIBs[:, 0].astype(int)
    # sys.exit()

    argsorted_PCI = np.argsort(orders_PCI)
    argsorted_MIB = np.argsort(order_MIB)

    PCI_computed = PCI_computed[
        argsorted_PCI, 1
    ]  # only last column really corresponds to PCI

    MIBs_ordered = MIBs[argsorted_MIB, 1:]

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
                MIBs_ordered[:, 0],
                MIBs_ordered[:, 1],
                label=legend_label,
                linestyle=linestyle,
                color=color_lines,
            )
        else:
            ax.plot(
                MIBs_ordered[:, 0],
                MIBs_ordered[:, 1],
                label=legend_label,
                linestyle=linestyle,
            )
        # ax.scatter(MIBs_ordered[:, 0], MIBs_ordered[:,1], label=legend_label.replace('_', ''), marker='.')

    return ax, MIBs_ordered, PCI_computed


# %%

if __name__ == "__main__":
    type_of_testing = "IDS"
    type_of_DS = "A"
    # type_of_example='simple'
    type_of_example = "Safoora_GNSS"
    alpha_0 = 0.01
    alpha_prime = 0.206
    type_of_alpha = "Kok_IDS"

    # for_partition = 'P24'
    for_partition = "P13"
    for_partition = "P1,3"

    ## Modify below to true if you want to compare with IDS or RIDS
    bool_compare_DIA_with_IDS = False
    type_of_testing_compare = "IDS"  # with which type do we want to compare?
    type_of_DS_compare = "C"  # with which type of DS do we want to compare?

    colors_default = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    linestyles_list = [":", "-.", "--", "-"]
    # linestyles_list = [  '-.', '-', '--']

    #
    # # type_of_example='simple'
    # alpha_0 = 0.01
    # alpha_prime = 0.01
    # type_of_alpha = 'Kok_IDS'
    # PCI_goal=0.6

    # for_partition = 'P13'
    # b1i = int(for_partition[1])
    # b2i = int(for_partition[2])

    # print('Starting')

    # alpha_string = 'type_of_alpha_{}alpha_0_{}'.format(type_of_alpha,alpha_0)
    # ## Get here an array, and load the data for use in the latex table
    # # For copying and pasting more easily.

    # types_of_testing = ['DS', 'IDS', 'R_IDS', 'classical DIA']
    # # types_of_testing = ['DS', 'IDS', 'R_IDS']
    # types_of_DS = ['A', 'B', 'C', 'D']
    # pci_goals = [0.6, 0.7, 0.8]

    # combs = []
    # for type_of_testing in types_of_testing:
    #     for type_of_DS in types_of_DS:

    #         if 'IDS' in type_of_testing and type_of_DS == 'D':
    #             pass
    #         else:
    #             combs.append([type_of_testing, type_of_DS])
    #         if type_of_testing== 'classical DIA':
    #             break
    #     if type_of_testing == 'classical DIA':
    #         break

    # m=7
    # MIBs_res = np.zeros((3*m, len(combs)+1 ))
    # if type_of_example == 'SPP_GNSS':
    #     mdir = r'C:\Users\bgvannoort\Documents\IDS\Results\MIBs\SPP_GNSS'
    # elif type_of_example == 'simple':
    #     mdir = r'C:\Users\bgvannoort\Documents\IDS\Results\MIBs\simple'
    # for i in range(m):
    #     # if i != 1:
    #     #     continue

    #     for i_PCI, pci in enumerate(pci_goals):
    #         hypt = 'H'+str(i+1)
    #         for i_col, combination in enumerate(combs):
    #             type_of_testing, type_of_DS = combination
    #             if type_of_testing != 'classical DIA':
    #                 full_dir = os.path.join(mdir, hypt, 'PCI_goal={}'.format(pci), type_of_testing, type_of_DS, alpha_string)
    #                 full_dir = os.path.join(mdir, hypt, 'PCI_goal={}'.format(pci), type_of_testing, type_of_DS)
    #             else:
    #                 full_dir = os.path.join(mdir, hypt, 'PCI_goal={}'.format(pci), type_of_testing, 'A', alpha_string)
    #             fname=os.path.join(full_dir, 'MIB_vectors_di_0.txt')
    #             data = np.loadtxt(fname)
    #             MIB = data[1]

    #             idx_to_fil = int(i) * 3 + i_PCI
    #             try:
    #                 MIBs_res[idx_to_fil, i_col+1] = MIB
    #             except:
    #                 MIB = data[0,1]
    #                 MIBs_res[idx_to_fil, i_col+1] = MIB

    #             MIBs_res[idx_to_fil, 0] = pci

    #         # determine the entry in the MIBS_res array.
    # MIBs_res = np.round(MIBs_res, 2)
    # sys.exit()

    alpha_string = "type_of_alpha_" + type_of_alpha + "alpha_0_" + str(alpha_0)

    plt.close("all")
    # os.makedirs(resDir, exist_ok=True)

    # MIB_row = np.concatenate(([d_i], MIB_vec.flatten()))
    # PCI_row = np.array([d_i, PCI_comp])

    #     # Save to text file
    # with open(os.path.join(resDir, 'MIB_vectors.txt'), 'a') as f_mib:
    #     np.savetxt(f_mib, MIB_row.reshape(1, -1), fmt='%.6e')

    # with open(os.path.join(resDir, 'PCI_computed.txt'), 'a') as f_pci:
    #     np.savetxt(f_pci, PCI_row.reshape(1, -1), fmt='%.6e')
    plt.rcParams["font.size"] = 16
    res_MIBS = []
    res_PCI = []

    # PCI_goals = [ 0.6, 0.7, 0.8]
    PCI_goals = [0.5, 0.6, 0.7, 0.8]

    if type_of_testing != "classical DIA":
        fig, ax = plt.subplots(1, 3, figsize=(19, 6.63))
        fig2, (ax_main, ax_zoom) = plt.subplots(1, 2, figsize=(14.66, 7.14))
        fig134, ax123 = plt.subplots()
        for idx_PCI_goal, PCI_goal in enumerate(PCI_goals):
            for i_type, types in enumerate(["A"]):  # , 'B', 'C']):
                ax[i_type], MIBs, PCI_comp = plot_MIB_surface(
                    type_of_testing,
                    types,
                    type_of_example,
                    alpha_0,
                    alpha_prime,
                    type_of_alpha,
                    for_partition,
                    PCI_goal,
                    q_i=2,
                    ax=ax[i_type],
                    linestyle=linestyles_list[idx_PCI_goal],
                    legend_label=r" $P_{\text{CI}}=" + rf"{PCI_goal}$ ",
                    color_lines=colors_default[i_type],
                )
                ax123, MIBs, PCI_comp = plot_MIB_surface(
                    type_of_testing,
                    types,
                    type_of_example,
                    alpha_0,
                    alpha_prime,
                    type_of_alpha,
                    for_partition,
                    PCI_goal,
                    q_i=2,
                    ax=ax123,
                    linestyle=linestyles_list[idx_PCI_goal],
                    legend_label=r" $P_{\text{CI}}=" + rf"{PCI_goal}$ ",
                    color_lines=colors_default[i_type],
                )
                ax_main, MIBs, PCI_comp = plot_MIB_surface(
                    type_of_testing,
                    types,
                    type_of_example,
                    alpha_0,
                    alpha_prime,
                    type_of_alpha,
                    for_partition,
                    PCI_goal,
                    q_i=2,
                    ax=ax_main,
                    linestyle=linestyles_list[idx_PCI_goal],
                    legend_label=rf"Type {types}"
                    + r" $P_{\text{CI}}="
                    + rf"{PCI_goal}$ ",
                    color_lines=colors_default[i_type],
                )
                ax_zoom, MIBs, PCI_comp = plot_MIB_surface(
                    type_of_testing,
                    types,
                    type_of_example,
                    alpha_0,
                    alpha_prime,
                    type_of_alpha,
                    for_partition,
                    PCI_goal,
                    q_i=2,
                    ax=ax_zoom,
                    linestyle=linestyles_list[idx_PCI_goal],
                    legend_label=rf"Type {types}"
                    + r" $P_{\text{CI}}="
                    + rf"{PCI_goal}$ ",
                    color_lines=colors_default[i_type],
                )

                # collections = ax[i_type].collections[0]
                # collections.set_facecolor(colors_default[i_type])
                lines = ax[i_type].get_lines()
                for line in lines:
                    line.set_color(colors_default[i_type])

                res_MIBS.append(MIBs)
                res_PCI.append(PCI_comp)
                # ax[i_type].set_xlim(-50, 50)
                # ax[i_type].set_ylim(-50, 50)
                ax[i_type].set_xlim(-400, 400)
                ax[i_type].set_ylim(-400, 400)
                # ax[i_type].set_aspect(1)

                ax[i_type].set_xlabel(r"$b_1$ [m]")
                ax[i_type].set_title(
                    "MIBs for {} {}".format(
                        type_of_testing.replace("R_", "Reverse "), types
                    )
                )
                ax[i_type].grid("on")
        ax[0].set_ylabel(r"$b_3$ [m]")
        ax[0].legend()
        # ax2.set_xlabel(r'$b_1$')
        # ax2.set_ylabel(r'$b_3$')
        # ax2.set_title('MIBs under {} for {}'.format(for_partition.replace('P', 'H'), type_of_testing))
        # ax2.set_xlim(-50, 50)
        # ax2.set_ylim(-50, 50)
        # ax2.legend()
        plt.tight_layout()
        ax_main.set_xlim(-200, 200)
        ax_main.set_ylim(-80, 80)
        # ax_main.set_xlim(-40, 40)
        # ax_main.set_ylim(-40, 40)
        # ax_main.set_xlim(-75, 75)
        # ax_main.set_ylim(-75, 75)
        # ax_main.set_aspect(1)

        ax_main.set_xlim(-10, 10)
        ax_main.set_ylim(-10, 10)
        ax123.set_xlim(-10, 10)
        ax123.set_ylim(-10, 10)
        ax123.set_xlabel(r"$b_1$ [m]")
        ax123.set_ylabel(r"$b_3$ [m]")
        ax123.set_title(
            "MIBs for {} for H{}".format(
                type_of_testing, for_partition.replace("P", "")
            )
        )

        # ax_main.set_xlabel(r'$b_2$ [m]')
        # ax_main.set_ylabel(r'$b_4$ [m]')
        ax_main.set_xlabel(r"$b_1$ [m]")
        ax_main.set_ylabel(r"$b_3$ [m]")
        ax_main.set_title(
            "MIBs for {} for H{}".format(
                type_of_testing, for_partition.replace("P", "")
            )
        )

        # ---- Define zoom region coordinates ----
        x1, x2 = 40, 80
        y1, y2 = -45, -10
        # x1, x2 = 0, 20
        # y1, y2 = 0, 20

        # ---- Set limits for zoomed-in plot ----
        ax_zoom.set_xlim(x1, x2)
        ax_zoom.set_ylim(y1, y2)

        plt.tight_layout()
        # ---- Draw rectangle on main plot ----
        rect = Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=3,
            edgecolor="gray",
            linestyle="--",
            facecolor="none",
        )
        ax_main.add_patch(rect)

        # ---- Connect corners with lines ----
        # Coordinates in display (screen) space
        fig.canvas.draw()  # Need this to get accurate transforms

        # Convert data to display coordinates
        transform = ax_main.transData.transform
        inv_transform = ax_zoom.transAxes.inverted().transform

        # Corners of the rectangle in main plot
        corners_main = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        # # Opposite corners of the zoom plot (normalized axes coords)
        # corners_zoom = [(0, 0), (1, 0), (1, 1), (0, 1)]

        # for (xm, ym), (xz, yz) in zip(corners_main, corners_zoom):
        #     start_disp = transform((xm, ym))
        #     start_fig = fig.transFigure.inverted().transform(start_disp)

        #     line = mlines.Line2D(
        #         [start_fig[0], ax_zoom.get_position().x0 + xz * ax_zoom.get_position().width],
        #         [start_fig[1], ax_zoom.get_position().y0 + yz * ax_zoom.get_position().height],
        #         transform=fig.transFigure, color="gray", linewidth=0.8
        #     )
        #     fig.lines.append(line)

        # ax_zoom.set_xticklabels([])
        # ax_zoom.set_yticklabels([])
        ax_zoom.grid("on")
        ax_main.grid("on")

        ax_main.set_title(
            "MIBs under {} for {}".format(
                for_partition.replace("P", "H"), type_of_testing
            )
        )
        ax_zoom.set_title(rf"Zoomed in on [{x1}, {x2}] $\times$ [{y1}, {y2}]")

        # Custom legend handles: only linestyles matter
        custom_lines = [
            Line2D([0], [0], color="black", linestyle=ls, lw=2)
            for ls in linestyles_list
        ]
        labels_PCI = [
            r"$P_{\text{CI}}=0.6$",
            r"$P_{\text{CI}}=0.7$",
            r"$P_{\text{CI}}=0.8$",
        ]
        labels_PCI = [
            r"$P_{\text{CI}}=0.5$",
            r"$P_{\text{CI}}=0.6$",
            r"$P_{\text{CI}}=0.7$",
            r"$P_{\text{CI}}=0.8$",
        ]

        ax_zoom.legend(
            custom_lines,
            labels_PCI,
            title=r"$P_{\text{CI}}$ values",
            loc="upper left",
            bbox_to_anchor=(0.0, 0.8, 0.2, 0.2),
        )
        ax123.legend(
            custom_lines,
            labels_PCI,
            title=r"$P_{\text{CI}}$ values",
            loc="upper left",
            bbox_to_anchor=(0.0, 0.8, 0.2, 0.2),
        )
        ax_zoom.set_aspect(1)
        plt.show()
        print("HERE??")

        if bool_compare_DIA_with_IDS:
            custom_lines_axmain = [
                Line2D([0], [0], color=colors_default[0], linestyle="-", lw=2),
                Line2D([0], [0], color="black", linestyle="-", lw=2),
            ]
            custom_labels = [
                "{} {}".format(
                    type_of_testing_compare.replace("R_", "Reverse "),
                    type_of_DS_compare,
                ),
                "trad. DIA",
            ]
            ax_main.legend(custom_lines_axmain, custom_labels, loc="upper left")
            ax_main.set_title(
                "MIBs under {} for trad. DIA and {} A".format(
                    for_partition.replace("P", "H"),
                    type_of_testing_compare.replace("R_", "Reverse "),
                )
            )

    else:
        # fig, (ax_main, ax_zoom) = plt.subplots(1, 2, figsize=(19, 5.63), gridspec_kw={'width_ratios': [2, 1]})
        fig, (ax_main, ax_zoom) = plt.subplots(1, 2, figsize=(14.66, 7.14))
        for idx_PCI_goal, PCI_goal in enumerate(PCI_goals):
            ax_main, MIBs, PCI_comp = plot_MIB_surface(
                type_of_testing,
                "A",
                type_of_example,
                alpha_0,
                alpha_prime,
                type_of_alpha,
                for_partition,
                PCI_goal,
                q_i=2,
                ax=ax_main,
                linestyle=linestyles_list[idx_PCI_goal],
                legend_label=r" $P_{\text{CI}}=" + rf"{PCI_goal}$ ",
                color_lines="black",
            )
            ax_zoom, MIBs, PCI_comp = plot_MIB_surface(
                type_of_testing,
                "A",
                type_of_example,
                alpha_0,
                alpha_prime,
                type_of_alpha,
                for_partition,
                PCI_goal,
                q_i=2,
                ax=ax_zoom,
                linestyle=linestyles_list[idx_PCI_goal],
                legend_label=r" $P_{\text{CI}}=" + rf"{PCI_goal}$ ",
                color_lines="black",
            )

            res_MIBS.append(MIBs)
            res_PCI.append(PCI_comp)

            if bool_compare_DIA_with_IDS:
                ax_main, MIBs, PCI_comp = plot_MIB_surface(
                    type_of_testing_compare,
                    type_of_DS_compare,
                    type_of_example,
                    alpha_0,
                    alpha_prime,
                    type_of_alpha,
                    for_partition,
                    PCI_goal,
                    q_i=2,
                    ax=ax_main,
                    linestyle=linestyles_list[idx_PCI_goal],
                    legend_label=r" $P_{\text{CI}}=" + rf"{PCI_goal}$ ",
                    color_lines=colors_default[0],
                )
                ax_zoom, MIBs, PCI_comp = plot_MIB_surface(
                    type_of_testing_compare,
                    type_of_DS_compare,
                    type_of_example,
                    alpha_0,
                    alpha_prime,
                    type_of_alpha,
                    for_partition,
                    PCI_goal,
                    q_i=2,
                    ax=ax_zoom,
                    linestyle=linestyles_list[idx_PCI_goal],
                    legend_label=r" $P_{\text{CI}}=" + rf"{PCI_goal}$ ",
                    color_lines=colors_default[0],
                )

        lines = ax_main.get_lines()[0]
        lines.set_color("black")

        ax_main.set_xlim(-100, 100)
        # ax_main.set_ylim(-80, 80)
        # ax_main.set_xlim(-40, 40)
        ax_main.set_ylim(-40, 40)
        # ax_main.set_xlim(-75, 75)
        # ax_main.set_ylim(-75, 75)
        # ax_main.set_aspect(1)

        # ax_main.set_xlabel(r'$b_2$ [m]')
        # ax_main.set_ylabel(r'$b_4$ [m]')
        ax_main.set_xlabel(r"$b_1$ [m]")
        ax_main.set_ylabel(r"$b_3$ [m]")
        ax_main.set_title(
            "MIBs for {} for H{}".format(
                type_of_testing, for_partition.replace("P", "")
            )
        )

        # ---- Define zoom region coordinates ----
        x1, x2 = 40, 80
        y1, y2 = -45, -10
        # x1, x2 = 0, 20
        # y1, y2 = 0, 20

        # ---- Set limits for zoomed-in plot ----
        ax_zoom.set_xlim(x1, x2)
        ax_zoom.set_ylim(y1, y2)

        plt.tight_layout()
        # ---- Draw rectangle on main plot ----
        rect = Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=3,
            edgecolor="gray",
            linestyle="--",
            facecolor="none",
        )
        ax_main.add_patch(rect)

        # ---- Connect corners with lines ----
        # Coordinates in display (screen) space
        fig.canvas.draw()  # Need this to get accurate transforms

        # Convert data to display coordinates
        transform = ax_main.transData.transform
        inv_transform = ax_zoom.transAxes.inverted().transform

        # Corners of the rectangle in main plot
        corners_main = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        # # Opposite corners of the zoom plot (normalized axes coords)
        # corners_zoom = [(0, 0), (1, 0), (1, 1), (0, 1)]

        # for (xm, ym), (xz, yz) in zip(corners_main, corners_zoom):
        #     start_disp = transform((xm, ym))
        #     start_fig = fig.transFigure.inverted().transform(start_disp)

        #     line = mlines.Line2D(
        #         [start_fig[0], ax_zoom.get_position().x0 + xz * ax_zoom.get_position().width],
        #         [start_fig[1], ax_zoom.get_position().y0 + yz * ax_zoom.get_position().height],
        #         transform=fig.transFigure, color="gray", linewidth=0.8
        #     )
        #     fig.lines.append(line)

        # ax_zoom.set_xticklabels([])
        # ax_zoom.set_yticklabels([])
        ax_zoom.grid("on")
        ax_main.grid("on")

        ax_main.set_title(
            "MIBs under {} for {}".format(
                for_partition.replace("P", "H"), type_of_testing
            )
        )
        ax_zoom.set_title(rf"Zoomed in on [{x1}, {x2}] $\times$ [{y1}, {y2}]")

        # Custom legend handles: only linestyles matter
        custom_lines = [
            Line2D([0], [0], color="black", linestyle=ls, lw=2)
            for ls in linestyles_list
        ]
        # labels_PCI = [r'$P_{\text{CI}}=0.6$', r'$P_{\text{CI}}=0.7$', r'$P_{\text{CI}}=0.8$']
        labels_PCI = [
            r"$P_{\text{CI}}=0.5$",
            r"$P_{\text{CI}}=0.6$",
            r"$P_{\text{CI}}=0.7$",
            r"$P_{\text{CI}}=0.8$",
        ]

        ax_zoom.legend(
            custom_lines,
            labels_PCI,
            title=r"$P_{\text{CI}}$ values",
            loc="upper left",
            bbox_to_anchor=(0.0, 0.8, 0.2, 0.2),
        )
        ax_zoom.set_aspect(1)
        plt.show()
        print("HERE??")

        if bool_compare_DIA_with_IDS:
            custom_lines_axmain = [
                Line2D([0], [0], color=colors_default[0], linestyle="-", lw=2),
                Line2D([0], [0], color="black", linestyle="-", lw=2),
            ]
            custom_labels = [
                "{} {}".format(
                    type_of_testing_compare.replace("R_", "Reverse "),
                    type_of_DS_compare,
                ),
                "trad. DIA",
            ]
            ax_main.legend(custom_lines_axmain, custom_labels, loc="upper left")
            ax_main.set_title(
                "MIBs under {} for trad. DIA and {} A".format(
                    for_partition.replace("P", "H"),
                    type_of_testing_compare.replace("R_", "Reverse "),
                )
            )

    # A zoomin when type_of_example is 'simple'
    if type_of_example == "simple":
        if type_of_testing != "classical DIA":
            axins = inset_axes(ax2, width="40%", height="40%", loc="lower right")
            for idx_PCI_goal, PCI_goal in enumerate(PCI_goals):
                for i_type, types in enumerate(["A", "B", "C"]):
                    # axins = plot_MIB_surface(type_of_testing, types, type_of_example, alpha_0, alpha_prime, type_of_alpha, for_partition, q_i=2, ax=axins)
                    axins, _, _ = plot_MIB_surface(
                        type_of_testing,
                        types,
                        type_of_example,
                        alpha_0,
                        alpha_prime,
                        type_of_alpha,
                        for_partition,
                        q_i=2,
                        ax=axins,
                        PCI_goal=PCI_goal,
                        linestyle=linestyles_list[idx_PCI_goal],
                        color_lines=colors_default[i_type],
                    )
            axins.set_xlim(-12, -6)
            axins.set_ylim(-7, -3)
            # axins.set_aspect(1)
            axins.set_xticklabels([])
            axins.set_yticklabels([])
            axins.grid("on")
            ax2.grid("on")
            mark_inset(ax2, axins, loc1=1, loc2=3, fc="none", ec="0.5")

    sys.exit()
    # %% Plot here the comparison with IDS and RIDS in one plot

    for i_type, types in enumerate(["A", "B", "C"]):
        fig2, (ax_main, ax_zoom) = plt.subplots(1, 2, figsize=(14.66, 7.14))
        for idx_PCI_goal, PCI_goal in enumerate(PCI_goals):
            # Give IDS the color blue and RIDS orange
            ax_main, MIBs, PCI_comp = plot_MIB_surface(
                "IDS",
                types,
                type_of_example,
                alpha_0,
                alpha_prime,
                type_of_alpha,
                for_partition,
                PCI_goal,
                q_i=2,
                ax=ax_main,
                linestyle=linestyles_list[idx_PCI_goal],
                legend_label=rf"Type {types}" + r" $P_{\text{CI}}=" + rf"{PCI_goal}$ ",
                color_lines="blue",
            )
            ax_zoom, MIBs, PCI_comp = plot_MIB_surface(
                "IDS",
                types,
                type_of_example,
                alpha_0,
                alpha_prime,
                type_of_alpha,
                for_partition,
                PCI_goal,
                q_i=2,
                ax=ax_zoom,
                linestyle=linestyles_list[idx_PCI_goal],
                legend_label=rf"Type {types}" + r" $P_{\text{CI}}=" + rf"{PCI_goal}$ ",
                color_lines="blue",
            )

            ax_main, MIBs, PCI_comp = plot_MIB_surface(
                "R_IDS",
                types,
                type_of_example,
                alpha_0,
                alpha_prime,
                type_of_alpha,
                for_partition,
                PCI_goal,
                q_i=2,
                ax=ax_main,
                linestyle=linestyles_list[idx_PCI_goal],
                legend_label=rf"Type {types}" + r" $P_{\text{CI}}=" + rf"{PCI_goal}$ ",
                color_lines="orange",
            )
            ax_zoom, MIBs, PCI_comp = plot_MIB_surface(
                "R_IDS",
                types,
                type_of_example,
                alpha_0,
                alpha_prime,
                type_of_alpha,
                for_partition,
                PCI_goal,
                q_i=2,
                ax=ax_zoom,
                linestyle=linestyles_list[idx_PCI_goal],
                legend_label=rf"Type {types}" + r" $P_{\text{CI}}=" + rf"{PCI_goal}$ ",
                color_lines="orange",
            )

        ax_main.set_xlim(-200, 200)
        ax_main.set_ylim(-80, 80)
        # ax_main.set_xlim(-40, 40)
        # ax_main.set_ylim(-40, 40)
        # ax_main.set_xlim(-75, 75)
        # ax_main.set_ylim(-75, 75)
        # ax_main.set_aspect(1)

        # ax_main.set_xlabel(r'$b_2$ [m]')
        # ax_main.set_ylabel(r'$b_4$ [m]')
        ax_main.set_xlabel(r"$b_1$ [m]")
        ax_main.set_ylabel(r"$b_3$ [m]")
        ax_main.set_title(
            "MIBs for {} for H{}".format(
                type_of_testing, for_partition.replace("P", "")
            )
        )

        # ---- Define zoom region coordinates ----
        x1, x2 = 40, 80
        y1, y2 = -45, -10
        # x1, x2 = 0, 20
        # y1, y2 = 0, 20

        # ---- Set limits for zoomed-in plot ----
        ax_zoom.set_xlim(x1, x2)
        ax_zoom.set_ylim(y1, y2)

        plt.tight_layout()
        # ---- Draw rectangle on main plot ----
        rect = Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=3,
            edgecolor="gray",
            linestyle="--",
            facecolor="none",
        )
        ax_main.add_patch(rect)

        # ---- Connect corners with lines ----
        # Coordinates in display (screen) space
        fig.canvas.draw()  # Need this to get accurate transforms

        # Convert data to display coordinates
        transform = ax_main.transData.transform
        inv_transform = ax_zoom.transAxes.inverted().transform

        # Corners of the rectangle in main plot
        corners_main = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        # ax_zoom.set_xticklabels([])
        # ax_zoom.set_yticklabels([])
        ax_zoom.grid("on")
        ax_main.grid("on")

        ax_main.set_title(
            "MIBs under {} for IDS vs RIDS".format(for_partition.replace("P", "H"))
        )
        ax_zoom.set_title(rf"Zoomed in on [{x1}, {x2}] $\times$ [{y1}, {y2}]")

        # Custom legend handles: only linestyles matter
        custom_lines = [
            Line2D([0], [0], color="black", linestyle=ls, lw=2)
            for ls in linestyles_list
        ]
        labels_PCI = [
            r"$P_{\text{CI}}=0.6$",
            r"$P_{\text{CI}}=0.7$",
            r"$P_{\text{CI}}=0.8$",
        ]
        labels_PCI = [
            r"$P_{\text{CI}}=0.5$",
            r"$P_{\text{CI}}=0.6$",
            r"$P_{\text{CI}}=0.7$",
            r"$P_{\text{CI}}=0.8$",
        ]

        ax_zoom.legend(
            custom_lines,
            labels_PCI,
            title=r"$P_{\text{CI}}$ values",
            loc="upper left",
            bbox_to_anchor=(0.0, 0.8, 0.2, 0.2),
        )
        ax_zoom.set_aspect(1)
        plt.show()
        print("HERE??")

        custom_lines_axmain = [
            Line2D([0], [0], color="blue", linestyle="-", lw=2),
            Line2D([0], [0], color="orange", linestyle="-", lw=2),
        ]
        custom_labels = ["{} {}".format("IDS", types), "{} {}".format("RIDS", types)]
        ax_main.legend(custom_lines_axmain, custom_labels, loc="upper left")
