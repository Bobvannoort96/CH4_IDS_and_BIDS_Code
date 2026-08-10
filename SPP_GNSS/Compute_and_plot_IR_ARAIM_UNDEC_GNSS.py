"""
Description of code:
    Plot and Compute the total IR for a particular test procedure given a
    scenario.
    scenario_nr can be:
        - None --> default settings are used, alpha_0=0.01, beta_0=0.2 and
                   alpha_prime follows from B-method (=0.038)
        - 1 --> alpha_0 is chosen such that alpha_prime = PFA = P(H0)
            P(H0) follows from considering all Hi with qi <=2 and psat=1e-3
        - 2 --> same as 1 but with psat=1e-2
Author:
    Bob van Noort
Date:
    06-09-2025

"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import glob
import sys
import random
import re
import pandas as pd

sys.path.append(r"C:\Users\bgvannoort\Documents\IDS\Code")

from matplotlib.lines import Line2D
import pickle

from matplotlib.patches import Patch

from Functions import load_indices_partitions_GNSS_ex, setup_SPP_GNSS_example
from Functions import load_setup_parameters, get_idx_hypt_RIDS
from Plot_PCI_PWI_results_from_DelftBlue import (
    get_data,
    plot_PCD_PCI_together,
    plot_PCI,
    extract_bval,
)

from Plot_PCI_PWI_results_2Dhypts_from_DelftBlue import get_data_2Dhypt

# %%
random.seed(300)
type_of_example = "ARAIM_UNDEC_GNSS"
qmax = 2
alpha_type = "manual"

# alpha_type = 'manual'

colors_radii_IR = plt.rcParams["axes.prop_cycle"].by_key()["color"]

scenario_nr = "sameP0_OMT"  #
if scenario_nr == 1:
    alpha_0_dictionary = {
        "R_IDS_C": 0.0015,
        "R_IDS_B": 0.001317,
        "R_IDS_A": 0.001,
        "IDS_A": 0.001317,
        "IDS_B": 0.001317,
        "IDS_C": 0.0012,
        "classical DIA": 0.001317,
        "DS_A": 0.001317,
        "DS_B": 0.001317,
        "DS_C": 0.0012,
        "DS_D": 0.001317,
    }
    psat = 1e-3
elif scenario_nr == 2:
    alpha_0_dictionary = {
        "R_IDS_C": 0.017,
        "R_IDS_B": 0.02041,
        "R_IDS_A": 0.014,
        "IDS_A": 0.02041,
        "IDS_B": 0.02041,
        "IDS_C": 0.015,
        "classical DIA": 0.02041,
        "DS_A": 0.02041,
        "DS_B": 0.02041,
        "DS_C": 0.015,
        "DS_D": 0.02041,
    }
    psat = 1e-2
elif scenario_nr == 3:
    alpha_0_dictionary = {
        "R_IDS_C": 0.176,
        "R_IDS_B": 0.365,
        "R_IDS_A": 0.176,
        "IDS_A": 0.365,
        "IDS_B": 0.365,
        "IDS_C": 0.162,
        "classical DIA": 0.365,
        "DS_A": 0.365,
        "DS_B": 0.365,
        "DS_C": 0.162,
        "DS_D": 0.365,
    }
    psat = 1e-01
elif scenario_nr == "sameP0_OMT":
    alpha_0_dictionary = {
        "R_IDS_C": 0.01,
        "R_IDS_B": 0.01,
        "R_IDS_A": 0.01,
        "IDS_A": 0.01,
        "IDS_B": 0.01,
        "IDS_C": 0.01,
        "classical DIA": 0.01,
        "DS_A": 0.01,
        "DS_B": 0.01,
        "DS_C": 0.01,
        "DS_D": 0.01,
    }


plt.rcParams["font.size"] = 16
plt.rcParams.update(
    {
        "font.size": 16,  # General font size
        "axes.titlesize": 16,  # Title font
        "axes.labelsize": 20,  # X/Y label font
        "xtick.labelsize": 15,  # X tick labels
        "ytick.labelsize": 15,  # Y tick labels
        "legend.fontsize": 15,  # Legend font
        "figure.titlesize": 16,  # Figure title (if used)
    }
)
combinations = []
# for type_of_testing in ['classical DIA',  'IDS', 'R_IDS', 'DS']:
for type_of_testing in ["IDS", "R_IDS", "classical DIA"]:
    for type_of_DS in ["A", "B", "C"]:
        for pp in range(7):
            part_hypt = "P" + str(pp + 1)
            combinations.append([type_of_testing, type_of_DS, part_hypt])
        if type_of_testing == "classical DIA":
            break

close_plots = True  # Set to True if you want to close plots after every 'combination'
bool_close_all_at_start = (
    True  # Set to True if you want to close all opened plots at the start
)

bool_larger_bias = False
bool_normalized_biases_PCDPCI = False  # boolean, whether we want to plot the bi-axis on normalized scale, divided by the std of the observation.

bool_store_IRs = True
store_IRs_list = []

if bool_close_all_at_start:
    plt.close("all")

# combinations = [['R_IDS', 'A', 'P3']]#,['R_IDS', 'B', 'P5'],['R_IDS', 'C', 'P5']]

combinations = [["IDS", "A", "P3"], ["R_IDS", "A", "P3"], ["classical DIA", "B", "P3"]]
type_T_, type_DS_ = ",", ","

for comb in combinations:
    # plt.close('all')
    type_of_testing, type_of_DS, part_hypt = comb
    if type_of_testing == "classical DIA":
        alpha_0 = alpha_0_dictionary[type_of_testing]
    else:
        alpha_0 = alpha_0_dictionary[type_of_testing + "_" + type_of_DS]

    print("alpha_0", alpha_0)

    m, n, r, A, alpha, sigma, Qyy, Qyy_inv, B_T, Qtt, Qtt_inv = load_setup_parameters(
        type_of_example, alpha_method=alpha_type, alpha_0=alpha_0
    )

    q_i = 1

    indices_partitions, colors_indices = load_indices_partitions_GNSS_ex(
        False, load_colors=True
    )
    alpha_prime_string = "alpha_prime={}".format(np.round(alpha, 3))
    ## Check if dir exists for writing the results to for PCI/PWI PCD/PMD etc.

    if scenario_nr is None:
        main_dir = (
            r"C:\Users\bgvannoort\Documents\IDS\Results\TestingProbabilities\{}".format(
                type_of_example
            )
        )
    elif scenario_nr == "sameP0_OMT":
        main_dir = r"C:\Users\bgvannoort\Documents\IDS\Results\TestingProbabilities\{}\{}".format(
            type_of_example, scenario_nr
        )
    else:
        main_dir = r"C:\Users\bgvannoort\Documents\IDS\Results\TestingProbabilities\{}\scenario_nr_{}".format(
            type_of_example, scenario_nr
        )

    hypt_idx = indices_partitions[part_hypt]

    if type_of_testing != "classical DIA":
        fulldir = os.path.join(
            main_dir,
            type_of_testing,
            type_of_DS,
            alpha_type,
            f"alpha_0_{alpha_0}",
            "Hypothesis{}".format(hypt_idx),
            "qmax={}".format(qmax),
        )
        if q_i == 1:
            fig_save_dir = os.path.join(
                main_dir,
                "Figures_testing_probabilities",
                type_of_testing,
                type_of_DS,
                alpha_type,
                "alpha_0_" + str(alpha_0),
            )
        elif q_i == 2:
            fig_save_dir = os.path.join(
                main_dir,
                "Figures_testing_probabilities",
                type_of_testing,
                type_of_DS,
                alpha_type,
                "alpha_0_" + str(alpha_0),
                "heatmaps",
            )
        testing_string = type_of_testing + "_" + type_of_DS
    else:
        fulldir = os.path.join(
            main_dir,
            type_of_testing,
            alpha_prime_string,
            "Hypothesis{}".format(hypt_idx),
        )

        if q_i == 1:
            fig_save_dir = os.path.join(
                main_dir,
                "Figures_testing_probabilities",
                alpha_prime_string,
                type_of_testing,
            )
        elif q_i == 2:
            fig_save_dir = os.path.join(
                main_dir,
                "Figures_testing_probabilities",
                alpha_prime_string,
                type_of_testing,
                "heatmaps",
            )
        testing_string = type_of_testing

    if bool_larger_bias:
        fig_save_dir = os.path.join(fig_save_dir, "larger_biases")
        xlim = 100

    if not os.path.exists(fig_save_dir):
        os.makedirs(fig_save_dir)

    # load / get the data
    (
        mean_data_PCA,
        std_data_PCA,
        bvals_PCA,
        cols_PCA,
        mean_data_IR,
        std_data_IR,
        bvals_IR,
        cols_IR,
        radii_IR,
    ) = get_data(fulldir)

    # Per combination of type_of_testing and DS, we only have to execute this once...
    if type_T_ == type_of_testing and type_DS_ == type_of_DS:
        pass

    else:
        plot_PCD_PCI_together(
            type_of_testing,
            type_of_DS,
            alpha_0,
            type_of_example,
            alpha_type=alpha_type,
            bool_normalized_biases_PCDPCI=bool_normalized_biases_PCDPCI,
            colors_PCD_PCI_plot=colors_indices,
            main_dir=main_dir,
        )

        type_T_ = type_of_testing
        type_DS_ = type_of_DS

    # %%
    # Construct here the corresponding labels from the data. From the column_names
    labels_PCA = []
    partitions = []
    for el in cols_PCA:
        if el == "PCA_MD":
            labels_PCA.append(r"$\mathcal{P}_0$")
            partitions.append("P0")
        elif el == "b_i":
            pass
        elif el == "P_OMEGA":
            labels_PCA.append(r"$\mathcal{P}_{\Omega}$")
            partitions.append("P99")
        else:
            zeta = el.split("_CI_")
            part = zeta[-1].replace(",", "")
            labels_PCA.append(r"$\mathcal{P}_" + rf"{{{part}}}$")
            partitions.append("P" + part)

    fig, ax = plt.subplots(1, 2, figsize=(19.2, 9.83))  # for the testing probabilities
    # Loop over all the components
    for idx_el, (lbl, partition) in enumerate(zip(labels_PCA, partitions)):
        if partition != "P99":  # not equal to undecided region
            if len(partition) == 3:  # we are dealing with q=2 hypothesis identification
                ax[0].plot(
                    bvals_PCA,
                    mean_data_PCA[:, idx_el],
                    label=lbl,
                    color=colors_indices[partition],
                    linewidth=2,
                    linestyle="--",
                )
                ax[1].plot(
                    bvals_PCA,
                    std_data_PCA[:, idx_el],
                    label=lbl,
                    color=colors_indices[partition],
                    linewidth=2,
                    linestyle="--",
                )
            else:
                ax[0].plot(
                    bvals_PCA,
                    mean_data_PCA[:, idx_el],
                    label=lbl,
                    color=colors_indices[partition],
                    linewidth=2,
                )
                ax[1].plot(
                    bvals_PCA,
                    std_data_PCA[:, idx_el],
                    label=lbl,
                    color=colors_indices[partition],
                    linewidth=2,
                )

        else:
            # last entry in mean_data equals the undec region
            ax[0].plot(
                bvals_PCA,
                mean_data_PCA[:, -1],
                label=lbl,
                color=colors_indices[partition],
                linewidth=2,
            )
            ax[1].plot(
                bvals_PCA,
                std_data_PCA[:, -1],
                label=lbl,
                color=colors_indices[partition],
                linewidth=2,
            )

    ax[0].set_xlabel("$b_" + f"{part_hypt.replace('P', '')}$ [m]")
    ax[0].set_ylabel("P [-]")
    ax[0].set_title("Testing probabilities under H" + part_hypt.replace("P", ""))
    ax[0].set_yscale("log")
    ax[0].set_ylim(1e-5, 1.0)

    ax[0].set_xlim(np.min(bvals_PCA), np.max(bvals_PCA))
    ax[0].grid("on")

    ax[1].set_xlabel(r"$b_" + f"{part_hypt.replace('P', '')}$ [m]")
    ax[1].set_ylabel(r"$\sigma_P$ [-]")
    ax[1].set_ylim(1e-6, 1e-2)
    ax[1].set_xlim(np.min(bvals_PCA), np.max(bvals_PCA))
    # ax[1].legend()
    ax[1].legend(loc="upper right", ncol=6, bbox_to_anchor=(1.05, 1.05))
    ax[1].set_yscale("log")
    ax[1].grid("on")

    if not bool_larger_bias:
        if np.max(bvals_PCA) > 50:
            ax[0].set_xlim(0, 50)
            ax[1].set_xlim(0, 50)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            fig_save_dir, "PCA_CI_WI_under_H{}".format(part_hypt.replace("P", ""))
        )
    )

    # Compute Pomega as a correction factor for the IR
    Pomega = mean_data_PCA[:, -1]

    ## For the IR probabilities
    fig, ax = plt.subplots(1, 2, figsize=(19.2, 9.83))
    for idx_el, lbl_rad in enumerate(radii_IR):
        label = "R = {} m".format(lbl_rad)
        ax[0].plot(
            bvals_IR,
            mean_data_IR[:, idx_el] / (1 - Pomega),
            label=label,
            color=colors_radii_IR[idx_el],
            linewidth=2,
        )
        ax[1].plot(
            bvals_IR,
            std_data_IR[:, idx_el] / np.sqrt(1 - Pomega),
            label=label,
            color=colors_radii_IR[idx_el],
            linewidth=2,
            linestyle="--",
        )

    ax[0].set_xlabel("$b_" + f"{part_hypt.replace('P', '')}$ [m]")
    ax[0].set_ylabel("P [-]")
    ax[0].set_title(
        r"$P(\bar{x} \notin \mathcal{B}_x \mid \mathcal{H}_"
        + rf"{{{part_hypt.replace('P', '')}}})$ for {testing_string}"
    )
    ax[0].set_yscale("log")
    ax[0].set_ylim(1e-5, 1.0)
    ax[0].set_xlim(np.min(bvals_IR), np.max(bvals_IR))
    ax[0].grid("on")

    ax[1].set_xlabel(rf"$b_{{{part_hypt.replace('P', '')}}}$ [m]")
    ax[1].set_ylabel(r"$\sigma_P$ [-]")
    ax[1].set_ylim(1e-6, 1e-2)
    ax[1].set_xlim(np.min(bvals_IR), np.max(bvals_IR))
    # ax[1].legend()
    ax[1].legend(loc="upper right", ncols=2, bbox_to_anchor=(1.05, 1.05))
    ax[1].set_yscale("log")
    ax[1].grid("on")

    if not bool_larger_bias:
        if np.max(bvals_IR) > 50:
            ax[0].set_xlim(0, 50)
            ax[1].set_xlim(0, 50)
    plt.tight_layout()

    plt.savefig(
        os.path.join(fig_save_dir, "IR_under_H{}".format(part_hypt.replace("P", "")))
    )
    if close_plots:
        plt.close("all")
    # sys.exit()

    if bool_store_IRs:
        # save also the actual IR values in a list
        store_IRs_list.append([type_of_testing, type_of_DS, part_hypt, mean_data_IR])

    # ---- NEW: Single-plot IR figure ----
    fig_single, ax_single = plt.subplots(figsize=(9.6, 6.0))
    for idx_el, lbl_rad in enumerate(radii_IR):
        label = "R = {} m".format(lbl_rad)
        ax_single.plot(
            bvals_IR,
            mean_data_IR[:, idx_el] / (1 - Pomega),
            label=label,
            color=colors_radii_IR[idx_el],
            linewidth=2,
        )

    ax_single.set_xlabel("$b_" + f"{part_hypt.replace('P', '')}$ [m]")
    ax_single.set_ylabel("P [-]")
    ax_single.set_title(
        r"$P(\bar{x} \notin \mathcal{B}_x \mid \mathcal{H}_"
        + rf"{{{part_hypt.replace('P', '')}}})$ for {testing_string}"
    )
    ax_single.set_yscale("log")
    ax_single.set_ylim(1e-5, 1.0)
    ax_single.set_xlim(np.min(bvals_IR), np.max(bvals_IR))
    ax_single.grid("on")
    ax_single.legend(loc="upper right", ncol=2)

    if not bool_larger_bias:
        if np.max(bvals_IR) > 50:
            ax_single.set_xlim(0, 50)

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            fig_save_dir, "IR_only_under_H{}".format(part_hypt.replace("P", ""))
        )
    )


# %% Combine the plots above, that produce the total summed up probabilities of outlier identifications (wrong and correct taken together)
# Just as done with the Safoora example
def plot_summed_components(
    types_of_testing, hypt_idx, bool_normalized_biases_PCDPCI=False
):
    type_of_DS = "A"
    linestyle_dict = {"R_IDS": "-", "IDS": "--", "classical DIA": "-."}

    # qmax=3 here, we cannot loop over all the components --> ~1300 lines in one plot...
    # Instead, we make a distinction between PMD, Pomega, P_WI_j(qj=1), PWI_j(qj=2), PWI_j(qj=3) and PCI.
    # Construct here the corresponding labels from the data. From the column_names

    # make a combined figure with all types_of_testing in one plot
    fig_combined, ax_combined = plt.subplots(figsize=(19.2, 9.83))
    for type_of_testing in types_of_testing:
        fig, ax = plt.subplots(figsize=(19.2, 9.83))
        if type_of_testing == "classical DIA":
            alpha_0 = alpha_0_dictionary[type_of_testing]
        else:
            alpha_0 = alpha_0_dictionary[type_of_testing + "_" + type_of_DS]

        print("alpha_0", alpha_0)

        m, n, r, A, alpha, sigma, Qyy, Qyy_inv, B_T, Qtt, Qtt_inv = (
            load_setup_parameters(
                type_of_example, alpha_method=alpha_type, alpha_0=alpha_0
            )
        )

        alpha_prime_string = "alpha_prime={}".format(np.round(alpha, 3))
        ## Check if dir exists for writing the results to for PCI/PWI PCD/PMD etc.

        if scenario_nr is None:
            main_dir = r"C:\Users\bgvannoort\Documents\IDS\Results\TestingProbabilities\{}".format(
                type_of_example
            )
        elif scenario_nr == "sameP0_OMT":
            main_dir = r"C:\Users\bgvannoort\Documents\IDS\Results\TestingProbabilities\{}\{}".format(
                type_of_example, scenario_nr
            )
        else:
            main_dir = r"C:\Users\bgvannoort\Documents\IDS\Results\TestingProbabilities\{}\scenario_nr_{}".format(
                type_of_example, scenario_nr
            )

        q_i = 1

        if type_of_testing != "classical DIA":
            fulldir = os.path.join(
                main_dir,
                type_of_testing,
                type_of_DS,
                alpha_type,
                f"alpha_0_{alpha_0}",
                "Hypothesis{}".format(hypt_idx),
                "qmax={}".format(qmax),
            )
            if q_i == 1:
                fig_save_dir = os.path.join(
                    main_dir,
                    "Figures_testing_probabilities",
                    type_of_testing,
                    type_of_DS,
                    alpha_type,
                    "alpha_0_" + str(alpha_0),
                )
            elif q_i == 2:
                fig_save_dir = os.path.join(
                    main_dir,
                    "Figures_testing_probabilities",
                    type_of_testing,
                    type_of_DS,
                    alpha_type,
                    "alpha_0_" + str(alpha_0),
                    "heatmaps",
                )
            testing_string = type_of_testing + "_" + type_of_DS
        else:
            fulldir = os.path.join(
                main_dir,
                type_of_testing,
                alpha_prime_string,
                "Hypothesis{}".format(hypt_idx),
            )

            if q_i == 1:
                fig_save_dir = os.path.join(
                    main_dir,
                    "Figures_testing_probabilities",
                    alpha_prime_string,
                    type_of_testing,
                )
            elif q_i == 2:
                fig_save_dir = os.path.join(
                    main_dir,
                    "Figures_testing_probabilities",
                    alpha_prime_string,
                    type_of_testing,
                    "heatmaps",
                )
            testing_string = type_of_testing

        fig_save_dir_combined = r"C:\Users\bgvannoort\Documents\IDS\Results\TestingProbabilities\ARAIM_UNDEC_GNSS\sameP0_OMT\Figures_testing_probabilities\RIDS_vs_IDS_vs_DIA"
        if not os.path.exists(fig_save_dir):
            os.makedirs(fig_save_dir)
        os.makedirs(fig_save_dir_combined, exist_ok=True)

        # load / get the data
        (
            mean_data_PCA,
            std_data_PCA,
            bvals_PCA,
            cols_PCA,
            mean_data_IR,
            std_data_IR,
            bvals_IR,
            cols_IR,
            radii_IR,
        ) = get_data(fulldir)

        sigma_i = np.sqrt(np.diag(Qyy)[hypt_idx - 1])

        if bool_normalized_biases_PCDPCI:
            bvals_PCA = bvals_PCA / sigma_i
            ax.set_xlabel(rf"$\frac{{b_{{{hypt_idx}}}}}{{\sigma_{{{hypt_idx}}}}}$ [m]")
        else:
            ax.set_xlabel("$b_" + f"{{{hypt_idx}}}$ [m]")

        labels_PCA = []
        partitions = []
        label_CI = "P_FA_{}_CI_{}".format(hypt_idx, hypt_idx)
        idx_correct = []  # indices in cols_PCA where at least obs P{i} is identified, though with other (healthy) obs
        idx_wrong = []  # indices in cols_PCA where P{i} is not identified at all.
        for idx, el in enumerate(cols_PCA):
            if el == "PCA_MD":
                labels_PCA.append(r"$\mathcal{P}_0$")
                partitions.append("P0")
            elif el == "b_i":
                pass
            elif el == "P_OMEGA":
                labels_PCA.append(r"$\mathcal{P}_{\Omega}$")
                partitions.append("P99")
            elif el == label_CI:
                zeta = el.split("_CI_")
                part = zeta[-1].replace(",", "")
                labels_PCA.append(r"$\mathcal{P}_" + rf"{{{part}}}$")
                partitions.append("P" + part)
                idx_PCI_in_data_array = idx - 1
            else:
                zeta = el.split("_CI_")
                part = zeta[-1].split(",")

                if str(hypt_idx) in part:
                    # Identification of outlier is at least present.

                    idx_correct.append(
                        idx - 1
                    )  # minus one because the first one in the list is b_i!
                else:
                    idx_wrong.append(idx - 1)

        labels_PCA.append(
            r"$\sum_{j} \mathcal{P}_{S_j}$" + rf" for ${hypt_idx} \in S_j$"
        )
        partitions.append("P_WI_correct_ID")

        labels_PCA.append(
            r"$\sum_{j} \mathcal{P}_{S_j}$" + rf" for ${hypt_idx} \notin S_j$"
        )
        partitions.append("P_WI_wrong_ID")
        # we need to also make a new color_indices array
        #         labels_PCA = [
        #     r'$\mathcal{P}_0$',
        #     r'$\mathcal{P}_1$',
        #     r'$\mathcal{P}_{\Omega}$',
        #     r'$\mathcal{P}_{\sum_j \mathcal{P}_{S_j}} \text{ for } 1 \in S_j$',
        #     r'$\mathcal{P}_{\sum_j \mathcal{P}_{S_j}} \text{ for } 1 \notin S_j$'
        # ]
        colors_list = ["grey", "green", "black", "lightgreen", "red"]
        colors_indices = dict(zip(partitions, colors_list))

        for idx_el, (lbl, partition) in enumerate(zip(labels_PCA, partitions)):
            if partition == "P_WI_correct_ID":  # not equal to undecided region
                ax.plot(
                    bvals_PCA,
                    np.sum(mean_data_PCA[:, idx_correct], axis=1),
                    label=lbl,
                    color=colors_indices[partition],
                    linewidth=2,
                    linestyle="--",
                )
                ax_combined.plot(
                    bvals_PCA,
                    np.sum(mean_data_PCA[:, idx_correct], axis=1),
                    label=lbl,
                    color=colors_indices[partition],
                    linewidth=2,
                    linestyle=linestyle_dict[type_of_testing],
                )
                # ax[1].plot(bvals_PCA, std_data_PCA[:, idx_el], label=lbl, color=colors_indices[partition], linewidth=2, linestyle='--')
            elif partition == "P_WI_wrong_ID":
                ax.plot(
                    bvals_PCA,
                    np.sum(mean_data_PCA[:, idx_wrong], axis=1),
                    label=lbl,
                    color=colors_indices[partition],
                    linewidth=2,
                )
                ax_combined.plot(
                    bvals_PCA,
                    np.sum(mean_data_PCA[:, idx_wrong], axis=1),
                    label=lbl,
                    color=colors_indices[partition],
                    linewidth=2,
                    linestyle=linestyle_dict[type_of_testing],
                )
                # ax[1].plot(bvals_PCA, std_data_PCA[:, idx_el], label=lbl, color=colors_indices[partition], linewidth=2)

            elif partition == "P99":
                # last entry in mean_data equals the undec region
                ax.plot(
                    bvals_PCA,
                    mean_data_PCA[:, -1],
                    label=lbl,
                    color=colors_indices[partition],
                    linewidth=2,
                )
                ax_combined.plot(
                    bvals_PCA,
                    mean_data_PCA[:, -1],
                    label=lbl,
                    color=colors_indices[partition],
                    linewidth=2,
                    linestyle=linestyle_dict[type_of_testing],
                )
                # ax[1].plot(bvals_PCA, std_data_PCA[:, -1], label=lbl, color=colors_indices[partition], linewidth=2)
            elif partition == "P" + str(hypt_idx):
                ax.plot(
                    bvals_PCA,
                    mean_data_PCA[:, idx_PCI_in_data_array],
                    label=lbl,
                    color=colors_indices[partition],
                    linewidth=2,
                )
                ax_combined.plot(
                    bvals_PCA,
                    mean_data_PCA[:, idx_PCI_in_data_array],
                    label=lbl,
                    color=colors_indices[partition],
                    linewidth=2,
                    linestyle=linestyle_dict[type_of_testing],
                )
            elif partition == "P0":
                ax.plot(
                    bvals_PCA,
                    mean_data_PCA[:, 0],
                    label=lbl,
                    color=colors_indices[partition],
                    linewidth=2,
                )
                ax_combined.plot(
                    bvals_PCA,
                    mean_data_PCA[:, 0],
                    label=lbl,
                    color=colors_indices[partition],
                    linewidth=2,
                    linestyle=linestyle_dict[type_of_testing],
                )

        ax.set_ylabel("P [-]")
        ax.set_title(
            rf"Testing probabilities under $H_{{{hypt_idx}}}$"
            + rf" for {testing_string.replace('_', ' ')} "
        )
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1.0)
        ax.set_xlim(np.min(bvals_PCA), np.max(bvals_PCA))
        ax.grid("on")
        ax.legend(loc="upper right", bbox_to_anchor=(1.05, 1.05))

        fig.tight_layout()
        os.makedirs(
            os.path.join(fig_save_dir, "Sum_of_outlier_identifications"), exist_ok=True
        )

        if bool_normalized_biases_PCDPCI:
            fig.savefig(
                os.path.join(
                    fig_save_dir,
                    "Sum_of_outlier_identifications",
                    "PCA_CI_WI_under_H{}_normalized_b_axis".format(hypt_idx),
                )
            )
        else:
            fig.savefig(
                os.path.join(
                    fig_save_dir,
                    "Sum_of_outlier_identifications",
                    "PCA_CI_WI_under_H{}".format(hypt_idx),
                )
            )

    ## make a manual legend
    labels_handles = []

    for color, label in zip(colors_list, labels_PCA):
        labels_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle=None,
                lw=0,
                marker="o",
                markersize=8,
                label=label,
            )
        )

    for type_of_testing in types_of_testing:
        lbl = type_of_testing.replace("R_", "R ")
        lbl = lbl.replace("classical ", "")
        labels_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                linestyle=linestyle_dict[type_of_testing],
                label=lbl,
            )
        )

    if bool_normalized_biases_PCDPCI:
        ax_combined.set_xlabel(
            rf"$\frac{{b_{{{hypt_idx}}}}}{{\sigma_{{{hypt_idx}}}}}$ [m]"
        )
        ax_combined.set_xlim(np.min(bvals_PCA), 20)
    else:
        ax_combined.set_xlabel("$b_" + f"{{{hypt_idx}}}$ [m]")
        ax_combined.set_xlim(np.min(bvals_PCA), 50)

    ax_combined.set_ylabel("P [-]")
    ax_combined.set_title("Test probabilities for ARAIM GNSS example")
    ax_combined.set_yscale("log")
    ax_combined.set_ylim(1e-5, 1.0)

    ax_combined.grid("on")
    ax_combined.legend(
        handles=labels_handles, loc="upper right", bbox_to_anchor=(1.05, 1.05)
    )

    fig_combined.tight_layout()
    os.makedirs(
        os.path.join(fig_save_dir_combined, "Sum_of_outlier_identifications"),
        exist_ok=True,
    )

    if bool_normalized_biases_PCDPCI:
        fig_combined.savefig(
            os.path.join(
                fig_save_dir_combined,
                "Sum_of_outlier_identifications",
                "PCA_CI_WI_under_H{}_combined_plot_normalized_bi_axis".format(hypt_idx),
            )
        )
    else:
        fig_combined.savefig(
            os.path.join(
                fig_save_dir_combined,
                "Sum_of_outlier_identifications",
                "PCA_CI_WI_under_H{}_combined_plot".format(hypt_idx),
            )
        )

    return fig_combined, ax_combined


types_of_testing = ["R_IDS", "IDS", "classical DIA"]

f, axx = plot_summed_components(
    types_of_testing, hypt_idx=6, bool_normalized_biases_PCDPCI=True
)
# %%


def compare_test_procedures(alpha_type_compare="Kok_IDS"):
    compare_types = ["R_IDS", "IDS", "classical DIA"]
    type_of_DS_compare = "A"
    linestyle_dict = {"R_IDS": "-", "IDS": "--", "classical DIA": "-."}
    bool_normalized_biases = False

    under_Hypt = "3"  # this is under actual H{actual_Hypt}

    fig_PCI_comb, ax_PCI_combined = plt.subplots(figsize=(19.2, 9.83))
    bool_normalized_biases_PCI = False

    for type_of_testing_compare in compare_types:
        if type_of_testing_compare == "classical DIA":
            alpha_0_compare = alpha_0_dictionary[type_of_testing_compare]
        else:
            alpha_0_compare = alpha_0_dictionary[
                type_of_testing_compare + "_" + type_of_DS_compare
            ]

        ax_PCI_combined, labels_PCI_combined = plot_PCI(
            type_of_testing_compare,
            type_of_DS_compare,
            alpha_0_compare,
            type_of_example,
            alpha_type=alpha_type_compare,
            bool_normalized_biases_PCI=bool_normalized_biases_PCI,
            colors_PCI_plot=colors_indices,
            ax=ax_PCI_combined,
            main_dir=main_dir,
            bool_normalized_biases_PCDPCI=True,
        )

    for type_of_testing_compare in compare_types:
        label_to_use = type_of_testing_compare.replace("R_", "R")
        label_to_use = label_to_use.replace("classical ", "")
        labels_PCI_combined.append(
            Line2D(
                [0],
                [0],
                color="black",
                linestyle=linestyle_dict[type_of_testing_compare],
                linewidth=2,
                label=label_to_use,
            )
        )

    ax_PCI_combined.legend(
        handles=labels_PCI_combined,
        ncols=3,
        loc="lower right",
        frameon=True,
        edgecolor="black",
        borderpad=0.35,
        labelspacing=0.3,
        handlelength=2,
        handletextpad=0.3,
        columnspacing=0.5,
    )
    ax_PCI_combined.set_title(
        r"$P_{\mathrm{CI}_i}$ probabilities for"
        + rf" {type_of_example}, $\alpha'={np.round(alpha, 3)}$ "
    )

    return fig_PCI_comb, ax_PCI_combined


fig_comb, ax_comb = compare_test_procedures(alpha_type_compare=alpha_type)


# %%
## Make here a compare plot for IDS vs RIDS vs Trad DIA (for example)
def plot_IR_vs_bias_under_Hi(hypt_i=3):
    compare_types = ["R_IDS", "IDS", "classical DIA"]
    type_of_DS_compare = "A"
    linestyle_dict = {"R_IDS": "-", "IDS": "--", "classical DIA": "-."}
    bool_normalized_biases = False
    alpha_prime = alpha
    under_Hypt = str(hypt_i)  # this is under actual H{actual_Hypt}
    fig, ax = plt.subplots(figsize=(19.2, 9.83))

    sigma_i = (np.diag(Qyy)[int(under_Hypt) - 1]) ** (0.5)
    for type_of_testing_compare in compare_types:
        if type_of_testing_compare == "classical DIA":
            alpha_0_compare = alpha_0_dictionary[type_of_testing_compare]
        else:
            alpha_0_compare = alpha_0_dictionary[
                type_of_testing_compare + "_" + type_of_DS_compare
            ]

        if type_of_testing_compare == "classical DIA":
            newdir_compare = os.path.join(
                main_dir,
                type_of_testing_compare,
                "alpha_prime={}".format(np.round(alpha_prime, 3)),
                "Hypothesis{}".format(under_Hypt),
            )
        elif type_of_testing_compare == "R_IDS":
            newdir_compare = os.path.join(
                main_dir,
                type_of_testing_compare,
                type_of_DS_compare,
                alpha_type,
                "alpha_0_" + str(alpha_0_compare),
                "Hypothesis{}".format(under_Hypt),
                f"qmax={qmax}",
            )
        elif type_of_testing_compare == "IDS":
            newdir_compare = os.path.join(
                main_dir,
                type_of_testing_compare,
                type_of_DS_compare,
                alpha_type,
                "alpha_0_" + str(alpha_0_compare),
                "Hypothesis{}".format(under_Hypt),
                f"qmax={qmax}",
            )
        print("Newdir, directory taken for the data")
        print(newdir_compare)
        (
            mean_data_PCA_c,
            std_data_PCA_c,
            bvals_PCA_c,
            cols_PCA_c,
            mean_data_IR_c,
            std_data_IR_c,
            bvals_IR_c,
            cols_IR_c,
            radii_IR_c,
        ) = get_data(newdir_compare)

        if bool_normalized_biases:
            bias_to_plot = bvals_IR_c / sigma_i

            # ax.set_xlim(0,np.max(bvals_IR_c))
            ax.set_xlabel(
                rf"$\frac{{b_{{{under_Hypt}}}}}{{\sigma_{{{under_Hypt}}}}}$ [m]"
            )
        else:
            bias_to_plot = bvals_IR_c

            # ax.set_xlim(0,np.max(bvals_IR_c))
            ax.set_xlabel("$b_" + f"{{{under_Hypt}}}$ [m]")

        Pomega = mean_data_PCA_c[:, -1]
        for idx_el, lbl_rad in enumerate(radii_IR_c):
            label = "R = {} m".format(lbl_rad)
            ax.plot(
                bias_to_plot,
                mean_data_IR_c[:, idx_el] / (1 - Pomega),
                label=label,
                color=colors_radii_IR[idx_el],
                linewidth=2,
                linestyle=linestyle_dict[type_of_testing_compare],
            )

    ax.set_xlim(0, np.max(bias_to_plot))
    ax.set_ylabel("P [-]")

    ax.set_title(
        r"$P(\bar{x} \notin \mathcal{B}_x \mid \mathcal{H}_"
        + r"{{{}}})$ for {} vs {} vs {}".format(under_Hypt, *compare_types)
    )
    ax.set_yscale("log")

    ax.set_ylim(1e-5, 1.0)

    custom_lines_axmain = [
        Line2D([0], [0], color="black", linestyle="-", lw=2),
        Line2D([0], [0], color="black", linestyle="--", lw=2),
        Line2D([0], [0], color="black", linestyle="-.", lw=2),
    ]
    custom_labels = [
        "RIDS {}".format(type_of_DS_compare),
        "IDS {}".format(type_of_DS_compare),
        "traditional DIA",
    ]
    for idx_el, lbl_rad in enumerate(radii_IR_c):
        label = "R = {} m".format(lbl_rad)
        custom_lines_axmain.append(
            Line2D(
                [0],
                [0],
                color=colors_radii_IR[idx_el],
                linestyle=None,
                lw=0,
                marker="o",
                markersize=8,
            )
        )
        custom_labels.append(label)

    ax.legend(custom_lines_axmain, custom_labels, loc="lower right")
    ax.grid("on")
    return fig, ax


for hypt in range(1, m + 1):
    fig, ax = plot_IR_vs_bias_under_Hi(hypt_i=hypt)
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            r"C:\Users\bgvannoort\Documents\IDS\Results\TestingProbabilities\ARAIM_UNDEC_GNSS\sameP0_OMT\Figures_testing_probabilities\IDS_vs_RIDS_vs_DIA",
            "combined_IR_under_H_{}.png".format(hypt),
        )
    )


# %% Compare the IR for different test procedures. We can compare it based on a particular bval=X times sigma, or by considering the worst case


def plot_IR_vs_bias(type_of_DS_IR, psat):

    ## Plot the IR vs bias (one-dimensional)
    ## Also compute the total IR, based on either a maximum IR approach (select maximum conditional (IR | H_i))
    ## or based on computing the IR at a given b_i value (also in 2D)

    PHi = psat * (1 - psat) ** (m - 1)
    PHij = psat**2 * (1 - psat) ** (m - 2)
    linestyle_dict = {"R_IDS": "-", "IDS": "--", "classical DIA": "-."}
    type_of_alpha_IR = "manual"
    colors_radii_IR = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    PH0 = 1 - m * PHi - (m * (m - 1) / 2) * PHij  # Assuming only q=1 hypotheses

    P_H_list = [PH0] + m * [PHi]

    evaluationMethod = (
        "maximum"  # The method for computing the IR assigned to an H with q=2
    )
    use_lower_biases = True  # for comparison with trad DIA, we should not go into 'numerical instability' territory.

    types_of_testing = ["R_IDS", "IDS", "classical DIA"]
    # types_of_testing = ['IDS', 'classical DIA']

    print("At IR computations")
    fig, ax_IR = plt.subplots(figsize=(18.8, 9.83))  # 3 subplots in a row
    results_IR_tot = []
    all_IR_data = np.zeros((len(types_of_testing), int(m) + int(m * (m - 1) / 2)))
    # Reload the data
    for type_of_testing_IR in types_of_testing:
        if type_of_testing_IR != "R_IDS":
            alpha_prime_string = "alpha_prime={}".format(np.round(alpha, 3))

        if type_of_testing_IR == "classical DIA":
            alpha_0_IR = alpha_0_dictionary[type_of_testing_IR]
        else:
            alpha_0_IR = alpha_0_dictionary[type_of_testing_IR + "_" + type_of_DS_IR]

        if type_of_testing_IR != "classical DIA":
            fulldir = os.path.join(
                main_dir,
                type_of_testing_IR,
                type_of_DS_IR,
                type_of_alpha_IR,
                f"alpha_0_{alpha_0_IR}",
                "Hypothesis{}",
                f"qmax={qmax}",
            )
            fig_save_dir = os.path.join(
                main_dir,
                "Figures_testing_probabilities",
                type_of_testing_IR,
                type_of_DS_IR,
                type_of_alpha_IR,
                f"alpha_0_{alpha_0_IR}",
            )
            if q_i == 2:
                fig_save_dir = os.path.join(fig_save_dir, "heatmaps")
            testing_string = f"{type_of_testing_IR}_{type_of_DS_IR}"
            title_string = rf"Total $P(\bar{{x}} \notin \mathcal{{B}}_x)$ for {type_of_testing_IR} type {type_of_DS_IR}"
        else:
            fulldir = os.path.join(
                main_dir, type_of_testing_IR, alpha_prime_string, "Hypothesis{}"
            )
            fig_save_dir = os.path.join(
                main_dir,
                "Figures_testing_probabilities",
                alpha_prime_string,
                type_of_testing_IR,
            )
            if q_i == 2:
                fig_save_dir = os.path.join(fig_save_dir, "heatmaps")
            testing_string = type_of_testing_IR
            title_string = (
                rf"Total $P(\bar{{x}} \notin \mathcal{{B}}_x)$ for {type_of_testing_IR}"
            )

        if not os.path.exists(fig_save_dir):
            os.makedirs(fig_save_dir)

        print("Fulldir used:")
        print(fulldir)

        IR_tot = 0
        IR_tot_all_H = (
            0  # becomes an nr_of_Rx_list; just numbers not a function of bias
        )
        r_penalty = 0
        bias_loc_penalty = 3
        max_IRs = []

        # Loop over the q=1 hypotheses
        for hypt in np.arange(1, m + 1):
            (
                mean_data_PCA,
                std_data_PCA,
                bvals_PCA,
                cols_PCA,
                mean_data_IR,
                std_data_IR,
                bvals_IR,
                cols_IR,
                radii_IR,
            ) = get_data(fulldir.format(hypt))

            if use_lower_biases:
                mean_data_PCA = mean_data_PCA[:50, :]
                std_data_PCA = std_data_PCA[:50, :]
                bvals_PCA = bvals_PCA[:50]

                mean_data_IR = mean_data_IR[:50, :]
                std_data_IR = std_data_IR[:50, :]
                bvals_IR = bvals_IR[:50]

            IR_tot += mean_data_IR * P_H_list[hypt]
            max_IRs.append(np.max(mean_data_IR, axis=0))

            b_val_pen = bias_loc_penalty * Qyy[hypt - 1, hypt - 1] ** 0.5
            (idx_for_pen,) = np.where(np.array(bvals_PCA) - b_val_pen > 0)
            idx_for_pen = idx_for_pen[0]
            PWI_tot_bval_pen = 1 - mean_data_PCA[idx_for_pen, hypt]
            r_penalty += PWI_tot_bval_pen * P_H_list[hypt]

            # Evaluate here for the inclusion of all q=1 and q=2 hypotheses, based on evaluationMethod
            if evaluationMethod == "maximum":
                IR_to_add = np.max(mean_data_IR, axis=0)
            elif evaluationMethod == "bias":
                raise NotImplementedError("Method 'bias' not yet implemented")
            # print('shape of IR_to_add', IR_to_add.shape)
            # print('shape of PH0', PH0)
            # print('Shape of P_HI_list', len(P_H_list))
            IR_tot_all_H += P_H_list[hypt] * IR_to_add

        IR_tot += (
            PH0
            * np.ones(IR_tot.shape[0]).reshape(-1, 1)
            @ mean_data_IR[0, :].reshape(1, -1)
        )
        IR_tot_all_H += PH0 * mean_data_IR[0, :]
        r_penalty += PH0 * (1 - mean_data_PCA[0, 0])
        IR_under_H0 = mean_data_IR[0, :] * PH0

        print(f"IR_under_H0 for {type_of_testing}", IR_under_H0)

        print(f"Mean or total penalty for {type_of_testing_IR} is {r_penalty}")

        for idx_r, r_x in enumerate(radii_IR):
            # Plot the IR curve
            ax_IR.plot(
                bvals_IR,
                IR_tot[:, idx_r],
                label=rf"$R_x = {r_x}$ m",
                color=colors_radii_IR[idx_r],
                linestyle=linestyle_dict[type_of_testing_IR],
            )

            # H0 reference line
            # ax_IR.axhline(IR_under_H0[idx_r], color=colors_radii_IR[idx_r], linestyle=':')

        # ax_IR.set_title(rf'$R_x = {el_r}$ m')
        ax_IR.set_xlabel("$b_i$ [m]")
        ax_IR.set_ylabel(r"$P_{IR}$ [-]")
        ax_IR.set_yscale("log")
        ax_IR.set_ylim(1e-5, 1.0)
        ax_IR.set_xlim(0, np.max(bvals_IR))
        ax_IR.grid("on")

        ## The contribution of the 2D hypotheses
        # b1value range
        b1_range = np.arange(-20, 21, 1)

        # b2value range -- necessary only for the shaping of the arrays. Default is 0 to 20.
        b2_range = np.arange(0, 21, 1)
        for hypt in range(8, 29):
            ## This is all for one hypothesis
            print("At H{}".format(hypt))
            for idxb1, b1 in enumerate(b1_range):
                if idxb1 == 0:
                    # Initialize the data arrays for the IR
                    # Initialize the data arrays for testing
                    mean_data_PCA_all_2D = np.zeros(
                        (len(b1_range), len(b2_range), len(cols_PCA) - 1)
                    )
                    std_data_PCA_all_2D = np.zeros(
                        (len(b1_range), len(b2_range), len(cols_PCA) - 1)
                    )

                    mean_data_IR_all_2D = np.zeros(
                        (len(b1_range), len(b2_range), len(cols_IR) - 1)
                    )
                    std_data_IR_all_2D = np.zeros(
                        (len(b1_range), len(b2_range), len(cols_IR) - 1)
                    )

                if use_lower_biases and np.abs(b1) > 10:
                    continue  # go to the next b1 value

                # load / get the data
                # print("B1=", b1)
                (
                    mean_data_PCA_2D,
                    std_data_PCA_2D,
                    bvals_PCA_2D,
                    cols_PCA_2D,
                    mean_data_IR_2D,
                    std_data_IR_2D,
                    bvals_IR_2D,
                    cols_IR_2D,
                    radii_IR_2D,
                ) = get_data_2Dhypt(fulldir.format(hypt), b1)

                for idxb2, b2 in enumerate(bvals_PCA_2D):
                    if use_lower_biases and np.abs(b2) > 10:
                        continue

                    # store the data
                    mean_data_PCA_all_2D[idxb1, idxb2, :] = mean_data_PCA_2D[idxb2, :]
                    std_data_PCA_all_2D[idxb1, idxb2, :] = std_data_PCA_2D[idxb2, :]

                for idxb2, b2 in enumerate(bvals_IR_2D):
                    mean_data_IR_all_2D[idxb1, idxb2, :] = mean_data_IR_2D[idxb2, :]
                    std_data_IR_all_2D[idxb1, idxb2, :] = std_data_IR_2D[idxb2, :]

            idx_PCI = hypt_idx

            PCI_mean = mean_data_PCA_all_2D[:, :, idx_PCI]
            PWI_mean = (
                mean_data_PCA_all_2D.sum(axis=-1) - PCI_mean
            )  # total wrong identification probability including MD

            if evaluationMethod == "maximum":
                r_penalty += PHij * np.max(PWI_mean)
            elif evaluationMethod == "bias":
                raise NotImplementedError("Method 'bias' not yet implemented")

            # Loop over the radii_IR_2D
            for idx_r, r_x in enumerate(radii_IR_2D):
                IR_data_r_x = mean_data_IR_all_2D[:, :, idx_r]
                if evaluationMethod == "maximum":
                    IR_to_add = np.max(IR_data_r_x)

                elif evaluationMethod == "bias":
                    raise NotImplementedError("Method 'bias' not yet implemented")
                IR_tot_all_H[idx_r] += PHij * IR_to_add

        print(
            "The total IR considering q=1 and q=2 hypothesis for {}".format(
                type_of_testing_IR
            )
        )
        print("{} using method {}".format(np.round(IR_tot_all_H, 3), evaluationMethod))
        results_IR_tot.append(IR_tot_all_H)

    # Add your custom legend
    legend_lines = []
    legend_labels = []

    for idx_el, lbl_rad in enumerate(radii_IR_2D):
        label = r"$R_{\mathcal{B}_x}" + "= {}$ m".format(lbl_rad)
        legend_lines.append(
            Line2D(
                [0],
                [0],
                color=colors_radii_IR[idx_el],
                linestyle=None,
                lw=0,
                marker="o",
                markersize=8,
            )
        )
        legend_labels.append(label)

    # legend_lines.append(Line2D([0], [0], color='black', linestyle=':'))
    # legend_labels.append(r'$P(\bar{x} \notin \mathcal{B}_x \bigcap \mathcal{H}_0)$')

    for type_of_testing_IR in types_of_testing:
        legend_lines.append(
            Line2D(
                [0], [0], color="black", linestyle=linestyle_dict[type_of_testing_IR]
            )
        )
        legend_labels.append(
            type_of_testing_IR.replace("_", "").replace("classical ", "")
        )

    # Global title
    fig.suptitle(
        rf"Total $P(\bar{{x}} \notin \mathcal{{B}}_x)$ for {type_of_example}, "
        rf"$p_{{sat}} = {psat},\ q_{{\text{{max}}}} = 1$",
        fontsize=16,
    )

    fig.legend(
        legend_lines,
        legend_labels,
        loc="upper right",
        ncol=1,
        fontsize=12,
        bbox_to_anchor=(0.9, 0.9, 0.1, 0.1),
        frameon=True,
        edgecolor="black",
        borderpad=0.35,
        labelspacing=0.3,
        handlelength=2,
        handletextpad=0.3,
        columnspacing=0.5,
    )

    plt.tight_layout()

    return fig, ax_IR, results_IR_tot


psat = 1e-2
fig, ax, results_IR_tot = results_plot_IR_vs_bias = plot_IR_vs_bias(
    type_of_DS_IR="A", psat=psat
)
totalFigName = os.path.join(
    main_dir,
    "Figures_testing_probabilities",
    "RIDS_vs_IDS_vs_DIA",
    f"psat={psat}",
    f"Total_IR_DIA_IDS_RIDS_{type_of_DS}.pkl",
)
os.makedirs(
    totalFigName.replace(f"Total_IR_DIA_IDS_RIDS_{type_of_DS}.pkl", ""), exist_ok=True
)

with open(totalFigName, "wb") as f:
    pickle.dump(fig, f)
plt.savefig(totalFigName.replace(".pkl", ".png"))

print("Results per test procedure")
print(results_IR_tot)
