"""
Description of code:
    Compute the non-centrality parameters under a particular hypothesis.
    We can compute the non-centrality parameters as a function of the bias b_j
    under any hypothesis. In addition, we can compute lambda (nc-param) for the
    chi2 distribution for the Tq statistic or the x-parameter (|| x_bar - x||_Qx0)
    Author:
    Bob van Noort
Date:
    01 - 07 - 2025

"""

import numpy as np
import scipy
import os
import sys
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from Functions import *
from matplotlib.patches import Patch


def get_nc_params(
    type_of_example, type_of_alpha, type_of_DS, type_of_testing, alpha_0, qmax, F_mat
):
    nb_samples = int(1e5)
    nb_bvals = 50
    N_sims = 20
    m, n, r, A, alpha, sigma, Qyy, Qyy_inv, B_T, Qtt, Qtt_inv = load_setup_parameters(
        type_of_example, alpha_method=type_of_alpha, alpha_0=alpha_0
    )

    k = int(m + m * (m - 1) / 2)
    cj_cjtplus_matrices = [np.zeros((m, r))]
    c_matrices_all = [np.zeros((m, 1))]

    for i in range(m):
        ci = np.eye(m)[:, i].reshape(-1, 1)
        ci_ctiplus = ci @ plusmat(B_T @ ci, Qtt_inv, inverse=True)
        cj_cjtplus_matrices.append(ci_ctiplus)
        c_matrices_all.append(ci)

    for i in range(m):
        for j in range(i + 1, m):
            cij = np.eye(m)[:, [i, j]]
            cij_ctiplus = cij @ plusmat(B_T @ cij, Qtt_inv, inverse=True)
            cj_cjtplus_matrices.append(cij_ctiplus)
            c_matrices_all.append(cij)

    # Following Zaminpardaz's paper on identifiability, theta:
    Qx0 = np.linalg.inv(A.T @ Qyy_inv @ A)
    A_plus = Qx0 @ A.T @ Qyy_inv
    Qtheta0 = F_mat.T @ Qx0 @ F_mat

    if type_of_example == "SPP_GNSS":
        brange = np.linspace(0, 50, nb_bvals)
    P_A = A @ plusmat(A, Qyy_inv, inverse=True)

    store_lambda_theta = np.zeros((m, nb_bvals))
    for hypt in np.arange(m):
        cj = np.eye(m)[:, hypt].reshape(-1, 1)
        print("At hypt", hypt, " for type of testing", type_of_testing)
        for i_bj, bj in enumerate(brange):
            lambda_theta_list = []

            for _ in range(N_sims):
                mean_t = B_T @ cj * np.array([[bj]])
                # print('At b_j = {} now'.format(bj))
                t_sample = np.random.multivariate_normal(
                    mean_t.flatten(), Qtt, size=nb_samples
                ).T
                Pis = compute_Pis(
                    t_sample,
                    Qtt,
                    B_T,
                    alpha_prime,
                    type_of_testing,
                    alpha_method=type_of_alpha,
                    type_of_example=type_of_example,
                    type_of_DS=type_of_DS,
                    alpha_0=alpha_0,
                    qmax=qmax,
                ).astype(int)

                E_hat_Pomega = np.sum(Pis == -1) / nb_samples
                # print('E_hat_Pomega', E_hat_Pomega)
                byi_hat = cj * bj  # approx of the byi_bar_vector (24) in Zaminpardaz

                for s in range(1, k + 1):  # the null hypt does not contribute...
                    where_equal_s = Pis == s
                    if np.sum(where_equal_s) == 0:
                        continue
                    # print(t_sample.shape)
                    tpj_s = t_sample[:, where_equal_s].sum(axis=1)
                    E_hat_tpj_s = ((tpj_s / nb_samples) / (1 - E_hat_Pomega)).reshape(
                        (r, 1)
                    )  # second term in 24
                    # print("E_hat_tpj_s", E_hat_tpj_s)
                    # print("Ehat_tpj_s.shape", E_hat_tpj_s.shape)
                    # print('cj_cjtplus_matrices[s].shape', cj_cjtplus_matrices[s].shape)
                    byi_hat -= cj_cjtplus_matrices[s] @ E_hat_tpj_s

                # print("byi_hat", byi_hat)
                b_theta = F_mat.T @ A_plus @ byi_hat
                lambda_theta = np.linalg.norm(
                    b_theta.T @ np.linalg.inv(Qtheta0) @ b_theta
                )
                lambda_theta_list.append(lambda_theta)

            store_lambda_theta[hypt, i_bj] = np.mean(lambda_theta_list)
    return brange, store_lambda_theta


if __name__ == "__main__":
    type_of_example = "ARAIM_UNDEC_GNSS"
    type_of_alpha = "manual"
    type_of_DS = "A"
    type_of_testing = "IDS"
    top_save_dir = os.path.join(
        r"C:\Users\bgvannoort\Documents\IDS\Results\NC_params", type_of_example
    )
    qmax = 2
    xmax = 50
    Identity_Qtheta = False
    # styling and plotting setups
    colors_hypothesis = {
        "P0": "#f5f5f5",
        "P1": "#ffff00",
        "P2": "#ff0000",
        "P3": "#0000ff",
        "P4": "#808080",
        "P5": "#800080",
        "P6": "#008000",
        "P7": "#ffa500",
    }
    linestyles = ["-", "--", ":"]

    plt.rcParams["font.size"] = 16

    idx_type = 0
    types_of_testing = ["R_IDS", "IDS", "classical DIA"]
    normalize_b_axis = True

    idx_type = 0
    scenario_nr = "sameP0_OMT"
    alpha_0_dictionary = {
        "R_IDS_C": 0.012,
        "R_IDS_B": 0.014,
        "R_IDS_A": 0.01,
        "IDS_A": 0.014,
        "IDS_B": 0.014,
        "IDS_C": 0.01,
        "classical DIA": 0.014,
        "DS_A": 0.014,
        "DS_B": 0.014,
        "DS_C": 0.01,
        "DS_D": 0.014,
    }

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
        top_save_dir = os.path.join(top_save_dir, scenario_nr)

    if Identity_Qtheta:
        top_save_dir = os.path.join(top_save_dir, "Identity_Qtheta")

    fig_combined, ax_combined = plt.subplots(figsize=(19.2, 9.83))
    for type_of_testing in types_of_testing:
        # Leads to an overall alpha of 0.05

        if type_of_testing == "classical DIA":
            alpha_0 = alpha_0_dictionary[type_of_testing]
        else:
            alpha_0 = alpha_0_dictionary[type_of_testing + "_" + type_of_DS]

        m, n, r, A, alpha, sigma, Qyy, Qyy_inv, B_T, Qtt, Qtt_inv = (
            load_setup_parameters(
                type_of_example, alpha_method=type_of_alpha, alpha_0=alpha_0
            )
        )
        alpha_prime = np.round(alpha, 3)
        sigma_i_list = np.sqrt(np.diag(Qyy))
        F_mat = np.zeros(
            (n, 2)
        )  # only distill the x and y components (first two compoents) of x
        F_mat[0, 0] = 1
        F_mat[1, 1] = 1

        ## Now we get the results from Delftblue
        # # types_of_testing = ['IDS']
        # for type_of_testing in types_of_testing:
        #     brange, store_nc_param_xbar = get_nc_params(type_of_example, type_of_alpha,
        #                                                 type_of_DS, type_of_testing, alpha_0, qmax, F_mat)

        # %% Plot results

        fig, ax = plt.subplots(figsize=(19.2, 9.83))
        labels = []
        label_handles = []
        if type_of_testing != "classical DIA":
            full_save_dir = os.path.join(
                top_save_dir,
                type_of_testing,
                type_of_DS,
                type_of_alpha,
                "alpha_0=" + str(alpha_0),
                "qmax={}".format(qmax),
            )

        else:
            full_save_dir = os.path.join(
                top_save_dir, type_of_testing, "alpha_prime={}".format(alpha_prime)
            )
        figdir = os.path.join(top_save_dir, "Figures", type_of_testing)
        os.makedirs(figdir, exist_ok=True)

        for hypt in np.arange(m):
            full_dir_file_nc = os.path.join(
                full_save_dir, "NC_params_hypt_{}.txt".format(hypt)
            )
            full_dir_file_b = os.path.join(full_save_dir, "brange_{}.txt".format(hypt))

            store_nc_param_xbar = np.loadtxt(full_dir_file_nc, delimiter=",")
            brange = np.loadtxt(full_dir_file_b, delimiter=",")

            if normalize_b_axis:
                ax.plot(
                    brange / sigma_i_list[hypt],
                    store_nc_param_xbar,
                    label=rf"$\mathcal{{H}}_{{{hypt + 1}}}$",
                    color=colors_hypothesis["P" + str(hypt + 1)],
                    linestyle=linestyles[idx_type],
                    linewidth=3,
                )
                ax_combined.plot(
                    brange / sigma_i_list[hypt],
                    store_nc_param_xbar,
                    label=rf"$\mathcal{{H}}_{{{hypt + 1}}}$",
                    color=colors_hypothesis["P" + str(hypt + 1)],
                    linestyle=linestyles[idx_type],
                    linewidth=3,
                )

                ax.set_xlabel(r"$\frac{b_i}{\sigma_i} $ [-]", fontsize=16)
                ax_combined.set_xlabel(r"$\frac{b_i}{\sigma_i} $ [-]", fontsize=16)

            else:
                ax.plot(
                    brange,
                    store_nc_param_xbar,
                    label=rf"$\mathcal{{H}}_{{{hypt + 1}}}$",
                    color=colors_hypothesis["P" + str(hypt + 1)],
                    linestyle=linestyles[idx_type],
                    linewidth=3,
                )
                ax_combined.plot(
                    brange,
                    store_nc_param_xbar,
                    label=rf"$\mathcal{{H}}_{{{hypt + 1}}}$",
                    color=colors_hypothesis["P" + str(hypt + 1)],
                    linestyle=linestyles[idx_type],
                    linewidth=3,
                )

                ax.set_xlabel(r"$b_i$ [m]", fontsize=16)
                ax_combined.set_xlabel(r"$b_i$ [m]", fontsize=16)
            label_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle=None,
                    linewidth=0,
                    markersize=15,
                    color=colors_hypothesis["P" + str(hypt + 1)],
                    label=rf"$\mathcal{{H}}_{{{hypt + 1}}}$",
                )
            )

        idx_type += 1

        ax.set_ylabel(r"$\lambda_{\bar{x}}$ [m]")
        ax_combined.set_ylabel(r"$\lambda_{\bar{x}}$ [m]")
        ax.set_yscale("log")
        ax_combined.set_yscale("log")

        ax.set_ylim(1e-4, 20)
        ax_combined.set_ylim(1e-4, 20)

        ax.set_xlim(0, xmax)
        ax_combined.set_xlim(0, xmax)

        ax.grid("on")
        ax_combined.grid("on")

        # Custom legend entries

        # legend_lines = []

        # for idx, line in enumerate(linestyles):
        #     legend_lines.append(
        #     Line2D([0], [0], color='black', linestyle=line, linewidth=2, label=types_of_testing[idx]))

        # for idx, hypt in enumerate(range(m)):
        #     legend_lines.append(
        #     Line2D([0], [0], color=colors_hypothesis['P' + str(hypt+1)], linestyle='-', linewidth=2, label=rf'$\mathcal{{H}}_{{{hypt+1}}}$'))
        # Add legend with labels
        # ax.legend(handles=legend_lines, loc='upper right')
        ax.legend(
            handles=label_handles,
            loc="upper right",
            frameon=True,
            edgecolor="black",
            borderpad=0.35,
            labelspacing=0.3,
            handlelength=2,
            handletextpad=0.3,
        )
        ax.set_title(
            rf"Non-centrality parameter $\lambda_{{\bar{{x}}}}$ for {type_of_testing} and {type_of_example.replace('_', ' ')} example"
        )
        plt.tight_layout()

        if normalize_b_axis:
            figname_save_fig = rf"NC_param_{type_of_testing}_{type_of_DS}_alpha_0_{alpha_0}_normalized_b_axis.png"
        else:
            figname_save_fig = (
                rf"NC_param_{type_of_testing}_{type_of_DS}_alpha_0_{alpha_0}.png"
            )
        fig.savefig(os.path.join(figdir, figname_save_fig))

    for i_ls, ls in enumerate(["-", "--", "dotted"]):
        txt_name = types_of_testing[i_ls].replace("R_", "R")
        txt_name = txt_name.replace("classical ", "")
        label_handles.append(
            Line2D([0], [0], color="black", linestyle=ls, lw=2, label=txt_name)
        )
    ax_combined.set_title(
        rf"Non-centrality parameter $\lambda_{{\bar{{x}}}}$ for {type_of_example.replace('_', ' ')} example"
    )
    ax_combined.legend(
        handles=label_handles,
        loc="upper right",
        frameon=True,
        edgecolor="black",
        borderpad=0.35,
        labelspacing=0.3,
        handlelength=2,
        handletextpad=0.3,
    )
    # ax_combined.tight_layout()
