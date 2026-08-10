"""
Description of code:
    Plot the conditional DIA PDF given a hypotheses and an example.
    Only consider either 1D PDFs or 2D pdfs.
Author:
    Bob van Noort
Date:
    11 june 2025

"""

import scipy
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable

from Functions import *


def fast_gaussian_pdf(X, mean, cov_inv, log_norm_const):
    # X: (..., d) array
    diffs = X - mean
    md2 = np.einsum("...i,ij,...j->...", diffs, cov_inv, diffs)  # Mahalanobis squared
    return np.exp(-0.5 * md2 + log_norm_const)


def compute_DIA_PDF_under_H(
    mean_t,
    mean_xhat0,
    Pcti,
    Li,
    Qx0x0,
    nb_samples_t,
    alpha_prime: float,
    type_of_testing: str,
    S: list = [],
    qmax: int = 5,
    idx_S: int = 0,
    cti_list: list = None,
    P_region: list = [],
    alpha_0: float = 0.01,
    type_of_DS: str = "B",
    lastOMT: bool = True,
    alpha_method: str = "Kok_IDS",
    type_of_example: str = "simple",
    up_to_component=2,
):

    # Input
    # The mean of the misclosure vector                    mean_t
    # The mean of xhat0 under H0 or Ha                     mean_xhat0
    # The vc-matrix of the misclosure vector               Q_tt
    # The projection matrices of the cti vectors           Pcti
    # The Li-terms                                         Li
    # The vc-matrix of the BLUE of xhat0                   Q_x0x0
    # The level of significnace of the OMT                 alpha
    # The number of samples of the misclosure vector       nb_samples_t

    # Output
    # The vectors spanning the 2D space of x               x1_x2_vectors
    # The DIA PDF under some H (H0 or Ha)                  f_DIA_H
    # The PDF of xhat0 under H (H0 or Ha)                  f_x0_H
    # The conditional PDFs (components of f_DIA_H)         f_FA_or_WI_CI_Hi
    # The probabilities of FA_i or WI_i, CI_i              Prob_i_comp

    # up to component specifies up to which component of x we would like to plot the pdf.
    # if 2, then only x and y
    # if 1, then only x.
    print("type of example", type_of_example)

    m, n, r, A, alpha, sigma, Qyy, Qyy_inv, Bt, Qtt, Qtt_inv = load_setup_parameters(
        type_of_example=type_of_example, alpha_method=alpha_method, alpha_0=alpha_0
    )
    if r > 4:
        arr_of_idxes, arr_of_parts = get_idx_hypt_RIDS(m, subset=[], qmax=qmax)
        k = len(arr_of_parts)
    else:
        k = int(m + m * (m - 1) / 2)

    # Generate pseudo-random samples from the PDF of the misclosure vector
    samples_t = np.random.multivariate_normal(mean_t, Qtt, nb_samples_t).T
    Pis = compute_Pis(
        samples_t,
        Qtt=Qtt,
        B_T=Bt,
        alpha_prime=alpha,
        type_of_testing=type_of_testing,
        S=S,
        qmax=qmax,
        idx_S=idx_S,
        cti_list=cti_list,
        P_region=P_region,
        alpha_0=alpha_0,
        type_of_DS=type_of_DS,
        lastOMT=lastOMT,
        alpha_method=alpha_method,
        type_of_example=type_of_example,
    )

    Pis = Pis.astype(int)

    # return Pis
    Li_t = np.zeros((k, up_to_component, nb_samples_t))
    for idx in range(0, k):
        Li_t[idx] = Li[idx] @ samples_t
    # Place the indices into a larger array
    i_Pi_arr = np.zeros((nb_samples_t, k + 2), dtype=int)
    i_Pi_arr[np.arange(nb_samples_t), Pis] = (
        1  # the -1th element, last element, corresponds to Undec.
    )

    # Compute the Probabilities of associated to each alternative hypothesis (under H0 they are the Prob_FA_i, if under Ha, then Prob_WI_i, Prob_CI_i)
    Prob_i_comp = np.zeros(k)  # plus H0 and undec region
    for idx in range(0, k):
        Prob_i_comp[idx] = np.mean(
            i_Pi_arr[:, idx + 1]
        )  # should contain only the alternative hypotheses, idx=0 is H0

    print(Prob_i_comp, np.sum(Prob_i_comp))

    # Do a sanity check over Prob_i_comp as it SHOULD NOT contain any values EXACTLY equal to 0
    if np.any(Prob_i_comp == 0):
        pass
        # raise Exception("The Prob_i_comp contains at least one 0 values. Try another value of ba, which should be lower than the current one")

    P_FA_or_CD_comp = np.sum(Pis != 0) / nb_samples_t
    P_CA_or_MD_comp = 1 - P_FA_or_CD_comp

    P_omega = i_Pi_arr[:, -1].mean()

    print("PFA_OR_CD", P_FA_or_CD_comp)
    print("PCA_OR_MD", P_CA_or_MD_comp)
    # Do a 'sanity' check on the P_FA_or_CD_comp and the sum of the Prob_i_comp
    if not np.isclose(P_FA_or_CD_comp, np.sum(Prob_i_comp) + P_omega, atol=1e-8):
        raise ValueError(
            "The computed P_FA_or_CD_comp and the sum of the computed Prob_i_comp do not coincide for a tolerance of 1e-8"
        )

    # (Temporary) Create a 2D meshgrid for which the DIA PDF will be computed
    if type_of_example == "Safoora_GNSS":
        x1 = np.linspace(-1.5, 1.5, 50) * 2
        x2 = np.linspace(-1.5, 1.5, 50) * 2
    else:
        x1 = np.linspace(-1.5, 1.5, 50) * 11
        x2 = np.linspace(-1.5, 1.5, 50) * 11
    x1_x2_vectors = np.vstack((x1, x2)).T

    # Create a 2D grid based on the x1 and x2
    X1, X2 = np.meshgrid(x1, x2)
    X_range = np.vstack((X1.flatten(), X2.flatten())).T
    size_X_range = X_range.shape[0]

    # Allocate memory for storage of the DIA PDF components
    f_DIA_H = np.zeros(size_X_range)
    f_x0_H = np.zeros(size_X_range)
    f_FA_or_WI_CI_Hi = np.zeros((k, 1, size_X_range))
    summed_weighted_f_Hi = np.zeros(size_X_range)

    # Compute the DIA PDF under H (H0 or a Ha)
    f_x0_H = scipy.stats.multivariate_normal.pdf(X_range, mean_xhat0, Qx0x0)

    # Loop over the x values
    for idx_X in range(0, size_X_range):
        # Loop over the number of alternative hypotheses
        for idx_k in range(0, k):
            if Prob_i_comp[idx_k] < 1 / nb_samples_t:
                # print("Prob_i_comp is zero for ", idx_k)
                pass
            else:
                # if idx_k == 27:
                #     print(np.mean(scipy.stats.multivariate_normal.pdf((X_range[idx_X,:] + Li_t[idx_k].T), mean_xhat0, Qx0x0) *i_Pi_arr[:,idx_k])/Prob_i_comp[idx_k])
                #     print(scipy.stats.multivariate_normal.pdf((X_range[idx_X,:] + Li_t[idx_k].T), mean_xhat0, Qx0x0) *i_Pi_arr[:,idx_k+1])
                #     print("where larger 0, sum", np.sum(scipy.stats.multivariate_normal.pdf((X_range[idx_X,:] + Li_t[idx_k].T), mean_xhat0, Qx0x0) *i_Pi_arr[:,idx_k+1] > 0))
                #     print('sum i_Pi[:, idx_k]', np.sum(i_Pi_arr[:,idx_k+1]))
                #     print('arg values', (X_range[idx_X,:] + Li_t[idx_k].T))
                f_FA_or_WI_CI_Hi[idx_k, :, idx_X] = (
                    np.mean(
                        scipy.stats.multivariate_normal.pdf(
                            (X_range[idx_X, :] + Li_t[idx_k].T), mean_xhat0, Qx0x0
                        )
                        * i_Pi_arr[:, idx_k + 1]
                    )
                    / Prob_i_comp[idx_k]
                )

        summed_weighted_f_Hi[idx_X] = np.sum(
            f_FA_or_WI_CI_Hi[:, :, idx_X].T * Prob_i_comp
        )
        f_DIA_H[idx_X] = (
            f_x0_H[idx_X] * P_CA_or_MD_comp + summed_weighted_f_Hi[idx_X]
        ) / (1 - P_omega)

    return x1_x2_vectors, f_DIA_H, f_x0_H, f_FA_or_WI_CI_Hi, Prob_i_comp, Pis


def compute_DIA_matrices(A, B, Qyy, Qtt, c_vectors, up_to_component):
    # k does not include the null hypothesis and neither does it include the undecided region.
    m, n = A.shape
    r = m - n
    k = len(c_vectors)
    Pcti = np.zeros((k, r, r))
    Li = np.zeros((k, up_to_component, r))

    for idx in range(0, k):
        ci = c_vectors[idx]

        Btci_plus = plusmat(B.T @ ci, Qtt)
        Pcti[idx, :, :] = B.T @ ci @ Btci_plus
        Li[idx, :, :] = (plusmat(A, Qyy) @ ci @ Btci_plus)[:up_to_component, :]

    return Pcti, Li


# %%
if __name__ == "__main__":
    # type_of_example = 'SPP_GNSS'
    # type_of_example = 'Safoora_GNSS'
    type_of_example = "ARAIM_UNDEC_GNSS"
    # type_of_testing = 'classical DIA'
    type_of_testing = "R_IDS"
    type_of_DS = "A"
    alpha_method = "manual"
    # bias_values = [40, 50, 60, 70, 80, 90]
    under_H_idx = 4  # this is under Hypt 'under_H_idx+1'.
    bool_other_biases = False

    alpha_0 = 0.01
    # qmax=2

    print(
        "Carrying out the cond. PDF computation of {} type {}".format(
            type_of_testing, type_of_DS
        )
    )
    nb_samples_t = int(1e5)
    plt.rcParams["font.size"] = 16
    # Example of the observation model
    m, n, r, A, alpha, sigma, Qyy, Qyy_inv, Bt, Qtt, Qtt_inv = load_setup_parameters(
        type_of_example=type_of_example, alpha_method=alpha_method, alpha_0=alpha_0
    )

    store_dia_PDFs = []
    if type_of_example == "Safoora_GNSS":
        r_bx_regions = np.array([0.2, 0.4, 0.6, 1.0, 2.0, 3.0])
        bias_values = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
        qmax = 3
    elif type_of_example == "SPP_GNSS" or type_of_example == "ARAIM_UNDEC_GNSS":
        r_bx_regions = np.arange(1, 17, 2)  # for spp gnss example
        bias_values = [5, 7, 9, 11, 15, 25]
        qmax = 2

    for bval in bias_values:
        if type_of_testing == "classical DIA":
            testing_string = type_of_testing

            saving_string = os.path.join(
                os.getcwd(),
                "..",
                "Results",
                "PDFs",
                type_of_example,
                type_of_testing,
                "alpha_prime=" + str(np.round(alpha, 3)),
            )
            os.makedirs(saving_string, exist_ok=True)
            saving_string1 = os.path.join(
                saving_string,
                "PDF_of_DIA_estimator_under_H{}_bval_{}.png".format(
                    under_H_idx + 1, np.round(bval, 1)
                ),
            )

            saving_string2 = os.path.join(
                saving_string,
                "PDF_of_DIA_estimator_under_H{}_bval_{}_WI_".format(
                    under_H_idx + 1, np.round(bval, 1)
                )
                + "{}.png",
            )
        else:
            testing_string = type_of_testing + "_" + type_of_DS
            testing_string = testing_string.replace("R_", "Reverse ")
            testing_string = testing_string.replace("_", " type ")

            saving_string = os.path.join(
                os.getcwd(),
                "..",
                "Results",
                "PDFs",
                type_of_example,
                type_of_testing,
                type_of_DS,
                alpha_method + "_alpha0_" + str(alpha_0),
            )
            os.makedirs(saving_string, exist_ok=True)
            saving_string1 = os.path.join(
                saving_string,
                "PDF_of_DIA_estimator_under_H{}_bval_{}.png".format(
                    under_H_idx + 1, np.round(bval, 1)
                ),
            )

            saving_string2 = os.path.join(
                saving_string,
                "PDF_of_DIA_estimator_under_H{}_bval_{}_WI_".format(
                    under_H_idx + 1, np.round(bval, 1)
                )
                + "{}.png",
            )

        if bool_other_biases:
            saving_string = os.path.join(saving_string, "other_biases")
            os.makedirs(saving_string, exist_ok=True)
            saving_string1 = os.path.join(
                saving_string,
                "PDF_of_DIA_estimator_under_H{}_bval_{}.png".format(
                    under_H_idx + 1, np.round(bval, 1)
                ),
            )

            saving_string2 = os.path.join(
                saving_string,
                "PDF_of_DIA_estimator_under_H{}_bval_{}_WI_".format(
                    under_H_idx + 1, np.round(bval, 1)
                )
                + "{}.png",
            )

        saving_string_PDFs = os.path.join(saving_string, "PDF_csv")
        os.makedirs(saving_string_PDFs, exist_ok=True)

        up_to_component = 2  # up to which component do we want to compute the PDF. If 2, then we consider x and y
        # if equal to 1, then only x.

        Aplus = plusmat(A, Qyy)
        Qx0x0 = np.linalg.inv(A.T @ Qyy_inv @ A)

        # Set the model misspecifications signature vectors
        r = Qtt.shape[0]
        m = A.shape[0]
        c_vectors = np.identity(m)

        c_vectors = []
        partitions_list = []
        if type_of_example == "Safoora_GNSS" or type_of_example == "Sebastian_GNSS":
            for i in range(m):
                c_vectors.append(np.eye(m)[:, i].reshape(-1, 1))
                partitions_list.append("P" + str(i + 1))
            for i in range(m):
                for j in range(i + 1, m):
                    c_vectors.append(np.eye(m)[:, [i, j]])
                    partitions_list.append("P" + str(i + 1) + "," + str(j + 1))

            for i in range(m):
                for j in range(i + 1, m):
                    for jj in range(j + 1, m):
                        c_vectors.append(np.eye(m)[:, [i, j, jj]])
                        partitions_list.append(
                            "P" + str(i + 1) + "," + str(j + 1) + "," + str(jj + 1)
                        )
        else:
            for i in range(m):
                c_vectors.append(np.eye(m)[:, i].reshape(-1, 1))
                partitions_list.append("P" + str(i + 1))
            for i in range(m):
                for j in range(i + 1, m):
                    c_vectors.append(np.eye(m)[:, [i, j]])
                    partitions_list.append("P" + str(i + 1) + str(j + 1))

        # Get the required "DIA" matrices to compute the DIA PDF
        Pcti, Li = compute_DIA_matrices(A, Bt.T, Qyy, Qtt, c_vectors, up_to_component)

        start_time = time.time()
        # Compute the DIA PDF under the hypt_idx+1
        mean_t = (np.zeros((r, 1)) + Bt @ c_vectors[under_H_idx] * bval).flatten()
        mean_xhat0 = (
            np.zeros((n, 1)) + Aplus @ c_vectors[under_H_idx] * bval
        ).flatten()

        # beware to only insert the first 2 components of xhat0
        x1_x2_vectors, f_DIA_H0, f_x0_H0, f_FA_Hi, Prob_FA_i_comp, Pis = (
            compute_DIA_PDF_under_H(
                mean_t,
                mean_xhat0[:up_to_component],
                Pcti,
                Li,
                Qx0x0[:up_to_component, :up_to_component],
                nb_samples_t,
                alpha_prime=alpha,
                type_of_testing=type_of_testing,
                type_of_DS=type_of_DS,
                type_of_example=type_of_example,
                alpha_method=alpha_method,
                qmax=qmax,
            )
        )

        Prob_CA_comp = 1 - np.sum(Prob_FA_i_comp)

        # toc
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("The runtime is: ", elapsed_time, "seconds")

        # %%
        ####################################################################### Plotting #################################################################################################################

        k = len(c_vectors)
        n_cols = x1_x2_vectors.shape[0]
        n_rows = n_cols

        circles = []
        default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for radius in r_bx_regions:
            circles.append((0, 0, radius))

        f_DIA_reshaped = np.reshape(f_DIA_H0, (n_rows, n_cols))

        f_x0_reshaped = np.reshape(f_x0_H0, (n_rows, n_cols))

        f_FA_reshaped = np.zeros((k, n_cols, n_rows))
        for idx in range(0, k):
            f_FA_reshaped[idx] = np.reshape(f_FA_Hi[idx], (n_rows, n_cols))

        # define the levels and colorbar limits for all subplots
        vmin = 1e-9
        vmaxDIA = np.max(f_DIA_reshaped)
        vmaxx0 = np.max(f_x0_reshaped)
        levelsDIA = np.linspace(vmin, vmaxDIA, 16)
        levelsx0 = np.linspace(vmin, vmaxx0, 16)

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(19, 8))

        # Clip data to avoid underflow below vmin
        f_DIA_reshaped_clipped = np.clip(f_DIA_reshaped, vmin, None)
        f_x0_reshaped_clipped = np.clip(f_x0_reshaped, vmin, None)

        # Use LogNorm with defined range
        norm_DIA = LogNorm(vmin=vmin, vmax=vmaxDIA)
        norm_x0 = LogNorm(vmin=vmin, vmax=vmaxx0)
        try:
            subplot1 = axs[0].contourf(
                x1_x2_vectors[:, 0],
                x1_x2_vectors[:, 1],
                f_DIA_reshaped_clipped,
                levels=16,
                norm=norm_DIA,
            )
            subplot2 = axs[1].contourf(
                x1_x2_vectors[:, 0],
                x1_x2_vectors[:, 1],
                f_x0_reshaped_clipped,
                levels=16,
                norm=norm_x0,
            )

            # Add to both subplots
            for idx_circ, (cx, cy, r) in enumerate(circles):
                circle1 = Circle(
                    (cx, cy),
                    r,
                    label=r"$\mathcal{B}_x --" + r" r = {} m$".format(r),
                    edgecolor=default_colors[idx_circ],
                    facecolor="none",
                    linewidth=1.5,
                )
                circle2 = Circle(
                    (cx, cy),
                    r,
                    label=r"$\mathcal{B}_x --" + r" r = {} m$".format(r),
                    edgecolor=default_colors[idx_circ],
                    facecolor="none",
                    linewidth=1.5,
                )

                axs[0].add_patch(circle1)  # Add to subplot 1
                axs[1].add_patch(circle2)  # Add to subplot 2

            axs[0].set_xlabel(r"$x_1$ [m]")
            axs[0].set_ylabel(r"$x_2$ [m]")
            cbar1 = fig.colorbar(
                subplot1, ax=axs[0], orientation="vertical", shrink=0.8
            )
            axs[0].set_title(
                "PDF of DIA estimator under $\\mathcal{{H}}_{}$ at $b_{} = {}$ m\nfor {}".format(
                    under_H_idx + 1,
                    under_H_idx + 1,
                    bval,
                    testing_string.replace("classical", "traditional"),
                )
            )

            cbar1.set_label("", labelpad=10, loc="top")
            cbar1.ax.set_title(r"$f_x$ [-]", fontsize=16, pad=12, loc="center")

            axs[1].set_title(
                "PDF of $\\hat{{x}}_0$ under $\\mathcal{{H}}_{}$ at $b_{} = {}$ m\nfor {}".format(
                    under_H_idx + 1,
                    under_H_idx + 1,
                    bval,
                    testing_string.replace("classical", "traditional"),
                )
            )
            # cbar2 = fig.colorbar(subplot2, ax=axs[1], orientation='vertical', shrink=0.8)

            # cbar2.set_label('', labelpad=10, loc='top')
            # cbar2.ax.set_title(r'$f_x$ [-]', fontsize=16, pad=12, loc='center')
            axs[1].set_xlabel(r"$x_1$ [m]")
            axs[1].set_ylabel(r"$x_2$ [m]")

            axs[0].set_aspect(1)
            axs[1].set_aspect(1)

            axs[0].grid("on")
            axs[1].grid("on")

            plt.tight_layout()
            plt.savefig(saving_string1)
        except:
            pass
        # store the DIA PDF
        store_dia_PDFs.append(f_DIA_reshaped_clipped)
        np.savetxt(
            os.path.join(
                saving_string_PDFs,
                "f_DIA_reshaped_under_H{}_bias_{}.csv".format(under_H_idx + 1, bval),
            ),
            f_DIA_reshaped_clipped,
        )
        # %%
        # Use this figure to plot the conditional pdf under the largest PWI
        idx_PWI_largest = np.argsort(-Prob_FA_i_comp)
        idx_to_use = idx_PWI_largest[
            1
        ]  # not the first one (corresponding to correct identification), but the second one
        f_to_use_reshaped = f_FA_reshaped[idx_to_use]
        np.savetxt(
            os.path.join(
                saving_string_PDFs,
                "f_largest_WI_reshaped_under_H{}_bias_{}_for_WI_{}.csv".format(
                    under_H_idx + 1, bval, partitions_list[idx_to_use].replace("P", "")
                ),
            ),
            f_to_use_reshaped,
        )
        fig, ax = plt.subplots(figsize=(10, 10))
        # define the levels and colorbar limits for all subplots
        vmin = 1e-9
        vmaxPWI = np.max(f_to_use_reshaped)

        levelsPWI = np.linspace(vmin, vmaxDIA, 16)

        # Clip data to avoid underflow below vmin
        f_PWI_reshaped_clipped = np.clip(f_to_use_reshaped, vmin, None)

        # Use LogNorm with defined range
        norm_PWI = LogNorm(vmin=vmin, vmax=vmaxPWI)
        try:
            plot1 = ax.contourf(
                x1_x2_vectors[:, 0],
                x1_x2_vectors[:, 1],
                f_PWI_reshaped_clipped,
                levels=16,
                norm=norm_PWI,
            )

            # Add to both subplots
            for idx_circ, (cx, cy, r) in enumerate(circles):
                circle1 = Circle(
                    (cx, cy),
                    r,
                    label=r"$\mathcal{B}_x --" + r" r = {} m$".format(r),
                    edgecolor=default_colors[idx_circ],
                    facecolor="none",
                    linewidth=1.5,
                )

                ax.add_patch(circle1)  # Add to subplot 1

            ax.set_xlabel(r"$x_1$ [m]")
            ax.set_ylabel(r"$x_2$ [m]")
            cbar1 = fig.colorbar(plot1, ax=ax, orientation="vertical", shrink=0.8)
            ax.set_title(
                "PDF $(\\bar{{x}} \\mid \\mathcal{{H}}_{{{}}}, \\text{{WI}}_{{{}}}^{{{}}})$ at $b_{{{}}} = {}$ m\nfor {}".format(
                    under_H_idx + 1,
                    under_H_idx + 1,
                    partitions_list[idx_to_use].replace("P", ""),
                    under_H_idx + 1,
                    bval,
                    testing_string.replace("classical", "traditional"),
                )
            )

            cbar1.set_label("", labelpad=10, loc="top")
            cbar1.ax.set_title(r"$f_x$ [-]", fontsize=16, pad=12, loc="center")

            ax.set_aspect(1)
            ax.grid("on")

            plt.tight_layout()
            plt.savefig(
                saving_string2.format(partitions_list[idx_to_use].replace("P", ""))
            )
        except:
            print(
                "The requested PDF for PWI of H{} is ~0 in this range of x1 and x2 values".format(
                    idx_to_use + 1
                )
            )

    # %% Make a combined figure of all DIA PDFs under H3 with varying biases.
    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(15.88, 9.08))
    fig.subplots_adjust(
        left=0.0350, right=0.90, bottom=0.072, top=0.965, hspace=0.095, wspace=0.000
    )
    plt.rcParams["font.size"] = 16

    vmax0 = 0
    for elz in store_dia_PDFs:
        vmax1 = np.max(elz)
        if vmax1 > vmax0:
            vmax0 = vmax1
    vmaxDIA = vmax0
    # Use LogNorm with defined range
    norm_DIA = LogNorm(vmin=vmin, vmax=vmaxDIA)

    idx = 0
    for i in range(2):
        for j in range(3):
            f_DIA_reshaped = store_dia_PDFs[int(idx)]
            levelsDIA = np.linspace(vmin, vmaxDIA, 16)

            # Clip data to avoid underflow below vmin
            f_DIA_reshaped_clipped = np.clip(f_DIA_reshaped, vmin, None)

            contour = axes[i, j].contourf(
                x1_x2_vectors[:, 0],
                x1_x2_vectors[:, 1],
                f_DIA_reshaped_clipped,
                levels=16,
                norm=norm_DIA,
            )
            for idx_circ, (cx, cy, r) in enumerate(circles):
                circle1 = Circle(
                    (cx, cy),
                    r,
                    label=r"$\mathcal{B}_x --" + r" r = {} m$".format(r),
                    edgecolor=default_colors[idx_circ],
                    facecolor="none",
                    linewidth=1.5,
                )

                axes[i, j].add_patch(circle1)  # Add to subplot 1

            axes[i, j].set_aspect(1)
            axes[i, j].grid(True)
            axes[i, j].set_title(
                r"$f_{\bar{x}}$ at $b"
                + "_{} = {}$ m".format(under_H_idx + 1, bias_values[idx]),
                fontsize=14,
            )

            if i == 1:
                axes[i, j].set_xlabel(r"$x_1$ [m]")
            if j == 0:
                axes[i, j].set_ylabel(r"$x_2$ [m]")
            idx += 1

    # Create single colorbar
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(
        ScalarMappable(norm=norm_DIA, cmap=contour.cmap),
        ax=axes,
        cax=cbar_ax,
        orientation="vertical",
        shrink=1.0,
        pad=0.02,
    )
    cbar.ax.set_title(r"$f_x$ [-]", fontsize=14, pad=12, loc="center")

    plt.tight_layout(rect=[0, 0, 0.92, 1])  # leave space for colorbar on the right
    plt.show()
    plt.savefig(
        os.path.join(
            saving_string, "Combined_DIA_PDF_under_H{}.png".format(under_H_idx + 1)
        )
    )
