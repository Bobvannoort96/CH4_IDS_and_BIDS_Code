"""
Created on 3 sept 2024
Author: BG van Noort

DIA computations of the PCI, PWI etc.

Computes only the data and writes it out to data (.txt) files. Use
the programs 'plot_IR_vs_bi.py' and 'Plot_PCI_PWI.py' to actually plot and visualize
the data.
"""

import numpy as np
import scipy
import scipy.stats
from Functions import *
import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt


def P_F_safety_region_GNSS(xhat, type_of_example, radii_list, x_true=None):
    # xhat should contain only the 'true' samples, i.e. without nans from the undecided region
    # type of example is necessary to check if a GNSS setup is chosen, otherwise, we have a different safety region

    if x_true is None:
        x_true = np.zeros((xhat.shape))
    else:
        pass

    xhat = xhat - x_true

    if type_of_example == "simple":
        xhatSquared = np.linalg.norm(xhat, axis=0) ** 2
    elif "GNSS" in type_of_example:
        h_hat = xhat[:2, :] ** 2
        xhatSquared = h_hat.sum(axis=0)
    else:
        raise Exception(
            "Example type does not contain GNSS or is not the simple example"
        )

    prob_fail = np.zeros(len(radii_list))
    for idx_rad, radius_Bx in enumerate(radii_list):
        outside_Bx = (xhatSquared > radius_Bx**2).sum()
        PIR = outside_Bx / len(xhatSquared)  # divide by nr of times a sol is provided.
        prob_fail[idx_rad] = PIR

    return prob_fail


def run_single_simulation(
    type_of_example,
    type_of_testing,
    type_of_DS,
    type_of_alpha,
    simnr,
    D1_hypothesis,
    b_value,
    qmax,
):

    np.random.seed(987 + simnr)  # Optional: seed variation for repeatability

    alpha_0 = 0.01
    alpha_prime = 0.05

    current_dir = os.getcwd()
    m, n, r, A, alpha, sigma, Qyy, Qyy_inv, B_T, Qtt, Qtt_inv = load_setup_parameters(
        type_of_example,
        alpha_method=type_of_alpha,
        alpha_0=alpha_0,
        alpha_prime=alpha_prime,
    )
    Aplus = plusmat(A, Qyy)
    lastOMT = True
    alpha_prime = modify_alpha_prime(None, alpha_0, m, type_of_alpha, r)

    if type_of_example == "Safoora_GNSS":
        arr_of_idxes, arr_of_parts = get_idx_hypt_RIDS(m, subset=[], qmax=qmax)
        ci_list, P_region = [], []
        for pp in arr_of_parts:
            idxes = np.array(pp.replace("P", "").split(","), dtype=int) - 1
            ci = np.eye(m)[:, idxes].reshape((m, len(idxes)))
            P_region.append([pp, ci])
            ci_list.append(ci)

        arr_of_idxes = [0, -1] + arr_of_idxes
        arr_of_parts = ["P0", "P99"] + arr_of_parts
        indices_partitions = dict(zip(arr_of_parts, arr_of_idxes))
        radii_Bx = np.array([0.2, 0.4, 0.6, 1.0, 2.0, 3.0])
    elif type_of_example == "SPP_GNSS":
        indices_partitions = load_indices_partitions_GNSS_ex(
            False
        )  # IDS_sep_order = False
        c_list = []
        P_region = []
        for i in range(m):
            c_i = np.eye(m)[:, i].reshape(-1, 1)
            c_list.append(c_i)
            partition = "P" + str(i + 1)
            P_region.append([partition, c_i])

        for i in range(m):
            c_i = np.eye(m)[:, i].reshape(-1, 1)
            for j in range(i + 1, m):
                c_j = np.eye(m)[:, j].reshape(-1, 1)
                cij = np.hstack((c_i, c_j))
                c_list.append(cij)
                partition = "P" + str(i + 1) + str(j + 1)
                P_region.append([partition, cij])
        qmax = 2  # Set the maximum number of outliers we could/want to identify.
        radii_Bx = np.arange(5, 25, 5)  # different radii for the circular region Bx
        ## Check if dir exists for writing the results to for PCI/PWI PCD/PMD etc.
        # writing_dir = os.path.join(current_dir, '..', 'Results', 'TestingProbabilities', 'SPP_GNSS')
    else:
        raise NotImplementedError("Only 'Safoora_GNSS' is implemented in this wrapper.")

    hypt_to_model = D1_hypothesis + 1
    partition_to_model = "P" + str(hypt_to_model)
    subset = [int(zeta) for zeta in str(D1_hypothesis)]
    idx_partition_to_model = get_idx_hypt_RIDS(m, subset=subset, qmax=qmax)

    for part, cvec in P_region:
        if part == partition_to_model:
            C_i_model = cvec

    if C_i_model.shape[1] != 1:
        print("Only 1D grid search implemented.")
        return

    N = int(1e5)
    y_mean = C_i_model @ np.array([[b_value]])
    y_sample = np.random.multivariate_normal(y_mean.flatten(), Qyy, size=(N,)).T
    t_sample = B_T @ y_sample
    print(f"Running b_i = {b_value}...")

    Pis = compute_Pis(
        t_sample,
        Qtt,
        B_T,
        alpha_prime,
        type_of_testing=type_of_testing,
        P_region=P_region,
        type_of_DS=type_of_DS,
        lastOMT=lastOMT,
        alpha_0=alpha_0,
        type_of_example=type_of_example,
        qmax=qmax,
        cti_list=None,
        S=[],
        idx_S=0,
    )

    xhat = np.zeros((n, N))
    identH0 = Pis == 0
    P_acceptH0 = np.sum(identH0) / N
    xhat[:, identH0] = Aplus @ y_sample[:, identH0]

    P_CI_WI = np.zeros(len(P_region))
    column_names = ["b_i", "PCA_MD"]
    for i, (part, cvec) in enumerate(P_region):
        Pci_perp = P_perp(P_mat(cvec, Qyy))
        Abar = Pci_perp @ A
        Abarplus = plusmat(Abar, Qyy)
        idx_to_identify = indices_partitions[part]
        identHi = Pis == idx_to_identify
        P_CI_WI[i] = np.sum(identHi) / N
        xhat[:, identHi] = Abarplus @ y_sample[:, identHi]
        column_name = f"P_FA_{part.replace('P', '')}_CI_{part.replace('P', '')}"
        if column_name not in column_names:
            column_names.append(column_name)

    identHomega = Pis == -1
    xhat[:, identHomega] = np.nan
    P_omega = np.sum(identHomega) / N
    column_names.append("P_OMEGA")

    all_probs = np.hstack(([b_value, P_acceptH0], P_CI_WI, P_omega))

    full_prob_failure = P_F_safety_region_GNSS(
        xhat[:, ~identHomega], type_of_example, radii_Bx
    )

    # Save to disk
    writing_dir = os.path.join(
        current_dir, "..", "Results", "TestingProbabilities", type_of_example
    )
    if type_of_testing in ["IDS", "DS", "R_IDS"]:
        dir_fname = os.path.join(
            writing_dir,
            type_of_testing,
            type_of_DS,
            type_of_alpha,
            f"Hypothesis{hypt_to_model}",
        )
    else:
        dir_fname = os.path.join(
            writing_dir, type_of_testing, f"Hypothesis{hypt_to_model}"
        )

    os.makedirs(dir_fname, exist_ok=True)

    prob_file = os.path.join(
        dir_fname, f"PCA_FA_CI_WI_bval_{b_value:.2f}_sim_{simnr}.csv"
    )
    df_probs = pd.DataFrame(all_probs.reshape(1, -1), columns=column_names)
    df_probs.to_csv(prob_file, index=False)
    print(f"Wrote probability data to {prob_file}")

    IR_file = os.path.join(
        dir_fname, f"IR_estimates_bval_{b_value:.2f}_sim_{simnr}.csv"
    )
    IR_data = np.hstack(([b_value], full_prob_failure))
    np.savetxt(IR_file, IR_data.reshape(1, -1), delimiter=",")
    print(f"Wrote IR data to {IR_file}")


if __name__ == "__main__":
    type_of_alpha = "manual"
    alpha_0 = 0.01
    alpha_prime = 0.05
    type_of_example = "SPP_GNSS"
    bool_much_larger_biases = False
    type_of_testing = "R_IDS"
    type_of_DS = "A"
    simnr = 8
    D1_hypothesis = 2
    qmax = 2
    Nsims = 50

    # for D1_hypothesis in range(0,7):
    #     for simnr in range(Nsims):
    #         run_single_simulation(
    #         type_of_example='SPP_GNSS',
    #         type_of_testing='IDS',
    #         type_of_DS='A',
    #         type_of_alpha=type_of_alpha,
    #         simnr=simnr,
    #         D1_hypothesis=D1_hypothesis,
    #         b_value=5.0,
    #         qmax=qmax
    #     )
    # sys.exit()

    # run_single_simulation(type_of_example, type_of_testing, type_of_DS, simnr, D1_hypothesis)
    # sys.exit()

    current_dir = os.getcwd()
    m, n, r, A, alpha, sigma, Qyy, Qyy_inv, B_T, Qtt, Qtt_inv = load_setup_parameters(
        type_of_example,
        alpha_method=type_of_alpha,
        alpha_0=alpha_0,
        alpha_prime=alpha_prime,
    )
    if type_of_example == "simple":
        # dictionaries of indices. See the description of identifications_list in function IDS_2its()
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
        ## carry out the hypothesis testing with the normal dia method
        c_list = []
        P_region = []
        for i in range(m):
            c_i = np.eye(m)[:, i].reshape(-1, 1)
            c_list.append(c_i)
            partition = "P" + str(i + 1)
            P_region.append([partition, c_i])

        for i in range(m):
            c_i = np.eye(m)[:, i].reshape(-1, 1)
            for j in range(i + 1, m):
                c_j = np.eye(m)[:, j].reshape(-1, 1)
                cij = np.hstack((c_i, c_j))
                c_list.append(cij)
                partition = "P" + str(i + 1) + str(j + 1)
                P_region.append([partition, cij])

        radii_Bx = np.arange(1, 6)  # different radii for the circular region Bx
        ## Check if dir exists for writing the results to for PCI/PWI PCD/PMD etc.
        # writing_dir = os.path.join(current_dir, '..', 'Results', 'TestingProbabilities', 'SimpleExample')
        qmax = 2  # Set the maximum number of outliers we could/want to identify.
    elif type_of_example == "SPP_GNSS":
        indices_partitions = load_indices_partitions_GNSS_ex(
            False
        )  # IDS_sep_order = False
        c_list = []
        P_region = []
        for i in range(m):
            c_i = np.eye(m)[:, i].reshape(-1, 1)
            c_list.append(c_i)
            partition = "P" + str(i + 1)
            P_region.append([partition, c_i])

        for i in range(m):
            c_i = np.eye(m)[:, i].reshape(-1, 1)
            for j in range(i + 1, m):
                c_j = np.eye(m)[:, j].reshape(-1, 1)
                cij = np.hstack((c_i, c_j))
                c_list.append(cij)
                partition = "P" + str(i + 1) + str(j + 1)
                P_region.append([partition, cij])
        qmax = 2  # Set the maximum number of outliers we could/want to identify.
        radii_Bx = np.arange(5, 25, 5)  # different radii for the circular region Bx
        ## Check if dir exists for writing the results to for PCI/PWI PCD/PMD etc.
        # writing_dir = os.path.join(current_dir, '..', 'Results', 'TestingProbabilities', 'SPP_GNSS')
    elif type_of_example == "Safoora_GNSS":
        qmax = 3
        arr_of_idxes, arr_of_parts = get_idx_hypt_RIDS(m, subset=[], qmax=qmax)

        ci_list, P_region = [], []
        for pp in arr_of_parts:
            idxes = pp.replace("P", "").split(",")
            idxes = np.int_(idxes) - 1
            ci = np.eye(m)[:, idxes].reshape((m, len(idxes)))
            P_region.append([pp, ci])
            ci_list.append(ci)

        ## create indices_partitions dictionary
        arr_of_idxes = [0, -1] + arr_of_idxes
        arr_of_parts = ["P0", "P99"] + arr_of_parts

        indices_partitions = dict(zip(arr_of_parts, arr_of_idxes))
        radii_Bx = np.array([0.2, 0.4, 0.6, 1.0, 2.0, 3.0])

    writing_dir = os.path.join(
        current_dir, "..", "Results", "TestingProbabilities", type_of_example
    )

    lastOMT = True  # only relevant in cas type_of_testing = "IDS"

    Aplus = plusmat(A, Qyy)
    options = [
        ["DS", "A"],
        ["DS", "B"],
        ["DS", "C"],
        ["DS", "D"],
        ["IDS", "A"],
        ["IDS", "B"],
        ["IDS", "C"],
        "classical DIA",
    ]
    options = [
        ["DS", "A"],
        ["DS", "B"],
        ["DS", "D"],
        ["IDS", "A"],
        ["IDS", "B"],
        "classical DIA",
    ]
    # options = [['IDS', 'A']]
    ## IDS and R_IDS input parameters
    S_0 = []  # the initial set

    idx_S_0 = 0
    cti_list_input = None

    # partition_to_model = 'P13'
    partition_to_model = "P1"
    for opt in options:
        # carry out the whole procedure per option, option is e.g. DS with 'A' or IDS with 'B' or classical DIA
        if type(opt) == list:
            type_of_testing, type_of_DS = opt
        else:
            type_of_testing = opt

        N_sims = 50

        N = int(1e5)  # nr of samples to generate for y
        # sys.exit()
        for simnr in range(N_sims):
            print("simnr", simnr)
            # START COMMENT -UNCOMMENT
            # ENTER BY HAND
            for D1_hypothesis in np.arange(m):
                hypt_to_model = (
                    D1_hypothesis + 1
                )  # this is the hypthesis to model under. See also variable 'indices_partitions'
                # Corresponds to H13 for the GNSS example

                partition_to_model = "P" + str(hypt_to_model)  # true for 1D hypts

                if (
                    type_of_example == "Safoora_GNSS"
                    or type_of_example == "Sebastian_GNSS"
                ):
                    subset = [int(partition_to_model.replace("P", "")) - 1]

                    idx_partition_to_model = get_idx_hypt_RIDS(
                        m, subset=subset, qmax=qmax
                    )
                    print(
                        "subset, idx_partition, D1_hypothesis",
                        subset,
                        idx_partition_to_model,
                        D1_hypothesis,
                    )
                else:
                    idx_partition_to_model = indices_partitions[partition_to_model]
                ## loop over P_region list to determine the C_i vector used for modelling
                ## END ENTER BY HAND
                ## END UNCOMMENT-COMMENT

                for part, cvec in P_region:
                    if part == partition_to_model:
                        C_i_model = cvec
                if C_i_model.shape[1] == 1:  # the grid search is 1D
                    return_probabilities = []

                    n_b_samples = 100
                    full_xhat = np.zeros((n_b_samples, n, N))
                    full_prob_failure = np.zeros((n_b_samples, len(radii_Bx)))
                    bi_range = np.linspace(0, 50, n_b_samples)
                    print(
                        "Simulation {}".format(simnr + 1),
                        " for ",
                        partition_to_model,
                        "for ",
                        type_of_testing,
                        type_of_DS,
                    )
                    # for q_i=1, we only have one b_i value..
                    for index, b_i in enumerate(bi_range):
                        b_i = np.array([[b_i]])
                        # b_i=np.array([[12]])

                        y_mean = C_i_model @ b_i  # we implicitly assume that x=0
                        y_sample = np.random.multivariate_normal(
                            y_mean.flatten(), Qyy, size=(N,)
                        ).T

                        t_sample = B_T @ y_sample

                        Pis = compute_Pis(
                            t_sample,
                            Qtt,
                            B_T,
                            alpha_prime,
                            type_of_testing=type_of_testing,
                            P_region=P_region,
                            type_of_DS=type_of_DS,
                            lastOMT=lastOMT,
                            alpha_0=alpha_0,
                            alpha_method=type_of_alpha,
                            type_of_example=type_of_example,
                            qmax=qmax,
                            cti_list=cti_list_input,
                            S=S_0,
                            idx_S=idx_S_0,
                        )
                        N = len(Pis)
                        # sys.exit()
                        # -------------- Compute probabilities and state estimates ---------------

                        # array that contains the estimates of x.
                        xhat = np.zeros((n, N))  # an n by N array.

                        # This is either missed detection (under Hi) or Correct acceptance (under H0)
                        identH0 = Pis == 0
                        P_acceptH0 = np.sum(identH0) / N  #
                        xhat[:, identH0] = Aplus @ y_sample[:, identH0]

                        # note that this is the array with the correct identifications and wrong identifications under Hi
                        # if under H0, then these correspond to the false alarms P_FA_i
                        P_CI_WI = np.zeros(len(P_region))
                        column_names = ["b_i", "PCA_MD"]
                        for i, (part, cvec) in enumerate(P_region):
                            ## For the
                            Pci_perp = P_perp(P_mat(cvec, Qyy))
                            Abar = Pci_perp @ A
                            Abarplus = plusmat(Abar, Qyy)

                            idx_to_identify = indices_partitions[part]
                            identHi = Pis == idx_to_identify
                            P_CI_WI[i] = np.sum(identHi) / N
                            # append xhat_i to estimates xhat
                            xhat[:, identHi] = Abarplus @ y_sample[:, identHi]
                            column_names.append(
                                "P_FA_{}_CI_{}".format(
                                    part.replace("P", ""), part.replace("P", "")
                                )
                            )

                        # Undecided region
                        identHomega = Pis == -1
                        P_omega = np.sum(identHomega) / N
                        xhat[:, identHomega] = np.nan * np.ones(sum(identHomega))
                        column_names.append("P_OMEGA")
                        all_probs = np.hstack((P_acceptH0, P_CI_WI, P_omega))
                        return_probabilities.append(all_probs)
                        full_xhat[index, :, :] = xhat
                        # print('Now at b_i = {}'.format(b_i))

                        ## Compute probabilities of IR / failures
                        xhat_no_und = xhat[:, ~identHomega]

                        # print("REWRITE THIS CODE BELOW")
                        full_prob_failure[index, :] = P_F_safety_region_GNSS(
                            xhat_no_und, type_of_example, radii_list=radii_Bx
                        )

                        # xhatSquared = np.linalg.norm(xhat_no_und, axis=0)**2
                        # for idx_rad, radius_Bx in enumerate(radii_Bx):
                        #     outside_Bx = (xhatSquared > radius_Bx**2).sum()
                        #     PIR = outside_Bx / len(xhatSquared) # divide by nr of times a sol is provided.
                        #     full_prob_failure[index, idx_rad] = PIR

                    return_probabilities = np.array(return_probabilities)
                    df_to_write = pd.DataFrame(
                        np.hstack(
                            (np.round(bi_range, 3).reshape(-1, 1), return_probabilities)
                        ),
                        columns=column_names,
                    )

                    # Create the string for the filename directory, append the DS type if carrying out data snooping or IDS.
                    if (
                        type_of_testing == "IDS"
                        or type_of_testing == "DS"
                        or type_of_testing == "R_IDS"
                    ):
                        dir_fname = os.path.join(
                            writing_dir,
                            type_of_testing,
                            type_of_DS,
                            type_of_alpha,
                            "Hypothesis" + str(hypt_to_model),
                        )
                    else:
                        dir_fname = os.path.join(
                            writing_dir,
                            type_of_testing,
                            "Hypothesis" + str(hypt_to_model),
                        )

                    # Make directory if not exists
                    if not os.path.exists(dir_fname):
                        os.makedirs(dir_fname)

                    fname = os.path.join(dir_fname, "PCA_FA_CI_WI.csv")

                    if os.path.exists(fname):
                        # If file exists, append the data to the existing file without overwriting
                        df_to_write.to_csv(fname, mode="a", header=False, index=False)
                        print(f"Data appended to existing file: {fname}")
                    else:
                        # If file does not exist, create a new one
                        df_to_write.to_csv(fname, index=False)
                        print(f"New file created: {fname}")

                    # Instead of writing x_hat completely to a data file, we first compute P(IR) and write that.
                    # fname_x = os.path.join(dir_fname, 'xhat_estimates_run_{}.csv'.format(simnr))
                    # towrite = bi_range.reshape(-1,1)
                    # for zz in range(n):
                    #     towrite = np.hstack((towrite, full_xhat[:,zz,:]))
                    # np.savetxt(fname_x, towrite, delimiter=',')

                    # Write P(IR) to a csv file
                    f_name_IR = os.path.join(
                        dir_fname, "IR_estimates_run_{}.csv".format(simnr)
                    )
                    towrite = np.round(bi_range.reshape(-1, 1), 3)
                    towrite = np.hstack((towrite, full_prob_failure))
                    np.savetxt(f_name_IR, towrite, delimiter=",")

                elif C_i_model.shape[1] == 2:
                    return_probabilities = []
                    hypt_to_model = indices_partitions[partition_to_model]
                    C_bar = P_perp(P_mat(A, Qyy)) @ C_i_model
                    Qbb = np.linalg.inv(C_bar.T @ np.linalg.inv(Qyy) @ C_bar)
                    sigma_b1 = Qbb[0, 0] ** 0.5
                    sigma_b2 = Qbb[1, 1] ** 0.5

                    b1 = np.arange(-21, 23, 2) * sigma_b1
                    b2 = np.arange(1, 23, 2) * sigma_b2

                    if bool_much_larger_biases:
                        b1 = np.linspace(-50, 50, len(b1)) * sigma_b1
                        b2 = np.linspace(1, 50, len(b2)) * sigma_b2

                    # Generate meshgrid
                    B1, B2 = np.meshgrid(b1, b2, indexing="ij")

                    # Stack and reshape into a 2D array where each row is (b1_i, b2_j)
                    b_i_array = np.column_stack([B1.ravel(), B2.ravel()]).T
                    n_b_samples = b_i_array.shape[1]
                    full_xhat = np.zeros((n_b_samples, n, N))
                    full_prob_failure = np.zeros((n_b_samples, len(radii_Bx)))

                    print("Simulation {}".format(simnr + 1))
                    # for q_i=1, we only have one b_i value..
                    all_IR_components = np.zeros(
                        (n_b_samples, len(radii_Bx), 1 + len(P_region))
                    )

                    for index, [b_1i, b_2i] in enumerate(zip(B1.ravel(), B2.ravel())):
                        # b_i = np.array([[b_i]])
                        # b_i=np.array([[12]])
                        b_i = np.array([[b_1i], [b_2i]])

                        y_mean = C_i_model @ b_i  # we implicitly assume that x=0
                        y_sample = np.random.multivariate_normal(
                            y_mean.flatten(), Qyy, size=(N,)
                        ).T

                        t_sample = B_T @ y_sample

                        Pis = compute_Pis(
                            t_sample,
                            Qtt,
                            B_T,
                            alpha_prime,
                            type_of_testing=type_of_testing,
                            P_region=P_region,
                            type_of_DS=type_of_DS,
                            lastOMT=lastOMT,
                            alpha_0=alpha_0,
                            type_of_example=type_of_example,
                            qmax=qmax,
                            cti_list=cti_list_input,
                            S=S_0,
                            idx_S=idx_S_0,
                        )

                        # print('Now at b_1i, b_2i', b_1i, b_2i)
                        # for el in range( int(m+m*(m-1)/2+2)):
                        #     idents = np.sum(Pis == el)
                        #     for keys, vals in indices_partitions.items():
                        #         if vals == el:
                        #             part = keys
                        #     print('At i, for {}, total identifications = {}'.format(part, idents))

                        # sys.exit()
                        N = len(Pis)

                        # -------------- Compute probabilities and state estimates ---------------

                        # array that contains the estimates of x.
                        xhat = np.zeros((n, N))  # an n by N array.

                        # This is either missed detection (under Hi) or Correct acceptance (under H0)
                        identH0 = Pis == 0
                        P_acceptH0 = np.sum(identH0) / N  #
                        # Undecided region
                        identHomega = Pis == -1
                        P_omega = np.sum(identHomega) / N

                        # # This is for H0.
                        # xhat[:, identH0] = Aplus @ y_sample[:, identH0]
                        # for idx_rad, radius_Bx in enumerate(radii_Bx):
                        #     xhat0_squared = np.linalg.norm(xhat[:, identH0], axis=0)**2
                        #     IR_components_part = (xhat0_squared > radius_Bx**2).sum()/N * (1 / (1-P_omega))
                        #     all_IR_components[index, idx_rad, 0]  = IR_components_part

                        # note that this is the array with the correct identifications and wrong identifications under Hi
                        # if under H0, then these correspond to the false alarms P_FA_i
                        P_CI_WI = np.zeros(len(P_region))
                        column_names = ["b_1i", "b_2i", "PCA_MD"]
                        column_names_PIR_comps = ["b_1i", "b_2i", "P0"]
                        for i, (part, cvec) in enumerate(P_region):
                            ## For the
                            Pci_perp = P_perp(P_mat(cvec, Qyy))
                            Abar = Pci_perp @ A
                            Abarplus = plusmat(Abar, Qyy)

                            idx_to_identify = indices_partitions[part]
                            identHi = Pis == idx_to_identify
                            P_CI_WI[i] = np.sum(identHi) / N
                            # append xhat_i to estimates xhat
                            xhat[:, identHi] = Abarplus @ y_sample[:, identHi]
                            column_names.append(
                                "P_FA_{}_CI_{}".format(
                                    part.replace("P", ""), part.replace("P", "")
                                )
                            )
                            # column_names_PIR_comps.append(part)
                            # for idx_rad, radius_Bx in enumerate(radii_Bx):
                            #     xhati_squared = np.linalg.norm(xhat[:, identHi], axis=0)**2
                            #     IR_components_part = (xhati_squared > radius_Bx**2).sum()/N * (1 / (1-P_omega))
                            #     all_IR_components[index, idx_rad, i+1]  = IR_components_part

                        xhat[:, identHomega] = np.nan * np.ones(sum(identHomega))
                        column_names.append("P_OMEGA")
                        all_probs = np.hstack((P_acceptH0, P_CI_WI, P_omega))
                        return_probabilities.append(all_probs)
                        full_xhat[index, :, :] = xhat
                        # print('Now at b_i = {}'.format(b_i))

                        ## Compute probabilities of IR / failures
                        xhat_no_und = xhat[:, ~identHomega]
                        xhatSquared = np.linalg.norm(xhat_no_und, axis=0) ** 2
                        for idx_rad, radius_Bx in enumerate(radii_Bx):
                            outside_Bx = (xhatSquared > radius_Bx**2).sum()
                            PIR = outside_Bx / len(
                                xhatSquared
                            )  # divide by nr of times a sol is provided.
                            full_prob_failure[index, idx_rad] = PIR

                    return_probabilities = np.array(return_probabilities)
                    df_to_write = pd.DataFrame(
                        np.hstack((b_i_array.T, return_probabilities)),
                        columns=column_names,
                    )

                    # Create the string for the filename directory, append the DS type if carrying out data snooping or IDS.
                    if (
                        type_of_testing == "IDS"
                        or type_of_testing == "DS"
                        or type_of_testing == "R_IDS"
                    ):
                        dir_fname = os.path.join(
                            writing_dir,
                            type_of_testing,
                            type_of_DS,
                            "Hypothesis" + str(hypt_to_model),
                        )
                    else:
                        dir_fname = os.path.join(
                            writing_dir,
                            type_of_testing,
                            "Hypothesis" + str(hypt_to_model),
                        )

                    if bool_much_larger_biases:
                        dir_fname = os.path.join(dir_fname, "much_larger_biases")
                    # Make directory if not exists
                    if not os.path.exists(dir_fname):
                        os.makedirs(dir_fname)

                    fname = os.path.join(dir_fname, "PCA_FA_CI_WI.csv")

                    if os.path.exists(fname):
                        # If file exists, append the data to the existing file without overwriting
                        df_to_write.to_csv(fname, mode="a", header=False, index=False)
                        print(f"Data appended to existing file: {fname}")
                    else:
                        # If file does not exist, create a new one
                        df_to_write.to_csv(fname, index=False)
                        print(f"New file created: {fname}")

                    # ## Write the components
                    # dirname_IR_comps = os.path.join(dir_fname, 'IR_Components')
                    # if not os.path.exists(dirname_IR_comps):
                    #     os.makedirs(dirname_IR_comps)

                    # for idx_rad, radius_Bx in enumerate(radii_Bx):
                    #     mean_IR = all_IR_components[:, idx_rad, :].mean(axis=0)
                    #     std_IR = all_IR_components[:, idx_rad, :].std(axis=0, ddof=1)
                    #     components_rad = all_IR_components[:, idx_rad, :]

                    #     ## The mean of the components
                    #     fname_IRcomps = os.path.join(dirname_IR_comps, 'All_components_P_IR_R={}_for_{}_simulations.csv'.format(radius_Bx, N_sims))

                    #     df_to_write_components_IR=pd.DataFrame(np.hstack((b_i_array.T, components_rad)), columns = column_names_PIR_comps)

                    #     if os.path.exists(fname_IRcomps):
                    #         # If file exists, append the data to the existing file without overwriting
                    #         df_to_write.to_csv(fname_IRcomps, mode='a', header=False, index=False)
                    #         print(f"Data appended to existing file: {fname_IRcomps}")
                    #     else:
                    #         # If file does not exist, create a new one
                    #         df_to_write.to_csv(fname_IRcomps, index=False)
                    #         print(f"New file created: {fname_IRcomps}")

                    # Instead of writing x_hat completely to a data file, we first compute P(IR) and write that.
                    # fname_x = os.path.join(dir_fname, 'xhat_estimates_run_{}.csv'.format(simnr))
                    # towrite = bi_range.reshape(-1,1)
                    # for zz in range(n):
                    #     towrite = np.hstack((towrite, full_xhat[:,zz,:]))
                    # np.savetxt(fname_x, towrite, delimiter=',')

                    # Write P(IR) to a csv file
                    f_name_IR = os.path.join(
                        dir_fname, "IR_estimates_run_{}.csv".format(simnr)
                    )
                    towrite = b_i_array.T
                    towrite = np.hstack((towrite, full_prob_failure))
                    np.savetxt(f_name_IR, towrite, delimiter=",")

            logstring = f""" Ran on {datetime.datetime.today().date()}
                Parameters that were used in the simulation.
                m = {m}
                n = {n}
                r = {r}
                A = {A}
                sigma = {sigma}
                Qyy = {Qyy}
                B_T = {B_T} 
                type_of_testing = {type_of_testing}
                lastOMT = {lastOMT} 
                type_of_DS = {type_of_DS}
                alpha_prime = {alpha_prime}
                alpha_0 = {alpha_0}
                radii_Bx = {radii_Bx}
                under_hypt = {hypt_to_model}
                N_sims = {N_sims}
                
            Instead of xhat, we decided to first compute the IR and write that data to a csv file.
            We did so in the file IR_estiamtes_run_(simnr).csv. The first column is the b_i value. 
            The second to last columns are the corresponding IRs for varying radii of Bx (circular). 
            See parameter radii_Bx above to see which are the values. 
            """
            with open(os.path.join(dir_fname, "README.txt"), "w") as f:
                f.write(logstring)
