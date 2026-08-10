# -*- coding: utf-8 -*-
"""
Description:
    This program loads data run previously on DelftBlue, for both correct and wrong identification probabilities, under
    both q=1 and q=2 hypothesis.

    We consider only H13 for the SPP_GNSS example for the paper on datasnooping.

    For IDS C, RIDS C and trad. DIA we compute the probabilities of masking and swamping,
    which are simply summations of particular wrong model selections.
"""

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import sys
import os
import re
import glob

sys.path.append("C:/Users/bgvannoort/Documents/IDS/Code")

import Functions


def extract_bval(filename, pattern=r"bval2_([0-9.]+)\.csv"):
    match = re.search(pattern, filename)
    if match:
        return float(match.group(1))
    else:
        return None


def get_data_2Dhypt(fulldirname, bval1):
    # get the files containing the data
    file_names_PCA = glob.glob(
        os.path.join(fulldirname, f"PCA_FA_CI_WI_bval1_{bval1:.2f}_bval2*.csv")
    )
    file_names_IR = glob.glob(
        os.path.join(fulldirname, f"IR_estimates_bval1_{bval1:.2f}_bval2*.csv")
    )

    # print('filenamesPCA', file_names_PCA)
    # print(os.path.join(fulldirname, 'IR_estimates_bval1_{bval1:.2f}_bval2*.csv'))

    # Extract bvals and pair with filenames
    bval_file_pairs_PCA = [(extract_bval(f), f) for f in file_names_PCA]
    bval_file_pairs_IR = [(extract_bval(f), f) for f in file_names_IR]

    # Filter out any None values just in case
    bval_file_pairs_PCA = [pair for pair in bval_file_pairs_PCA if pair[0] is not None]
    bval_file_pairs_IR = [pair for pair in bval_file_pairs_IR if pair[0] is not None]

    # Sort by bval
    bval_file_pairs_PCA.sort(key=lambda x: x[0])
    bval_file_pairs_IR.sort(key=lambda x: x[0])

    # Get sorted filenames and bvals for the second bias
    sorted_files_PCA = [pair[1] for pair in bval_file_pairs_PCA]
    bvals_PCA = [pair[0] for pair in bval_file_pairs_PCA]

    sorted_files_IR = [pair[1] for pair in bval_file_pairs_IR]
    bvals_IR = [pair[0] for pair in bval_file_pairs_IR]

    # Change the sorted file names list to the original filename
    file_names_PCA = sorted_files_PCA
    file_names_IR = sorted_files_IR

    ## Get the first file to get the column names
    # Column names
    df_data_bval_PCI_PWI = pd.read_csv(file_names_PCA[0], delimiter=",")
    df_data_bval_IR = pd.read_csv(file_names_IR[0])

    cols_PCA = df_data_bval_PCI_PWI.columns.tolist()
    cols_IR = df_data_bval_IR.columns.tolist()

    mean_data_PCA = np.zeros((len(file_names_PCA), len(cols_PCA) - 2))
    std_data_PCA = np.zeros((len(file_names_PCA), len(cols_PCA) - 2))

    mean_data_IR = np.zeros((len(file_names_IR), len(cols_IR) - 2))
    std_data_IR = np.zeros((len(file_names_IR), len(cols_IR) - 2))

    b_array = np.zeros(len(file_names_PCA))
    idx_b1 = 0
    for file_data_PCA, file_data_IR in zip(file_names_PCA, file_names_IR):
        df_data_bval_PCA = pd.read_csv(file_data_PCA, delimiter=",")
        df_data_bval_IR = pd.read_csv(file_data_IR)

        # try:
        # convert to arrays
        arr_data_bval_PCA = df_data_bval_PCA.to_numpy()
        arr_data_bval_IR = df_data_bval_IR.to_numpy()

        # Compute and store the means and stds
        # For testing
        mean_data_to_use_PCA = np.mean(arr_data_bval_PCA, axis=0)
        std_data_to_use_PCA = np.std(arr_data_bval_PCA, ddof=1, axis=0)

        mean_data_PCA[idx_b1, :] = mean_data_to_use_PCA[2:]
        std_data_PCA[idx_b1, :] = std_data_to_use_PCA[2:]

        ## For the IRs
        mean_data_to_use_IR = np.mean(arr_data_bval_IR, axis=0)
        std_data_to_use_IR = np.std(arr_data_bval_IR, axis=0, ddof=1)

        mean_data_IR[idx_b1, :] = mean_data_to_use_IR[2:]
        std_data_IR[idx_b1, :] = std_data_to_use_IR[2:]

        radii_IR = df_data_bval_IR.columns[2:]
        # except:
        #     pass
        idx_b1 += 1
    return (
        mean_data_PCA,
        std_data_PCA,
        bvals_PCA,
        cols_PCA,
        mean_data_IR,
        std_data_IR,
        bvals_IR,
        cols_IR,
        radii_IR,
    )


if __name__ == "__main__":
    indices_partitions, colors_indices = Functions.load_indices_partitions_GNSS_ex(
        False, load_colors=True
    )
    type_of_example = "SPP_GNSS"

    type_of_DS = "C"
    type_of_alpha = "Kok_IDS"
    procedures = ["IDS", "R_IDS", "classical DIA"]
    alpha_0 = 0.01

    qmax = 2  # Hard code based on part_hypt

    part_hypt = "P13"
    store_IR_grid = np.zeros((41, 21))  # change to correct dimensions..
    # store_IR_grid = np.zeros((len(combinations), 4, 22, 11)) # change to correct dimensions..

    # combinations = [['R_IDS', 'A', 'P13']]

    # b1value range
    b1_range = np.arange(-20, 21, 1)

    # b2value range -- necessary only for the shaping of the arrays. Default is 0 to 20.
    b2_range = np.arange(0, 21, 1)

    # The true set S
    S_true = [1, 3]  # observation 1 and 3 are truly faulty

    for type_of_testing in procedures:
        # plt.close('all')

        hypt_idx = indices_partitions[part_hypt]

        main_dir = (
            r"C:\Users\bgvannoort\Documents\IDS\Results\TestingProbabilities\SPP_GNSS"
        )

        if type_of_testing != "classical DIA":
            fulldir = os.path.join(
                main_dir,
                type_of_testing,
                type_of_DS,
                type_of_alpha,
                "alpha_0_" + str(alpha_0),
                "Hypothesis{}".format(hypt_idx),
                "qmax={}".format(qmax),
            )
        else:
            fulldir = os.path.join(
                main_dir,
                type_of_testing,
                "alpha_prime=0.038",
                "Hypothesis{}".format(hypt_idx),
            )
            # fulldir = os.path.join(main_dir, type_of_testing,  'Hypothesis{}'.format(hypt_idx))

        for idxb1, b1 in enumerate(b1_range):
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
            ) = get_data_2Dhypt(fulldir, b1)

            if idxb1 == 0:
                # Initialize the data arrays for the IR
                # Initialize the data arrays for testing
                mean_data_PCA_all = np.zeros(
                    (len(b1_range), len(b2_range), len(cols_PCA) - 2)
                )
                std_data_PCA_all = np.zeros(
                    (len(b1_range), len(b2_range), len(cols_PCA) - 2)
                )

                mean_data_IR_all = np.zeros(
                    (len(b1_range), len(b2_range), len(cols_IR) - 2)
                )
                std_data_IR_all = np.zeros(
                    (len(b1_range), len(b2_range), len(cols_IR) - 2)
                )

            for idxb2, b2 in enumerate(bvals_PCA):
                # store the data
                mean_data_PCA_all[idxb1, idxb2, :] = mean_data_PCA[idxb2, :]
                std_data_PCA_all[idxb1, idxb2, :] = std_data_PCA[idxb2, :]

            for idxb2, b2 in enumerate(bvals_IR):
                mean_data_IR_all[idxb1, idxb2, :] = mean_data_IR[idxb2, :]
                std_data_IR_all[idxb1, idxb2, :] = std_data_IR[idxb2, :]

        # Determine here the swamping and masking events
        cols_swamping = []  # contains indices from mean_data_IR_all array where  swamping occurred
        cols_masking = []  # contains indices ... where masking occurred.

        for idx_colPCA, col_label in enumerate(cols_PCA):
            idx_to_add = (
                idx_colPCA - 2
            )  # subtract directly the two columns corresponding to the b1, b2

            if col_label == "b_1i" or col_label == "b2i" or col_label == "P_OMEGA":
                continue
            elif col_label == "PCA_MD":  # missed detection
                cols_masking.append(idx_to_add)

            else:
                # Extract all numbers from each string
                numbers_only = re.findall(r"\d+", col_label)

                unique_nrs = []
                for el in numbers_only:
                    el = int(el)
                    if el not in unique_nrs:
                        unique_nrs.append(el)

                # Loop over the identified observations; in this example, since qmax=2
                # We will almost always have that masking and swamping occur similarly except once a Hi with q=1 hypt contains the correct outlier.

                # split into S_identified is from q=1 hypt or q=2
                if len(unique_nrs) == 1:
                    if unique_nrs[0] in S_true:  # correctly identify one outlier
                        cols_masking.append(idx_to_add)
                    else:  # incorrectly identify a wrong outlier
                        cols_masking.append(idx_to_add)
                        cols_swamping.append(idx_to_add)

                else:
                    if (
                        unique_nrs[0] in S_true and unique_nrs[1] in S_true
                    ):  # correct identification
                        continue  # neither masking nor swamping
                    else:
                        # it is both masking and swamping.
                        cols_masking.append(idx_to_add)
                        cols_swamping.append(idx_to_add)

        cols_masking = np.array(cols_masking, dtype=int)
        cols_swamping = np.array(cols_swamping, dtype=int)

        # print('cols_masking', cols_masking)
        # print('cols_swamping', cols_swamping)

        b1_to_eval_init = 21
        b2_to_eval_init = 0

        bvals_to_consider = [[15, 10], [10, 15], [-15, 10], [7, 5]]
        idxes_bval_to_consider = [[36, 10], [31, 15], [5, 10], [28, 5]]

        counter = 0
        for idxb1, idxb2 in idxes_bval_to_consider:
            bvals = bvals_to_consider[counter]

            print("idx_b1, idxb2", idxb1, idxb2)

            P_masking = mean_data_PCA_all[idxb1, idxb2, cols_masking].sum()
            P_swamping = mean_data_PCA_all[idxb1, idxb2, cols_swamping].sum()

            print(
                rf"Masking probabilities for {type_of_testing} {type_of_DS} are {np.round(P_masking, 2)} at b1={bvals[0]}, b2={bvals[1]}"
            )
            print(
                rf"Swamping probabilities for {type_of_testing} {type_of_DS} are {np.round(P_swamping, 2)} at b1={bvals[0]}, b2={bvals[1]}"
            )
            counter += 1
