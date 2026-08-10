"""
Description of code:
    Evaluate the total mean penalties for all hypts, inputting the probabilities
    for the hypotheses.

    The results of the simulations should be stored in directories with main directory
    being the example_type/penalty_funcs directory.
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import glob
import re

from matplotlib.patches import Circle
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable

from Functions import (
    load_indices_partitions_GNSS_ex,
    setup_SPP_GNSS_example,
    get_idx_hypt_RIDS,
)
from Functions import load_setup_parameters


def extract_bval2(filename, pattern=r"bval2_([0-9.]+)\.csv"):
    match = re.search(pattern, filename)
    if match:
        return float(match.group(1))
    else:
        return None


def get_data(fulldirname, bval1):
    # get the files containing the data
    file_names_PCA = glob.glob(
        os.path.join(fulldirname, f"PCA_FA_CI_WI_bval_{bval1:.2f}*.csv")
    )
    file_names_IR = glob.glob(
        os.path.join(fulldirname, f"IR_estimates_bval_{bval1:.2f}*.csv")
    )

    # Extract bvals and pair with filenames
    bval_file_pairs_PCA = [(extract_bval(f), f) for f in file_names_PCA]
    bval_file_pairs_IR = [(extract_bval(f), f) for f in file_names_IR]

    # Filter out any None values just in case
    bval_file_pairs_PCA = [pair for pair in bval_file_pairs_PCA if pair[0] is not None]
    bval_file_pairs_IR = [pair for pair in bval_file_pairs_IR if pair[0] is not None]

    # Sort by bval
    bval_file_pairs_PCA.sort(key=lambda x: x[0])
    bval_file_pairs_IR.sort(key=lambda x: x[0])

    # Get sorted filenames and bvals
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

    mean_data_PCA = np.zeros((len(file_names_PCA), len(cols_PCA) - 1))
    std_data_PCA = np.zeros((len(file_names_PCA), len(cols_PCA) - 1))

    mean_data_IR = np.zeros((len(file_names_IR), len(cols_IR) - 1))
    std_data_IR = np.zeros((len(file_names_IR), len(cols_IR) - 1))

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

        mean_data_PCA[idx_b1, :] = mean_data_to_use_PCA[1:]
        std_data_PCA[idx_b1, :] = std_data_to_use_PCA[1:]

        ## For the IRs ---> This does not go right yet, saved the files incorrectly on DelftBlue
        mean_data_to_use_IR = np.mean(arr_data_bval_IR, axis=0)
        std_data_to_use_IR = np.std(arr_data_bval_IR, axis=0, ddof=1)

        mean_data_IR[idx_b1, :] = mean_data_to_use_IR[1:]
        std_data_IR[idx_b1, :] = std_data_to_use_IR[1:]

        radii_IR = df_data_bval_IR.columns[1:]
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


def extract_bval(filename, pattern=r"bval_([0-9.]+)\.csv"):
    match = re.search(pattern, filename)
    if match:
        return float(match.group(1))
    else:
        return None


def get_data_2Dhypt(fulldirname, bval1, bval2):
    # get the files containing the data
    file_names_PCA = glob.glob(
        os.path.join(
            fulldirname, f"PCA_FA_CI_WI_bval1_{bval1:.2f}_bval2_{bval2:.2f}*.csv"
        )
    )
    file_names_IR = glob.glob(
        os.path.join(
            fulldirname, f"IR_estimates_bval1_{bval1:.2f}_bval2_{bval2:.2f}*.csv"
        )
    )

    # print('filenamesPCA', file_names_PCA)
    # print(os.path.join(fulldirname, 'IR_estimates_bval1_{bval1:.2f}_bval2*.csv'))

    # Extract bvals and pair with filenames
    bval_file_pairs_PCA = [(extract_bval2(f), f) for f in file_names_PCA]
    bval_file_pairs_IR = [(extract_bval2(f), f) for f in file_names_IR]

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


# %%
indices_partitions, colors_indices = load_indices_partitions_GNSS_ex(
    False, load_colors=True
)
type_of_example = "SPP_GNSS"

type_of_alpha = "Kok_IDS"
# type_of_alpha = 'manual'

bool_much_larger_biases = False
qmax = 2

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


bool_close_all_figs = True

all_PWIs = []

combinations = []
for type_of_testing in ["classical DIA"]:
    for type_of_DS in ["C"]:
        for outlier1 in range(1, 8):
            for outlier2 in range(outlier1 + 1, 8):
                part_hypt = "P" + str(outlier1) + str(outlier2)
                # for outlier1 in range(1, 2):
                #     for outlier2 in range(3, 4):
                #         part_hypt= 'P'+str(outlier1) + str(outlier2)
                combinations.append(
                    [type_of_testing, type_of_DS, part_hypt, (outlier1, outlier2)]
                )
        if type_of_testing == "classical DIA":
            break

alpha = 0.05

## Initialize here the probabilities
PH0 = 1 - alpha
scenario_type = 1

m, n, r, A, _, sigma, Qyy, Qyy_inv, B_T, Qtt, Qtt_inv = setup_SPP_GNSS_example(
    alpha_0=0.01, alpha_method=type_of_alpha
)


p_o = 0.00730282  # probability of one outlier follows from alpha

k = m + m * (m - 1) / 2  #

if scenario_type == 1:
    Phi_q1 = p_o * (1 - p_o) ** 6
    Phi_q2 = p_o**2 * (1 - p_o) ** 5
elif scenario_type == 2:
    Phi_q1 = alpha / k
    Phi_q2 = alpha / k

# Phi_q2 = Phi_q2 * 10
PHi_list = [Phi_q1] * m + [Phi_q2] * int(m * (m - 1) / 2)


all_sigmas = np.sqrt(np.diag(Qyy))
total_penalty = PH0 * alpha  # wrong decision under H0
at_x_sigma = 5


## For the 1D hypotheses.
for i_hypt in range(m):
    b1_val = all_sigmas[i_hypt] * at_x_sigma

    if type_of_example == "SPP_GNSS":
        indices_partitions, colors_indices = load_indices_partitions_GNSS_ex(
            False, load_colors=True
        )
        alpha_prime_string = "alpha_prime=0.05"
        ## Check if dir exists for writing the results to for PCI/PWI PCD/PMD etc.

        
        main_dir = r'D:\Documents_from_TUDelft_laptop\IDS\Results\TestingProbabilities\SPP_GNSS\penalty_funcs' 
    hypt_idx = i_hypt + 1

    if type_of_testing != "classical DIA":
        alpha_0 = alpha_0_dictionary[type_of_testing + "_" + type_of_DS]

        fulldir = os.path.join(
            main_dir,
            type_of_testing,
            type_of_DS,
            type_of_alpha,
            f"alpha_0_{alpha_0}",
            "Hypothesis{}".format(hypt_idx),
            "qmax={}".format(qmax),
        )

        testing_string = type_of_testing + "_" + type_of_DS
    else:
        alpha_0 = alpha_0_dictionary[type_of_testing]
        fulldir = os.path.join(
            main_dir,
            type_of_testing,
            alpha_prime_string,
            "Hypothesis{}".format(hypt_idx),
        )

        testing_string = type_of_testing

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
    ) = get_data(fulldir, b1_val)

    shape1, shape2 = mean_data_PCA.shape
    if shape1 > 1:
        raise Exception("There can only be one value one row in mean_data_PCA")
    idx_PCI = i_hypt + 1
    # print("PCI_hyptidx", mean_data_PCA[0,idx_PCI])

    PWI_i_hypt = 1 - mean_data_PCA[0, idx_PCI]
    total_penalty += PHi_list[i_hypt] * PWI_i_hypt

    all_PWIs.append(PWI_i_hypt)


if type_of_testing == "DS":
    # DS cannot identify 2D hypotheses, so PWI = 1 always
    for idx_hypt, Phypt in enumerate(PHi_list):
        if idx_hypt > 6:  # these are the 2D hypts
            total_penalty += Phypt
    print(
        f"Total penalty for {type_of_testing} and {type_of_DS} is {np.round(total_penalty, 4)}"
    )
    sys.exit()

## For the 2D hypotheses

for idx_combination, comb in enumerate(combinations):
    # plt.close('all')
    _, _, part_hypt, (outlier1, outlier2) = comb

    hypt_idx = indices_partitions[part_hypt]
    # print('part_hypt, hypt_idx', part_hypt, hypt_idx)

    # print('outlier1, outlier2', outlier1, outlier2)

    sigma1, sigma2 = all_sigmas[outlier1 - 1], all_sigmas[outlier2 - 1]
    b1_range = np.array([-at_x_sigma, at_x_sigma]) * sigma1  # we evaluate it at 3 sigma
    b2_range = np.array([at_x_sigma]) * sigma2  # we evaluate it at 3 sigma

    if type_of_example == "SPP_GNSS":
        indices_partitions = load_indices_partitions_GNSS_ex(
            False
        )  # IDS_sep_order = False

        ## Check if dir exists for writing the results to for PCI/PWI PCD/PMD etc.
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
            alpha_prime_string,
            "Hypothesis{}".format(hypt_idx),
        )
        # fulldir = os.path.join(main_dir, type_of_testing,  'Hypothesis{}'.format(hypt_idx))

    PWI_tot = 0
    for idxb1, b1 in enumerate(b1_range):
        b2_val = b2_range[0]
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
        ) = get_data_2Dhypt(fulldir, b1, b2_val)

        # print('bvalsPCA b1, b2', b1, b2_val)
        # print('mean_data_PCA', mean_data_PCA)

        PCI_hyptidx = mean_data_PCA[0, hypt_idx]
        # print("PCI_hyptidx", PCI_hyptidx)
        PWI_tot_cand = 1 - mean_data_PCA[0, hypt_idx]
        if PWI_tot_cand > PWI_tot:
            PWI_tot = PWI_tot_cand

    # sys.exit()
    all_PWIs.append(PWI_tot)
    # print('hypt_idx-1, PHilist', hypt_idx-1, PHi_list[hypt_idx-1])
    total_penalty += PHi_list[hypt_idx - 1] * PWI_tot
    # print("P(hypt) ", PHi_list[hypt_idx-1])


print(
    f"Total penalty for {type_of_testing} and {type_of_DS} is {np.round(total_penalty, 5)}"
)
