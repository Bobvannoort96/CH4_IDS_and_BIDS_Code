"""
This python program plots the partitioning of the 'classical DIA' method which
computes the test statistics for all q=1 and q=2 hypotheses and compares them all together.

Author: Bob van Noort
Date: 2 June
"""

import numpy as np
import sys
import scipy
import scipy.stats
import matplotlib.pyplot as plt
import pickle
import os
import time
import json

from Functions import *


def load_color_dict_GNSS_Ex(
    svFile=r"C:\\Users\\bgvannoort\\Documents\\IDS\\Sim Data\\SPP_GNSS\\colors_partitions_dict.json",
):

    with open(svFile, "r") as f:
        loaded_dict = json.load(f)

    return loaded_dict


## Ordinary DIA
@timing
def ordinary_DIA(
    factor, plot_figures=False, write_grid_to_file=False, type_of_example="simple"
):

    if type_of_example == "simple":
        colors_partitions = {
            "P0": "black",
            "P1": "yellow",
            "P2": "red",
            "P3": "blue",
            "P4": "grey",
            "P12": "orange",
            "P13": "green",
            "P14": "gold",
            "P21": "darkorange",
            "P23": "purple",
            "P24": "lightsalmon",
            "P31": "lime",
            "P32": "indigo",
            "P34": "cornflowerblue",
            "P41": "darkgoldenrod",
            "P42": "tomato",
            "P43": "lightsteelblue",
        }

    elif type_of_example == "SPP_GNSS" or type_of_example == "ARAIM_UNDEC_GNSS":
        colors_partitions = load_color_dict_GNSS_Ex()
    inPlane = False

    m, n, r, A, alpha, sigma, Qyy, Qyy_inv, Bt, Qtt, Qtt_inv = load_setup_parameters(
        type_of_example, alpha_0=0.01, alpha_method="Kok_IDS"
    )

    print("alpha used", alpha)
    print("A:", A)
    ## Generate t-space on a sphere.
    n_samples_x, n_samples_y = 1000, 1000
    x, y, z, t = generate_t_grid(n_samples_x, n_samples_y)

    t = t * factor
    identifications_all = np.zeros(t.shape[1])

    threshold_OMT = scipy.stats.chi2.isf(alpha, df=m - n)
    t_OMT = np.einsum("mi,ij,jm->m", t.T, Qtt_inv, t)
    (P0_indices,) = np.where(t_OMT < threshold_OMT)

    (P_outside_ind,) = np.where(t_OMT > threshold_OMT)

    ## carry out the hypothesis testing with the normal dia method
    c_list = []
    P_region = []
    fault_vectors = np.zeros((r, m))
    for i in range(m):
        c_i = np.eye(m)[:, i].reshape(-1, 1)
        c_list.append(c_i)
        fault_vectors[:, i] = (Bt @ c_i).flatten()
        P_region.append(["P" + str(i + 1), Bt @ c_i])
    for i in range(m):
        c_i = np.eye(m)[:, i].reshape(-1, 1)
        for j in range(i + 1, m):
            c_j = np.eye(m)[:, j].reshape(-1, 1)
            cij = np.hstack((c_i, c_j))
            c_list.append(cij)
            P_region.append(["P" + str(i + 1) + str(j + 1), [Bt @ c_i, Bt @ c_j]])

    # Construct the dictionary ourselves, i.e. such that it is always the same
    indices_partitions = {"P0": 0}
    for idx, (Part_, _) in enumerate(P_region):
        indices_partitions[Part_] = idx + 1  # make sure to skip idx=0 as that is H0
    indices_partitions["P99"] = 99
    # print('P_region; order of hypotheses')
    # print(P_region)
    print("indices partitions")
    print(indices_partitions)

    # clist contains all ci and cij vectors, i.e. for q=1 and q=2.
    # We should transform a test statistic for q=2 to one with a N(0,1) distribution,
    # such that we can compare the w-tests for q=1 'fairly' with those of q=2.

    # the P0 region for the normal DIA method is exactly identical to the one as before.
    w_tests_normal_DIA = np.zeros((len(c_list), len(P_outside_ind)))
    # this stores the cumulative density function, i.e. S_i = CDF_{chi^2_q_i}(Tq_i)
    S_array = np.zeros((len(c_list), len(P_outside_ind)))

    t_rej = t[:, P_outside_ind]  # rejected samples / t-grid points

    # test for only identifying multidimensional hypotheses
    S_array = np.ones((len(c_list), len(P_outside_ind)))

    for i, c_vec in enumerate(c_list):
        q = c_vec.shape[1]
        # # If only identifying multiple-hypotheses
        # if q == 1:
        #     continue

        cti = Bt @ c_vec
        Pcti = P_mat(cti, Qtt)
        t_proj = Pcti @ t_rej
        Tq_test = np.einsum("mi,ij,jm->m", t_proj.T, Qtt_inv, t_proj)

        Tq_test_alpha = scipy.stats.chi2.sf(Tq_test, df=q)
        # it is 1-CDF of alpha in fact! So we should take argmin later!
        S_array[i, :] = Tq_test_alpha

    Pijs = (
        np.argmin(S_array, axis=0) + 1
    )  # P0=0, P1=1, P2=2 etc.., matlab indexing starts at 1 as well.

    identifications_all[P_outside_ind] = Pijs

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "3d"})
    for i, c_vec in enumerate(c_list):
        partition, _ = P_region[i]

        wdir = (
            r"C:\Users\bgvannoort\Documents\IDS\Sim Data\SPP_GNSS\Tq_partition_2D_hypts"
        )

        # xx, yy, zz = write_data_for_matlab(x, y, z, identifications_all, partition, inPlane=inPlane,
        #                                    DIA='ordinary DIA',
        #                                    write_grid_to_file=write_grid_to_file,
        #                                    factor=factor, type_of_example=type_of_example,
        #                                    indices_partitions=indices_partitions, manualPath=True, workingDir=wdir)
        xx, yy, zz = write_data_for_matlab(
            x,
            y,
            z,
            identifications_all,
            partition,
            inPlane=inPlane,
            DIA="ordinary DIA",
            write_grid_to_file=write_grid_to_file,
            factor=factor,
            type_of_example=type_of_example,
            indices_partitions=indices_partitions,
        )
        if plot_figures:
            ax.plot_surface(
                xx,
                yy,
                zz,
                color=colors_partitions[partition],
                alpha=1.0,
                edgecolor=None,
                facecolor=colors_partitions[partition],
            )
            ax.set_xlabel("$t_1$")
            ax.set_ylabel("$t_2$")
            ax.set_zlabel("$t_3$")
            ax.set_title("Partitions for ordinary DIA")
            ax.view_init(40, 40)
            ax.legend()
        else:
            plt.close(fig)

    if type_of_example == "simple":
        txtFilename = os.path.join(
            os.getcwd(),
            "..",
            "Sim Data",
            "ordinary_DIA",
            "corresponding_identifications_factor_{}.txt".format(factor),
        )
        # txtFilename = os.path.join(r'C:\Users\bgvannoort\Documents\PhD\Python\IDS_IDS',
        #                            'Sim Data', 'ordinary_DIA', 'corresponding_identifications_factor_{}.txt'.format(factor))

    else:
        txtFilename = os.path.join(
            os.getcwd(),
            "..",
            "Sim Data",
            type_of_example,
            "ordinary_DIA",
            "corresponding_identifications_factor_{}.txt".format(factor),
        )
    np.savetxt(txtFilename, identifications_all)

    if False:
        print("Warning: note sure if this goes right here in the plotting part.")
        for i, (partition, cti) in enumerate(P_region):
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "3d"})
            (indices,) = np.where(identifications_all == i)
            count_identifications_DIA.append([len(indices), partition])
            ax.scatter(
                t[0, indices],
                t[1, indices],
                t[2, indices],
                label=partition,
                marker=".",
                color=colors_partitions[partition],
                alpha=0.7,
            )
            # ax.scatter(t_accept[0,:], t_accept[1,:], t_accept[2,:], label='P0', color='black', alpha=0.7, marker='.')
            if len(partition) == 2:  # q=1
                # plot corresponding fault vector
                cti_plot = np.hstack((cti * 10, cti * -10))
                ax.plot(
                    cti_plot[0, :],
                    cti_plot[1, :],
                    cti_plot[2, :],
                    label="$c_{t_i}$ for " + partition,
                    color="navy",
                )
            else:
                # plot the fault vectors here (note that there are 2 here, so in fact a plane)
                a1, a2 = cti[0], cti[1]
                normal = np.cross(a1.flatten(), a2.flatten())
                d_plane = -np.dot(a1.flatten(), normal)
                if (
                    np.abs(normal[2]) < 1e-8
                ):  ## the normal component of z is 0, i.e. plane lies in whole z-dimension
                    # pass # we need to find solution for this, as the normal component of z is zero
                    # xx, z_val = np.meshgrid(np.arange(tmin, tmax, 1), np.arange(tmin, tmax, 1))
                    xx, yy = np.meshgrid(
                        np.arange(tmin, tmax, 1), np.arange(tmin, tmax, 1)
                    )
                    z_val = (
                        (-normal[0] * xx - normal[1] * yy - d_plane) * 1.0 / normal[2]
                    )
                else:
                    xx, yy = np.meshgrid(
                        np.arange(tmin, tmax, 1), np.arange(tmin, tmax, 1)
                    )
                    z_val = (
                        (-normal[0] * xx - normal[1] * yy - d_plane) * 1.0 / normal[2]
                    )
                # plt3d.plot_surface(xx, yy, z_val, alpha=0.2, label='fault plane for '+partition, color='red')
                ax.plot_surface(
                    xx,
                    yy,
                    z_val,
                    alpha=0.4,
                    label="fault plane for " + partition,
                    color="red",
                )

                ## plot here the two fault lines
                cti1 = np.hstack((a1 * 10, a1 * -10))
                cti2 = np.hstack((a2 * 10, a2 * -10))
                ax.plot(
                    cti1[0, :],
                    cti1[1, :],
                    cti1[2, :],
                    label="$c_{t_i}$ for " + partition,
                    color="navy",
                )
                ax.plot(
                    cti2[0, :],
                    cti2[1, :],
                    cti2[2, :],
                    label="$c_{t_i}$ for " + "P" + partition[::-1][:-1],
                    color="navy",
                    linestyle="--",
                )

            ax.set_xlabel("$t_1$")
            ax.set_ylabel("$t_2$")
            ax.set_zlabel("$t_3$")
            ax.set_title("Partitions for " + partition + " with ordinary DIA")
            ax.view_init(40, 40)
            ax.legend()

        save_interactive_3D(fig, partition, "classical DIA", full_path=False)
    return identifications_all, S_array


## Ordinary DIA
@timing
def ordinary_DIA_penalized(
    factor,
    rho_alpha,
    pi_alpha,
    plot_figures=False,
    write_grid_to_file=False,
    type_of_example="simple",
):
    # rho_alpha is a list of length k with the corresponding rewards for correct identification.
    # pi_alpha is a list of the probability of occurrence for the hypotheses

    if type_of_example == "simple":
        colors_partitions = {
            "P0": "black",
            "P1": "yellow",
            "P2": "red",
            "P3": "blue",
            "P4": "grey",
            "P12": "orange",
            "P13": "green",
            "P14": "gold",
            "P21": "darkorange",
            "P23": "purple",
            "P24": "lightsalmon",
            "P31": "lime",
            "P32": "indigo",
            "P34": "cornflowerblue",
            "P41": "darkgoldenrod",
            "P42": "tomato",
            "P43": "lightsteelblue",
        }
    elif type_of_example == "SPP_GNSS" or type_of_example == "ARAIM_UNDEC_GNSS":
        colors_partitions = load_color_dict_GNSS_Ex()
    inPlane = False

    m, n, r, A, alpha, sigma, Qyy, Qyy_inv, Bt, Qtt, Qtt_inv = load_setup_parameters(
        type_of_example, alpha_0=0.01, alpha_method="Kok_IDS"
    )

    print("alpha used", alpha)
    print("A:", A)
    ## Generate t-space on a sphere.
    n_samples_x, n_samples_y = 1000, 1000
    x, y, z, t = generate_t_grid(n_samples_x, n_samples_y)

    t = t * factor
    identifications_all = np.zeros(t.shape[1])

    threshold_OMT = scipy.stats.chi2.isf(alpha, df=m - n)
    t_OMT = np.einsum("mi,ij,jm->m", t.T, Qtt_inv, t)
    (P0_indices,) = np.where(t_OMT < threshold_OMT)

    (P_outside_ind,) = np.where(t_OMT > threshold_OMT)

    ## carry out the hypothesis testing with the normal dia method
    c_list = []
    P_region = []
    fault_vectors = np.zeros((r, m))
    for i in range(m):
        c_i = np.eye(m)[:, i].reshape(-1, 1)
        c_list.append(c_i)
        fault_vectors[:, i] = (Bt @ c_i).flatten()
        P_region.append(["P" + str(i + 1), Bt @ c_i])
    for i in range(m):
        c_i = np.eye(m)[:, i].reshape(-1, 1)
        for j in range(i + 1, m):
            c_j = np.eye(m)[:, j].reshape(-1, 1)
            cij = np.hstack((c_i, c_j))
            c_list.append(cij)
            P_region.append(["P" + str(i + 1) + str(j + 1), [Bt @ c_i, Bt @ c_j]])

    # Construct the dictionary ourselves, i.e. such that it is always the same
    indices_partitions = {"P0": 0}
    for idx, (Part_, _) in enumerate(P_region):
        indices_partitions[Part_] = idx + 1  # make sure to skip idx=0 as that is H0
    indices_partitions["P99"] = 99
    # print('P_region; order of hypotheses')
    # print(P_region)
    print("indices partitions")
    print(indices_partitions)

    # clist contains all ci and cij vectors, i.e. for q=1 and q=2.
    # We should transform a test statistic for q=2 to one with a N(0,1) distribution,
    # such that we can compare the w-tests for q=1 'fairly' with those of q=2.

    # the P0 region for the normal DIA method is exactly identical to the one as before.
    w_tests_normal_DIA = np.zeros((len(c_list), len(P_outside_ind)))
    # this stores the cumulative density function, i.e. S_i = CDF_{chi^2_q_i}(Tq_i)

    t_rej = t[:, P_outside_ind]  # rejected samples / t-grid points

    # test for only identifying multidimensional hypotheses
    T_alpha_list = np.ones((len(c_list), len(P_outside_ind)))

    for i, c_vec in enumerate(c_list):
        q = c_vec.shape[1]
        # If only identifying multiple-hypotheses
        # if q == 1:
        #     continue

        cti = Bt @ c_vec
        Pcti = P_mat(cti, Qtt)
        t_proj = Pcti @ t_rej
        Tq_test = np.einsum("mi,ij,jm->m", t_proj.T, Qtt_inv, t_proj)

        T_alpha_list[i, :] = (
            Tq_test + 2 * np.log(pi_alpha[i]) + 2 * np.log(rho_alpha[i])
        )
        # it is 1-CDF of alpha in fact! So we should take argmin later!

    Pijs = (
        np.argmax(T_alpha_list, axis=0) + 1
    )  # P0=0, P1=1, P2=2 etc.., matlab indexing starts at 1 as well.

    identifications_all[P_outside_ind] = Pijs

    for i, c_vec in enumerate(c_list):
        partition, _ = P_region[i]

        wdir = r"C:\Users\bgvannoort\Documents\IDS\Sim Data\SPP_GNSS\ordinary_DIA\penalized_setup1"

        xx, yy, zz = write_data_for_matlab(
            x,
            y,
            z,
            identifications_all,
            partition,
            inPlane=inPlane,
            DIA="ordinary DIA",
            write_grid_to_file=write_grid_to_file,
            factor=factor,
            type_of_example=type_of_example,
            indices_partitions=indices_partitions,
            manualPath=True,
            workingDir=wdir,
        )
        # xx, yy, zz = write_data_for_matlab(x, y, z, identifications_all, partition, inPlane=inPlane,
        #                                    DIA='ordinary DIA',
        #                                    write_grid_to_file=write_grid_to_file,
        #                                    factor=factor, type_of_example=type_of_example,
        #                                    indices_partitions=indices_partitions)

    if type_of_example == "simple":
        txtFilename = os.path.join(
            os.getcwd(),
            "..",
            "Sim Data",
            "ordinary_DIA",
            "penalized",
            "corresponding_identifications_factor_{}.txt".format(factor),
        )
        # txtFilename = os.path.join(r'C:\Users\bgvannoort\Documents\PhD\Python\IDS_IDS',
        #                            'Sim Data', 'ordinary_DIA', 'corresponding_identifications_factor_{}.txt'.format(factor))

    else:
        txtFilename = os.path.join(
            os.getcwd(),
            "..",
            "Sim Data",
            type_of_example,
            "ordinary_DIA",
            "corresponding_identifications_factor_{}.txt".format(factor),
        )
    np.savetxt(txtFilename, identifications_all)

    return identifications_all, T_alpha_list


if __name__ == "__main__":
    ## ------------------- initializations for storing and plotting-------------
    count_identifications_DIA = []
    colors_partitions = {
        "P0": "black",
        "P1": "yellow",
        "P2": "red",
        "P3": "blue",
        "P4": "grey",
        "P12": "orange",
        "P13": "green",
        "P14": "gold",
        "P21": "darkorange",
        "P23": "purple",
        "P24": "lightsalmon",
        "P31": "lime",
        "P32": "indigo",
        "P34": "cornflowerblue",
        "P41": "darkgoldenrod",
        "P42": "tomato",
        "P43": "lightsteelblue",
    }

    ## -------------- Problem setup ---------------------------------

    m, n, r, A, alpha, sigma, Qyy, Qyy_inv, Bt, Qtt, Qtt_inv = setup_SPP_GNSS_example(
        alpha_0=0.01, alpha_method="Kok_IDS"
    )
    ## ---------------- Grid generation ----------------
    tmin, tmax = -10, 10

    # n_samples_x, n_samples_y = 500, 500
    # x,y,z,t_3D = generate_t_grid(n_samples_x, n_samples_y)

    # t_3D = factor* t_3D
    type_of_example = "ARAIM_UNDEC_GNSS"

    factors = np.arange(2, 10.2, 0.2)
    factors = np.hstack((factors, np.arange(10, 36, 1)))
    factors = [
        100,
        200,
        500,
    ]  # for large factors, i.e. large t, instability of the functions / cdfs plays massive role
    # If factor is too large, the algorithm will take the first inf value in the
    # list of w or transformed w values, i.e. hypt 12.
    # factors = [2.8, 3, 3.2, 3.4, 4, 5, 6, 7, 8, 9, 10, 15, 21]
    # factors=[3.3]
    factors = [3, 6, 11, 15, 25]

    ## create the rho_alpha and pi_alpha arrays
    k = int(m) + int(m * (m - 1) / 2)
    rho_alpha_list = np.ones(k) * 0.5
    pi_alpha_list = np.ones(k) * 0.001

    P_H_q1 = 0.001
    P_H_q2 = P_H_q1**2
    rho_H_q1 = 0.5
    rho_H_q2 = 1
    index = 0
    while index < k:
        if index < m:  # q_i = 1
            rho_alpha_list[index] = rho_H_q1
            pi_alpha_list[index] = P_H_q1
        else:
            rho_alpha_list[index] = rho_H_q2
            pi_alpha_list[index] = P_H_q2
        index += 1

    # %%
    for factor in factors:
        identifications_ordinaryDIA, S_array = ordinary_DIA(
            factor,
            plot_figures=False,
            write_grid_to_file=True,
            type_of_example=type_of_example,
        )

        # identifications_ordinaryDIA, S_array = ordinary_DIA_penalized(factor,
        #                                                     rho_alpha=rho_alpha_list,
        #                                                     pi_alpha=pi_alpha_list,
        #                                                     plot_figures=False,
        #                                                     write_grid_to_file=True,
        #                                                     type_of_example=type_of_example)

        # identifications_ordinaryDIA_SPP = ordinary_DIA(factor, plot_figures=True,
        #                                             write_grid_to_file=True, type_of_example='SPP_GNSS')
