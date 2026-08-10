"""
Created on Sun Jun  2 17:44:55 2024
Author: BG van Noort

DIA partitioning of DS method, no iterations.
Partitioning is plot on a sphere in 3D (i.e. projected on a 2D spherical surface.)
"""

import numpy as np
import scipy
import scipy.stats
from Functions import *
import os


if __name__ == "__main__":
    type_of_example = "SPP_GNSS"
    type_of_example = "simple"
    workingDir = os.path.join(r"C:\Users\bgvannoort\Documents\IDS")

    ## ------------------------ Problem Setup --------------------------
    if type_of_example == "simple":
        m, n, r, A, alpha, sigma, Qyy, Qyy_inv, Bt, Qtt, Qtt_inv = setup()
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
            "P99": 99,
        }
    elif type_of_example == "SPP_GNSS" or type_of_example == "ARAIM_UNDEC_GNSS":
        m, n, r, A, alpha, sigma, Qyy, Qyy_inv, Bt, Qtt, Qtt_inv = (
            setup_SPP_GNSS_example()
        )
        indices_partitions = load_indices_partitions_GNSS_ex(
            False
        )  # No order for simple DS

    alpha_0 = 0.01
    alpha = modify_alpha_prime(None, alpha_0, m, method="Kok_IDS", df_chi2_new=m - n)
    cti = Bt @ np.eye(m)
    type_of_DS = "D"
    print('alpha"=', alpha)

    ## ---------------- Generate t-grid on a sphere of radius 1 --------
    n_samples_x, n_samples_y = 1000, 1000
    x, y, z, t = generate_t_grid(n_samples_x, n_samples_y)
    # np.savetxt('H:\\My Documents\\PhD\\Python\\Case Study\\IDS\\Sim Data\\lower_res\\grid_x.txt', x)
    # np.savetxt('H:\\My Documents\\PhD\\Python\\Case Study\\IDS\\Sim Data\\lower_res\\grid_y.txt', y)
    # np.savetxt('H:\\My Documents\\PhD\\Python\\Case Study\\IDS\\Sim Data\\lower_res\\grid_z.txt', z)

    for factor in [3.3, 6]:  # np.arange(1,30,10):
        t_fac = t * factor
        # make the fault vectors
        ci_list = []
        P_region = []
        for i in range(m):
            ci = np.zeros((m, 1))
            ci[i] = 1
            ci_list.append(ci)
            P_region.append(["P" + str(i + 1), ci])

        ## carry out hypothesis testing
        identifications = ordinary_DS(
            t_fac,
            Qtt_inv,
            Bt,
            alpha,
            P_region,
            type_of_DS=type_of_DS,
            alpha_0=alpha_0,
            type_of_example=type_of_example,
        )
        # Add the regions P0 and P99 later to the Pregion list, because otherwise the algorithm will try to compute the
        # correspond w-test values for these partitions (which does not exist / are not defined)
        P_region.append(["P0", None])
        P_region.append(["P99", None])
        for partition, _ in P_region:
            write_data_for_matlab(
                x,
                y,
                z,
                identifications,
                partition,
                DIA="DS_DIA",
                write_grid_to_file=True,
                factor=factor,
                workingDir=workingDir,
                type_of_DS=type_of_DS,
                type_of_example=type_of_example,
                indices_partitions=indices_partitions,
            )
        # np.savetxt(workingDir + '\\Sim Data\\DS_DIA\\corresponding_identifications_factor_{}_type_of_DS_{}.txt'.format(factor, type_of_DS), identifications)
