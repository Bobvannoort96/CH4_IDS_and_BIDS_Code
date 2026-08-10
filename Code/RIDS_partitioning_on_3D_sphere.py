"""
Description of code:
    Python script to call any type of RIDS function from the
    Functions.py script.
Author:
    Bob van Noort
Date:

"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import itertools
from Functions import *
import time

if __name__ == "__main__":
    t0 = time.process_time()
    type_of_DS = "C"
    type_of_testing = "R_IDS"
    bool_start_with_OMT = False

    ## Continue the implementation here. See e.g. implemented_and_old_routines/RIDS_typeA...
    n_samples = 1000
    x, y, z, t_det = generate_t_grid(n_samples, n_samples)
    lastOMT = True
    type_of_example = "SPP_GNSS"
    # type_of_example = 'simple'
    # type_of_example = 'ARAIM_UNDEC_GNSS'
    # type_of_example = 'Safoora_GNSS'
    # t_det = np.random.rand(15, 100000)*10-10

    alpha_method = "Kok_IDS"
    S_0 = []
    alpha_0 = 0.01
    qmax = 2

    for factor in [10, 25, 50]:
        ci_list = None
        idx_S = None
        if type_of_DS == "A":
            Pis = RIDS_mult_its_type_A(
                t_det * factor,
                alpha_0=alpha_0,
                qmax=qmax,
                S=S_0,
                ci_list=ci_list,
                idx_S=idx_S,
                example_type=type_of_example,
                bool_start_with_OMT=bool_start_with_OMT,
            )
        elif type_of_DS == "B":
            Pis = RIDS_mult_its_type_B(
                t_det * factor,
                alpha_0=alpha_0,
                qmax=qmax,
                S=S_0,
                ci_list=ci_list,
                idx_S=idx_S,
                example_type=type_of_example,
            )

        elif type_of_DS == "C":
            Pis = RIDS_mult_its_type_C(
                t_det * factor,
                alpha_0=alpha_0,
                qmax=qmax,
                S=S_0,
                ci_list=ci_list,
                idx_S=idx_S,
                example_type=type_of_example,
            )

        tend = time.process_time()
        print("Simulation took {} seconds".format(np.round(tend - t0, 3)))

        if type_of_example == "simple":
            m, n, r, A, alpha, sigma, Qyy, Qyy_inv, Bt, Qtt, Qtt_inv = setup(
                alpha_0=alpha_0, alpha_method=alpha_method
            )
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
            if lastOMT:
                manualPath = r"C:\Users\bgvannoort\Documents\IDS\Sim Data\R_IDS\{}\no_separate_partitionings\lastOMT\alpha_type_Kok_IDS".format(
                    type_of_DS
                )
                # manualPath = r'C:\Users\bgvannoort\Documents\IDS\Sim Data\R_IDS\A\qmax=1\lastOMT\alpha_type_Kok_IDS'
            else:
                manualPath = r"C:\Users\bgvannoort\Documents\IDS\Sim Data\R_IDS\{}\no_separate_partitionings\alpha_type_Kok_IDS".format(
                    type_of_DS
                )

        elif type_of_example == "SPP_GNSS" or type_of_example == "ARAIM_UNDEC_GNSS":
            m, n, r, A, alpha, sigma, Qyy, Qyy_inv, Bt, Qtt, Qtt_inv = (
                setup_SPP_GNSS_example(alpha_0=alpha_0, alpha_method=alpha_method)
            )
            indices_partitions = load_indices_partitions_GNSS_ex(
                False
            )  # No order for simple DS
            if lastOMT:
                manualPath = r"C:\Users\bgvannoort\Documents\IDS\Sim Data\{}\R_IDS\{}\no_separate_partitionings\lastOMT\alpha_type_Kok_IDS".format(
                    type_of_example, type_of_DS
                )

            else:
                manualPath = r"C:\Users\bgvannoort\Documents\IDS\Sim Data\{}\R_IDS\{}\no_separate_partitionings\alpha_type_Kok_IDS".format(
                    type_of_example, type_of_DS
                )

        ci_list = []
        partition_list = ["P0"]
        for i in range(m):
            ci = np.eye(m)[:, i].reshape(-1, 1)
            ci_list.append(ci)
            partition_list.append("P" + str(i + 1))

        for i in range(m):
            for j in range(i + 1, m):
                cij = np.eye(m)[:, np.array([i, j])]
                ci_list.append(cij)
                partition_list.append("P" + str(i + 1) + str(j + 1))

        partition_list.append("P99")

        for i, part in enumerate(partition_list):
            if i == 0:
                removeFiles = True
            else:
                removeFiles = False

            write_data_for_matlab_RIDS(
                x,
                y,
                z,
                Pis,
                part,
                removeFiles=removeFiles,
                manualPath=manualPath,
                factor=factor,
                type_of_example=type_of_example,
                indices_partitions=indices_partitions,
            )
