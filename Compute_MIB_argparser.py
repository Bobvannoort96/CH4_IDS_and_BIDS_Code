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
from compute_MIB import *
import argparse


# if type_of_example == 'simple':
#     # dictionaries of indices. See the description of identifications_list in function IDS_2its()
#     indices_partitions = {'P0': 0, 'P1': 1, 'P2': 2, 'P3':3, 'P4':4,
#                 'P12': 5, 'P13': 6, 'P14': 7,
#                 'P21': 5, 'P23': 8, 'P24': 9,
#                 'P31': 6, 'P32': 8, 'P34': 10,
#                 'P41': 7, 'P42': 9, 'P43': 10, 'P99':99}
#     m, n, r, A, alpha, sigma, Qyy, Qyy_inv, B_T, Qtt, Qtt_inv = setup(alpha_0=alpha_0,
#                                                                       alpha_method=type_of_alpha,
#                                                                       alpha_prime=alpha_prime)

# elif type_of_example == 'SPP_GNSS':
#     indices_partitions = load_indices_partitions_GNSS_ex(False) # IDS_sep_order = False
#     m, n, r, A, alpha, sigma, Qyy, Qyy_inv, B_T, Qtt, Qtt_inv = setup_SPP_GNSS_example(alpha_0=alpha_0,
#                                                                                         alpha_method=type_of_alpha,
#                                                                                         alpha_prime=alpha_prime)

# q_i = len(for_partition.replace('P', ''))
# idx_under_hypt = indices_partitions[for_partition]

# n_points = 1000 # number of points on a 2D or 3D sphere for the unit vector d.

# if q_i == 1: # the MIB is 1D, simply a value
#     d_vectors = np.array([[1]])

#     # Gather the indices from 'for_partition'
#     idxes = int(for_partition.replace('P',''))

#     ci_model = np.eye(m)[:, idxes-1].reshape(-1,1)


#     pass
# elif q_i == 2: # the MIB is is a 2D figure, i.e. ellipse or circle
#     # Number of vectors to generate (resolution)


#     # Angles from 0 to 2π
#     theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

#     # Create unit vectors
#     d_vectors = np.stack((np.cos(theta), np.sin(theta)), axis=0)  # shape: (2, n_points)

#     idxes = for_partition.replace('P', '')
#     idxes = np.array( [int(zeta) for zeta in idxes] )
#     ci_model = np.eye(m)[:, idxes - 1 ]


# elif q_i == 3: # the MIB is a 3D figure, i.e. an ellipsoid or spheroid.


#     # Sample uniformly on the sphere
#     phi = np.random.uniform(0, 2 * np.pi, n_points)        # azimuthal angle
#     cos_theta = np.random.uniform(-1, 1, n_points)          # cos(theta) uniformly from -1 to 1
#     theta = np.arccos(cos_theta)                            # corresponding theta

#     # Convert spherical to Cartesian coordinates
#     x = np.sin(theta) * np.cos(phi)
#     y = np.sin(theta) * np.sin(phi)
#     z = np.cos(theta)

#     # Stack into 3xN array
#     d_vectors = np.vstack((x, y, z))  # shape (3, 1000)

#     idxes = for_partition.replace('P', '')
#     idxes = np.array( [int(zeta) for zeta in idxes] )
#     ci_model = np.eye(m)[:, idxes - 1 ]


# PCI_goal = 0.8
# cti_model = B_T @ ci_model
# N_samples = int(1e6)
# PCI_computed = 0.1 * PCI_goal
# max_its = 500
# store_MIB = np.zeros(d_vectors.shape)
# store_PCI_computed = np.zeros(d_vectors.shape[1])

# for d_i in range(d_vectors.shape[1]):

#     print('at d_i = ', d_i)
#     d_vector = d_vectors[:, d_i].reshape(-1, 1)
#     b_mag = 300.0  # Start from zero bias
#     d_b_mag = 0.1
#     it_counter = 0
#     PCI_computed = 1.1 * PCI_goal  # Reset for every d_vector


#     ## MIB Is very sensitive and a large increase in bias size, only slightly
#     ## increases the PCI.
#     # Find a more optimal MIB
#     MIB_range = np.arange(0, 20)*50
#     PCI_initialized = np.zeros(len(MIB_range))
#     for i_mib, Mib in enumerate(MIB_range):
#         # Simulate test statistics
#         t_mean = Mib * cti_model @ d_vector
#         t_samples = scipy.stats.multivariate_normal.rvs(mean=t_mean.flatten(), cov=Qtt, size=(N_samples,)).T

#         # Compute PCI
#         Pis = compute_Pis(
#             t_samples,
#             Qtt,
#             B_T,
#             alpha_prime,
#             type_of_testing,
#             type_of_DS=type_of_DS,
#             type_of_example=type_of_example
#         )
#         PCI_computed_init = np.sum(Pis == idx_under_hypt) / N_samples
#         PCI_initialized[i_mib] = PCI_computed_init

#     where_larger_than_PCI_goal = PCI_initialized > PCI_goal
#     if np.sum(where_larger_than_PCI_goal) == 0:
#         # the MIB is even larger than 500.
#         store_MIB[:, d_i] = np.nan
#         store_PCI_computed[d_i]=np.nan
#         continue
#     else:
#         b_mag = MIB_range[where_larger_than_PCI_goal][0]

#     print('b_mag', b_mag)
#     while PCI_computed > PCI_goal:
#         # Update bias
#         b_mag -= d_b_mag

#         # Simulate test statistics
#         t_mean = b_mag * cti_model @ d_vector
#         t_samples = scipy.stats.multivariate_normal.rvs(mean=t_mean.flatten(), cov=Qtt, size=(N_samples,)).T

#         # Compute PCI
#         Pis = compute_Pis(
#             t_samples,
#             Qtt,
#             B_T,
#             alpha_prime,
#             type_of_testing,
#             type_of_DS=type_of_DS,
#             type_of_example=type_of_example
#         )
#         PCI_computed = np.sum(Pis == idx_under_hypt) / N_samples
#         print(f'd_i={d_i} it={it_counter} PCI_computed={PCI_computed:.4f} b_mag={b_mag:.2f}')


#         it_counter+=1
#     ## Store the computed MIB
#     # But check if converged:
#     if it_counter == max_its:
#         store_MIB[:, d_i] = np.nan
#         store_PCI_computed[d_i]=np.nan
#     else:
#         store_MIB[:, d_i] = b_mag * d_vector.flatten()
#         store_PCI_computed[d_i]=PCI_computed


# %%


# if q_i == 2:
#     fig, ax = plt.subplots()
#     ax.plot(store_MIB[0,:], store_MIB[1,:], label='MIB surface')
#     ax.set_xlabel(r'$b_1$')
#     ax.set_ylabel(r'$b_2$')

# ax.set_title('MIB for H{}'.format(for_partition.replace('P', '')))
# ax.legend()


if __name__ == "__main__":
    ## If we want to run 'classical DIA', we have to hard-code it.
    # type_of_example='SPP_GNSS'
    type_of_example = "simple"
    alpha_0 = 0.01
    alpha_prime = 0.01
    type_of_alpha = "Kok_IDS"
    qmax = 2  # corresponding to the 'simple' example

    for_partition = "P24"
    print("Starting")

    write_data = True
    # parser = argparse.ArgumentParser()
    # parser.add_argument("d_i", help='The index of the d-array that we want to compute the MIB for')
    # parser.add_argument("type_of_testing", help='the testing type implemented, i.e. IDS, DS or classical DIA')
    # parser.add_argument("type_of_DS", help='The type of data snooping that should be used')
    # args = parser.parse_args()

    # MIB_vec, PCI_comp = compute_MIB_for_d_i(int(args.d_i), args.type_of_testing, args.type_of_DS, type_of_example, alpha_0, alpha_prime, type_of_alpha, for_partition, qmax=qmax)
    type_of_testing = "classical DIA"
    # type_of_testing = 'DS'
    type_of_DS = "A"
    PCI_goal = 0.6
    hypt_range = [1]

    for type_of_DS in ["A"]:
        for PCI_goal in [0.7, 0.8]:
            # for hypt in range(4):
            # for hypt in hypt_range:
            # for_partition='P'+str(hypt+1)
            for d_i in [385]:
                MIB_vec, PCI_comp = compute_MIB_for_d_i(
                    d_i,
                    type_of_testing,
                    type_of_DS,
                    type_of_example,
                    alpha_0,
                    alpha_prime,
                    type_of_alpha,
                    for_partition,
                    qmax=qmax,
                    PCI_goal=PCI_goal,
                )
                print("At di = ", d_i)
                print("MIB: ", MIB_vec)
                print("PCI_computed = ", PCI_comp)
            sys.exit()
            # MIB_vec, PCI_comp = compute_MIB_for_d_i(d_i,  'classical DIA', 'B', type_of_example, alpha_0, alpha_prime, type_of_alpha, for_partition, qmax=qmax)

            alpha_string = "type_of_alpha_" + type_of_alpha + "alpha_0_" + str(alpha_0)

            # resDir = os.path.join('/home/bgvannoort/results/MIBs',
            #                       type_of_example,
            #                       for_partition.replace('P', 'H'),
            #                       args.type_of_testing,
            #                       args.type_of_DS)
            if write_data:
                alpha_string = "type_of_alpha_{}alpha_0_{}".format(
                    type_of_alpha, alpha_0
                )
                resDir = os.path.join(
                    r"C:\Users\bgvannoort\Documents\IDS\Results\MIBs",
                    type_of_example,
                    for_partition.replace("P", "H"),
                    "PCI_goal={}".format(PCI_goal),
                    type_of_testing,
                    type_of_DS,
                    alpha_string,
                )

                os.makedirs(resDir, exist_ok=True)

                # d_i = int(args.d_i)

                MIB_row = np.concatenate(([d_i], MIB_vec.flatten()))
                PCI_row = np.array([d_i, PCI_comp])

                # Save to text file
                with open(
                    os.path.join(resDir, "MIB_vectors_di_{}.txt".format(d_i)), "w"
                ) as f_mib:
                    np.savetxt(f_mib, MIB_row.reshape(1, -1), fmt="%.6e")

                with open(
                    os.path.join(resDir, "PCI_computed_di_{}.txt".format(d_i)), "w"
                ) as f_pci:
                    np.savetxt(f_pci, PCI_row.reshape(1, -1), fmt="%.6e")

    # %% test:
    m, n, r, A, alpha, sigma, Qyy, Qyy_inv, Bt, Qtt, Qtt_inv = load_setup_parameters(
        type_of_example, type_of_alpha, alpha_0=alpha_0
    )

    meant = Bt @ np.eye(m)[:, 0].reshape(-1, 1) * 9.1796875
    t_samples = np.random.multivariate_normal(
        mean=meant.flatten(), cov=Qtt, size=int(1e6)
    ).T

    Pis_IDS_A = compute_Pis(
        t_samples,
        Qtt,
        Bt,
        alpha_prime,
        type_of_testing="IDS",
        type_of_DS="A",
        type_of_example=type_of_example,
        qmax=qmax,
    )

    Pis_DS_A = compute_Pis(
        t_samples,
        Qtt,
        Bt,
        alpha_prime,
        type_of_testing="DS",
        type_of_DS="A",
        type_of_example=type_of_example,
        qmax=qmax,
    )
