"""
This python program plots the partitioning of an iterative datasnooping procedure
and compares with a direct DIA method with formulating all hypotheses with
q=1 and q=2 at the start and performing DIA
Difference with IDS_partitioning_example_3D.py is that this plots on a 2D sphere in 3D
Author: Bob van Noort
Date: 2 June
"""

import numpy as np
import sys
import scipy
import scipy.stats
import matplotlib.pyplot as plt
import pickle
import scipy.spatial
import os
from Functions import *

from scipy.spatial.transform import Rotation
# from old_IDS_2its_functions import IDS_typeA_2its, IDS_typeB_2its, IDS_typeC_2its


def plot_plane_from_vectors(v1, v2, ax):
    # Generate a meshgrid for the parametric plane
    u, v = np.meshgrid(np.linspace(-1.5, 1.5, 10), np.linspace(-1.5, 1.5, 10))

    # Parametric equation of the plane
    X = u * v1[0] + v * v2[0]
    Y = u * v1[1] + v * v2[1]
    Z = u * v1[2] + v * v2[2]

    # Plot the surface/plane
    ax.plot_surface(X, Y, Z, color="grey", alpha=0.5)


## ------------------- initializations for storing and plotting-------------

# colors_partitions = {'P0': 'black', 'P1': 'yellow', 'P2': 'red', 'P3':'blue', 'P4':'grey',
#             'P12': 'orange', 'P13': 'green', 'P14': 'gold',
#             'P21': 'darkorange', 'P23': 'purple', 'P24': 'lightsalmon',
#             'P31': 'lime', 'P32': 'indigo', 'P34': 'cornflowerblue',
#             'P41': 'darkgoldenrod', 'P42': 'tomato', 'P43': 'lightsteelblue'}

## Make here the partitions P12, and P21 the same colors
# colors_partitions = {'P0': 'black', 'P1': 'yellow', 'P2': 'red', 'P3':'blue', 'P4':'grey',
#             'P12': 'orange', 'P13': 'green', 'P14': 'gold',
#             'P21': 'orange', 'P23': 'purple', 'P24': 'lightsalmon',
#             'P31': 'green', 'P32': 'purple', 'P34': 'cornflowerblue',
#             'P41': 'gold', 'P42': 'lightsalmon', 'P43': 'cornflowerblue', 'P99':'black'}


if __name__ == "__main__":
    np.random.seed(20)

    # type_of_example = 'Safoora_GNSS'
    type_of_example = "SPP_GNSS"
    # type_of_example='simple'
    # type_of_example='ARAIM_UNDEC_GNSS'
    # plt.close('all')
    makePlots = False
    inPlane = False
    print("Warning, inPlane is {}".format(inPlane))
    separate_order_IDS = False
    lastOMT = False
    tmin, tmax = -10, 10
    workingDir = "C:\\Users\\bgvannoort\\Documents\\IDS"

    type_of_DS = "C"
    type_of_alpha = "Kok_IDS"
    alpha_0 = 0.01

    m, n, r, A, alpha, sigma, Qyy, Qyy_inv, Bt, Qtt, Qtt_inv = load_setup_parameters(
        type_of_example, alpha_method=type_of_alpha, alpha_0=alpha_0
    )

    if type_of_example == "simple":
        cti = np.loadtxt(
            r"C:\Users\bgvannoort\Documents\IDS\Sim Data\fault_vectors.txt"
        )
        if separate_order_IDS:
            indices_partitions = {
                "P0": 0,
                "P1": 1,
                "P2": 2,
                "P3": 3,
                "P4": 4,
                "P12": 5,
                "P13": 6,
                "P14": 7,
                "P21": 8,
                "P23": 9,
                "P24": 10,
                "P31": 11,
                "P32": 12,
                "P34": 13,
                "P41": 14,
                "P42": 15,
                "P43": 16,
                "P99": -1,
            }
        else:
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
    elif type_of_example == "SPP_GNSS" or type_of_example == "ARAIM_UNDEC_GNSS":
        cti = np.loadtxt(
            rf"C:\Users\bgvannoort\Documents\IDS\Sim Data\{type_of_example}\fault_vectors.txt"
        )
        np.savetxt(
            rf"C:\Users\bgvannoort\Documents\IDS\Sim Data\{type_of_example}\Qyy_diag.txt",
            np.diag(Qyy),
        )
        indices_partitions = load_indices_partitions_GNSS_ex(separate_order_IDS)
    elif type_of_example == "Safoora_GNSS":
        pass
        ## TO IMPLEMENT:
        # indices_partitions = get_idx_hypt_RIDS(m, subset=[])
    else:
        raise NotImplementedError(
            "type of example not recognized/implemented: {}".format(type_of_example)
        )

    a1, a2 = cti[:, 0].reshape(-1, 1), cti[:, 1].reshape(-1, 1)
    a3, a4 = cti[:, 2].reshape(-1, 1), cti[:, 3].reshape(-1, 1)

    # type_of_alpha='manual'

    # alpha_0 = 0.001
    # alpha_0='default'

    # for factor in np.arange(1, 52, 10):

    for factor in [
        4,
        7,
        11,
    ]:  # if inPlane=True, they are angles over which to rotate the normal vector.
        if inPlane:  ## we slightly try to rotate the normal vector.
            a1, a2 = cti[:, 0].reshape(-1, 1), cti[:, 1].reshape(-1, 1)
            a3, a4 = cti[:, 2].reshape(-1, 1), cti[:, 3].reshape(-1, 1)
            fraction = factor / 90
            a1 = a1 * (1 - fraction) + a4 * fraction
            a2 = a2 * (1 - fraction) + a3 * fraction

            a1 = a1 / np.linalg.norm(a1)
            a2 = a2 / np.linalg.norm(a2)

            a1 = cti[:, 0].reshape(-1, 1)
            a2 = cti[:, 2].reshape(
                -1, 1
            )  # just take ct1 and ct3 as the spans of the plane.

            # n_vec_new = n_vec + zeta * np.sin(angle_of_rotation)
            # rotated_normal = n_vec_new / np.linalg.norm(n_vec_new)
            # # Find two new vectors in the plane orthogonal to the new normal
            # # Use Gram-Schmidt process to get two vectors orthogonal to the normal
            # v1_new = np.cross(rotated_normal, np.array([1, 0, 0]))  # Cross product with a random vector
            # v1_new = v1_new / np.linalg.norm(v1_new)  # Normalize the vector
            # v2_new = np.cross(rotated_normal, v1_new)  # Get second vector in the plane

            ## set the new vectors v1 and v2 as the new sampling vectors
            # a1, a2 = v1_new.reshape(-1,1), v2_new.reshape(-1,1)

        # Make data
        ## ---------------- Grid generation ----------------
        n_samples_x, n_samples_y = 1000, 1000
        if inPlane:
            x, y, z, t_3D = generate_t_grid_in_plane(
                a1, a2, n_samples_x, n_samples_y, bigger_range=False
            )
            t_new = t_3D
            dirNewGrid = r"C:\Users\bgvannoort\Documents\IDS\Sim Data\SPP_GNSS\IDS\C\separate_partitionings\inPlane\alpha_type_Kok_IDS\Rotate_Along_c1c2_vectors"
            np.savetxt(os.path.join(dirNewGrid, "grid_x.txt"), x, delimiter=",")
            np.savetxt(os.path.join(dirNewGrid, "grid_y.txt"), y, delimiter=",")
            np.savetxt(os.path.join(dirNewGrid, "grid_z.txt"), z, delimiter=",")
        else:
            x, y, z, t_3D = generate_t_grid(n_samples_x, n_samples_y)
            t_new = t_3D * factor

        # -------------- if we want to make projection in the P1/P2 fault line plane (can be any 2 fault lines)
        # if inPlane:
        #     a1, a2 = cti[:, 0].reshape(-1,1), cti[:,1].reshape(-1,1)
        #     normal = np.cross(a1.flatten(), a2.flatten())
        #     d_plane = -np.dot(a1.flatten(), normal)
        #     if np.abs(normal[2]) < 1e-8: ## the normal component of z is 0, i.e. plane lies in whole z-dimension
        #         # pass # we need to find solution for this, as the normal component of z is zero
        #         # xx, z_val = np.meshgrid(np.arange(tmin, tmax, 1), np.arange(tmin, tmax, 1))
        #         xx, yy = np.meshgrid(np.arange(tmin, tmax, 1), np.arange(tmin, tmax, 1))
        #         z_val = (-normal[0] * xx - normal[1] * yy - d_plane) * 1. /normal[2]
        #     else:
        #         xx, yy = np.meshgrid(np.arange(tmin, tmax, 1), np.arange(tmin, tmax, 1))
        #         z_val = (-normal[0] * xx - normal[1] * yy - d_plane) * 1. /normal[2]
        #     # plt3d.plot_surface(xx, yy, z_val, alpha=0.2, label='fault plane for '+partition, color='red')
        #     ax.plot_surface(xx, yy, z_val, alpha=0.4, label='fault plane for '+partition, color='red')

        # t_new = np.random.rand(r, int(1e6))*10 - 5
        # t_new = np.random.multivariate_normal(np.zeros(r), Qtt, size=int(1e6)).T

        Pis = compute_Pis(
            t_new,
            Qtt,
            B_T=Bt,
            alpha_prime=alpha,
            type_of_testing="IDS",
            type_of_DS=type_of_DS,
            alpha_method=type_of_alpha,
            alpha_0=alpha_0,
            S=[],
            idx_S=0,
            qmax=2,
            type_of_example=type_of_example,
            separate_order_IDS=separate_order_IDS,
        )
        # sys.exit()
        ## --------------------------------

        # ## Carry out hypothesis testing on the t-grid
        # if type_of_DS == 'A':
        #     t, identifications, identifications_list = IDS_typeA_2its(t_new, separate_order_IDS=separate_order_IDS,
        #                                                         inPlane=inPlane, lastOMT=lastOMT,
        #                                                         type_of_DS=type_of_DS, type_of_alpha=type_of_alpha,
        #                                                         type_of_example=type_of_example)
        # elif type_of_DS == 'B':
        #     t, identifications, identifications_list = IDS_typeB_2its(t_new, separate_order_IDS=separate_order_IDS,
        #                                                     inPlane=inPlane, lastOMT=lastOMT,
        #                                                     type_of_DS=type_of_DS, type_of_alpha=type_of_alpha,
        #                                                     type_of_example=type_of_example)
        # elif type_of_DS == 'C':
        #     t, identifications, identifications_list = IDS_typeC_2its(t_new, separate_order_IDS=separate_order_IDS,
        #                                                     inPlane=inPlane, lastOMT=lastOMT,
        #                                                     type_of_DS=type_of_DS, type_of_alpha=type_of_alpha,
        #                                                     type_of_example=type_of_example)
        # Pis = identifications_list

        # ## if inPlane, extract the new plane meshgrid from t
        # if inPlane:
        #     xx, yy, zz = t[0,:], t[1,:], t[2,:]
        #     x = xx.reshape((500, 500))
        #     y = yy.reshape((500, 500))
        #     z = zz.reshape((500, 500))

        total_points = 0

        ## plot figures.
        # count_identifications_IDS = plot_IDS_figuresv2(identifications)
        # plt.close('all') #close figures

        # ## Not sure if the saving of "corresponding_identifications_factor" goes into correct directory
        # if lastOMT:
        #     #Store the identifications list
        #     np.savetxt(workingDir + '\\Sim Data\\IDS\\lastOMT\\corresponding_identifications_factor_{}.txt'.format(factor), identifications_list)
        # else:
        #     #Store the identifications list
        #     np.savetxt(workingDir + '\\Sim Data\\IDS\\corresponding_identifications_factor_{}.txt'.format(factor), identifications_list)
        ## Plot the 2D sphere in 3D
        if makePlots:
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "3d"})

            # if inPlane:
            #     # Plot the original plane spanned by a1 and a2
            #     plot_plane_from_vectors(a1, a2, ax)

        for idx_, partition in enumerate(indices_partitions.keys()):
            if partition == "P99":
                pass  # simply go on
            elif len(partition) == 2:
                pass  # also continue

            elif not separate_order_IDS:
                if int(partition[1]) > int(partition[2]):
                    continue  # skip these.

            ## If idx_ = 0, remove all files in the directory to save.
            if idx_ == 0:
                removeFiles = True
            else:
                removeFiles = False

            xx, yy, zz = write_data_for_matlab(
                x,
                y,
                z,
                Pis,
                partition,
                write_grid_to_file=True,
                separate_order_IDS=separate_order_IDS,
                inPlane=inPlane,
                factor=factor,
                workingDir=workingDir,
                lastOMT=lastOMT,
                type_of_DS=type_of_DS,
                type_of_alpha=type_of_alpha,
                a1=a1,
                a2=a2,
                removeFiles=removeFiles,
                manualPath=False,
                angle=factor,
                type_of_example=type_of_example,
                indices_partitions=indices_partitions,
            )
            if makePlots:
                ax.plot_surface(
                    xx,
                    yy,
                    zz,
                    color=colors_partitions[partition],
                    alpha=1.0,
                    edgecolor=None,
                    facecolor=colors_partitions[partition],
                    # facecolors = cm.jet(part_norm),
                    rstride=1,
                    cstride=1,
                    linewidth=0,
                    antialiased=False,
                )

            # # if partition == 'P2':
            # #     sys.exit()
            # total_points += tvals.shape[1]

        if makePlots:
            plot_3Dline(ax, a1, color="black")
            plot_3Dline(ax, a2, color="black")
            ax.set_xlim([tmin, tmax])
            ax.set_ylim([tmin, tmax])
            ax.set_zlim([tmin, tmax])
            ax.set_aspect("equal")
            ax.set_xlabel("$t_1$")
            ax.set_ylabel("$t_2$")
            ax.set_zlabel("$t_3$")
            ax.set_title(
                "Partitions for 3D misclosure space, IDS\nRadius of sphere = "
                + str(factor)
            )
            ax.view_init(40, 40)
            ax.legend(bbox_to_anchor=(1.1, 1.1))
