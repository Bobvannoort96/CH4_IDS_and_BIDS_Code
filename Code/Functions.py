"""
This file contains all relevant functions for the scripts in the current directory
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
from functools import wraps
import json
import math
import itertools
from numba import njit


def plot_3Dline(ax, cti, color):
    cti_plot = np.hstack((cti * 10, cti * -10))
    ax.plot(cti_plot[0, :], cti_plot[1, :], cti_plot[2, :], color=color)
    return ax


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print("func:%r  took: %2.4f sec" % (f.__name__, te - ts))
        return result

    return wrap


def plot_faultline(ax, cti, color, partition):
    cti_plot = np.hstack((cti * 10, cti * -10))
    ax.plot(
        cti_plot[0, :],
        cti_plot[1, :],
        cti_plot[2, :],
        label="$c_{t_i}$ for " + partition,
        color=color,
    )
    return ax


def plusmat(A, Qyy, inverse=False):
    # If inverse=True, the Qyy matrix is already inverted. Otherwise, it has to be inverted.
    if inverse:
        return np.linalg.inv(A.T @ Qyy @ A) @ A.T @ Qyy
    else:
        return np.linalg.inv(A.T @ np.linalg.inv(Qyy) @ A) @ A.T @ np.linalg.inv(Qyy)


def P_mat(A, Qyy, inverse=False):
    return A @ plusmat(A, Qyy, inverse=inverse)


@njit
def plusmat_invQ(A, Qyy_inv):
    # Assumes Qyy_inv is already inverted
    AtQ = A.T @ Qyy_inv
    return np.linalg.inv(AtQ @ A) @ AtQ


@njit
def P_mat_invQ(A, Qyy_inv):
    return A @ plusmat_invQ(A, Qyy_inv)


@njit
def P_perp(P):
    n = P.shape[0]
    I = np.eye(n)
    return I - P


# def P_perp(P):
#     # P should be a diagonal projection matrix
#     return np.eye(P.shape[0]) - P

# @njit # this functions seems to not really speed things up..
# def compute_OMTs_RIDS(Bt, Qtt_inv, t_det, combinations, m, qmax):
#     n_samples = t_det.shape[1]
#     OMTs = np.zeros((len(combinations), n_samples))
#     Qtt_inv = np.ascontiguousarray(Qtt_inv)
#     for i in range(len(combinations)):
#         idxes = combinations[i]
#         ci = np.eye(m)[:, idxes]
#         cti = Bt @ ci
#         P_perp_cti = P_perp(P_mat_invQ(cti, Qtt_inv)) # np.eye(cti.shape[0]) - cti @ np.linalg.inv(cti.T @ Qtt_inv @ cti) @ cti.T @ Qtt_inv
#         proj_t = P_perp_cti @ t_det
#         for j in range(n_samples):
#             v = np.ascontiguousarray(proj_t[:, j])
#             OMTs[i, j] = v.T @ Qtt_inv @ v
#     return OMTs


def modify_alpha_prime(alpha, alpha_0, m, method, df_chi2_new, beta_0=0.2, **kwargs):
    """
    Function that modifies the value of alpha_prime during the several iterations.
    Different methods exist, such as the Bonferonni equation, KOK_IDS method, and the simple
    division by m at every iteration.

    Parameters
    ----------
    alpha : float
        # alpha is the initial alpha prime set, and
        # method is an argument, can have values of "iteration"
    alpha_0 : float
        alpha_0 the significance of the w-test
    m : int
        number of (left over) measurements
    method : str
        can have values:
            'iteration': then the alpha_prime is updated as alpha_prime = alpha_prime/m every iteration
            'Kok_IDS': the same method as in Kock IDS, by keeping a constant lambda_0, a keywordargument of beta_0
                should then be provided
            'Bonferonni': alpha_prime = m * alpha_0 (note that 'm' thus changes every iteration, as one observation is subtracted)
            'manual': alpha_prime is hardcoded to the value in this function. alpha_0 is still given as input and unaltered.
    df_chi2_new: int
        nr of degrees of freedom for the chi2 distribution, normally at the start equal to r=m-n
    **kwargs : e.g. beta_0


    Returns
    -------
    alpa_prime_new
        The new required level of significance alpha_prime to set for the next iteration.

    """
    if method == "iteration":
        alpha_prime = alpha / m
    elif method == "Kok_IDS":
        T_alpha0 = scipy.stats.chi2.isf(alpha_0, df=1)
        lambda_0 = np.linspace(0.01, 100, 10000)
        beta_0_calculated = scipy.stats.ncx2.cdf(T_alpha0, df=1, nc=lambda_0)
        (indBeta0,) = np.where(beta_0_calculated < beta_0)
        # print(beta_0_calculated)

        if len(indBeta0) == 0:
            raise Exception(
                "The lambda_0 parameter is not found, increase the maximum of its array"
            )
        indx = indBeta0[0]
        lambda_0 = lambda_0[indx]
        # print('lambda_0', lambda_0)
        new_threshold = scipy.stats.ncx2.ppf(
            beta_0_calculated[indx], df=df_chi2_new, nc=lambda_0
        )
        # print('newthreshold', new_threshold)
        alpha_prime = scipy.stats.chi2.sf(new_threshold, df=df_chi2_new)
        # print('alpha_prime', alpha_prime)

    elif method == "Bonferonni":
        alpha_prime = m * alpha_0
    elif method == "manual":
        alpha_prime = 0.05

    else:
        raise Exception(
            "Wrong 'method' is given as argument. Value of method variable {}".format(
                method
            )
        )
    return alpha_prime


def save_interactive_3D(fig, title, folder, full_path=False, on_sphere=False):
    # saves a figure in the default_dir (if full_path = False), in the folder dir
    # with name title as filename.
    # if on_sphere ==True, then the t-space is plotted on a (unit) sphere for visualization
    if full_path:
        full_dir = folder
        if on_sphere:
            full_dir = folder
        else:
            full_dir = folder + "\\" + "on_sphere\\"
    else:
        if on_sphere:
            full_dir = (
                "H:\\My Documents\\PhD\\Python\\Case Study\\IDS\\3D plots\\"
                + folder
                + "\\"
                + "on_sphere"
                + "\\"
            )
        else:
            full_dir = (
                "H:\\My Documents\\PhD\\Python\\Case Study\\IDS\\3D plots\\"
                + folder
                + "\\"
            )
    output = open(full_dir + title + ".fig.pickle", "wb")
    pickle.dump(fig, output)
    output.close()


# generates the 3D grid for the misclosure space.
def generate_t_grid(n_samples_x, n_samples_y):
    # polar coords, radius=1
    ua = np.linspace(0, 2 * np.pi, n_samples_x)
    va = np.linspace(0, np.pi, n_samples_y)

    # express in cartesian coords
    xa = np.outer(np.cos(ua), np.sin(va))
    ya = np.outer(np.sin(ua), np.sin(va))
    za = np.outer(np.ones(np.size(ua)), np.cos(va))
    t_3Da = np.vstack((xa.flatten(), ya.flatten(), za.flatten()))
    return xa, ya, za, t_3Da


# generates the 3D grid for the misclosure space but then on a projected plane
# spanned by two vectors (given as input, a1, a2)
def generate_t_grid_in_plane(a1, a2, n_samples_x, n_samples_y, bigger_range=False):
    # Reinitialize t if we want to plot the projection in the fault plane
    normal = np.cross(a1.flatten(), a2.flatten())
    d_plane = -np.dot(a1.flatten(), normal)
    if bigger_range:
        multiply = 10
    else:
        multiply = 1

    # the normal component of z is 0, i.e. plane lies in whole z-dimension
    if np.abs(normal[2]) < 1e-6:
        # pass # we need to find solution for this, as the normal component of z is zero
        xa, za = np.meshgrid(
            np.linspace(-10, 10, n_samples_x) * multiply,
            np.linspace(-10, 10, n_samples_y) * multiply,
        )

        ya = (-normal[0] * xa - normal[2] * za - d_plane) * 1.0 / normal[1]
    else:
        xa, ya = np.meshgrid(
            np.linspace(-10, 10, n_samples_x) * multiply,
            np.linspace(-10, 10, n_samples_y) * multiply,
        )
        za = (-normal[0] * xa - normal[1] * ya - d_plane) * 1.0 / normal[2]
    # plt3d.plot_surface(xx, yy, z_val, alpha=0.2, label='fault plane for '+partition, color='red')

    t_3Da = np.vstack((xa.flatten(), ya.flatten(), za.flatten()))
    return xa, ya, za, t_3Da


def load_setup_parameters(
    type_of_example, alpha_method, alpha_prime=0.01, alpha_0=0.01
):

    if type_of_example == "simple" or type_of_example == "Simple":
        m, n, r, A, alpha, sigma, Qyy, Qyy_inv, Bt, Qtt, Qtt_inv = setup(
            alpha_prime=alpha_prime, alpha_0=alpha_0, alpha_method=alpha_method
        )
    elif type_of_example == "SPP_GNSS":
        m, n, r, A, alpha, sigma, Qyy, Qyy_inv, Bt, Qtt, Qtt_inv = (
            setup_SPP_GNSS_example(
                alpha_0=alpha_0, alpha_prime=alpha_prime, alpha_method=alpha_method
            )
        )

    elif type_of_example == "Safoora_GNSS":
        m, n, r, A, alpha, sigma, Qyy, Qyy_inv, Bt, Qtt, Qtt_inv = (
            setup_GNSS_example_Safoora(
                alpha_0=alpha_0, alpha_prime=alpha_prime, alpha_method=alpha_method
            )
        )

    elif type_of_example == "Sebastian_GNSS":
        m, n, r, A, alpha, sigma, Qyy, Qyy_inv, Bt, Qtt, Qtt_inv = (
            setup_GNSS_example_Sebastian(
                alpha_0=alpha_0, alpha_prime=alpha_prime, alpha_method=alpha_method
            )
        )
    elif type_of_example == "ARAIM_UNDEC_GNSS":
        m, n, r, A, alpha, sigma, Qyy, Qyy_inv, Bt, Qtt, Qtt_inv = (
            setup_GNSS_example_ARAIM(
                alpha_0=alpha_0, alpha_prime=alpha_prime, alpha_method=alpha_method
            )
        )
    elif type_of_example == "RIDS_EXAMPLE_ALPHA_PRIME":
        m, n, r, A, alpha, sigma, Qyy, Qyy_inv, Bt, Qtt, Qtt_inv = (
            RIDS_example_setup_alpha_prime(
                alpha_0=alpha_0, alpha_prime=alpha_prime, alpha_method=alpha_method
            )
        )
    else:
        raise NotImplementedError(
            "Example type {} not recognized".format(type_of_example)
        )

    return m, n, r, A, alpha, sigma, Qyy, Qyy_inv, Bt, Qtt, Qtt_inv


def RIDS_example_setup_alpha_prime(
    alpha_0=0.01, alpha_prime=0.01, alpha_method="Kok_IDS"
):
    ## -------------------- Params -----------------

    # This is Example1:
    # m = 10
    # n = 3
    # r = m-n

    # sigma = 5

    ## ---------------- SETUP -----------------

    # alpha = modify_alpha_prime(alpha_prime, alpha_0, m, alpha_method, r)

    # A = np.random.rand(m,n)
    # Qyy = sigma**2 *np.diag(np.random.rand(m)**2)

    ## ------------------------- EXAMPLE 1 ----------------------------
    # A = np.array([[0.66448788, 0.51502983, 0.98917836],
    #         [0.5818944 , 0.99643435, 0.45238036],
    #         [0.63352664, 0.35837426, 0.77483963],
    #         [0.24301238, 0.73602736, 0.77271068],
    #         [0.92992534, 0.37882301, 0.85280772],
    #         [0.84263871, 0.79652092, 0.24354619],
    #         [0.85075808, 0.03867353, 0.60284204],
    #         [0.47326013, 0.34214622, 0.32028229],
    #         [0.84355186, 0.95325281, 0.42316533],
    #         [0.30001384, 0.95531726, 0.41439415]])*20

    # Qyy = np.diag(np.array([0.88814228, 0.1052362 , 0.276576  , 0.17607621, 0.17880375,
    #         0.34097542, 0.58663172, 0.45519934, 0.45942048, 0.37059009])*sigma**2)

    # -------------------------END EXAMPLE 1 ----------------------------

    # This is Example2:
    # ------------------------- EXAMPLE 2 ----------------------------

    # m = 10
    # n = 4
    # r = m-n

    # sigma = 5

    # ## ---------------- SETUP -----------------

    # alpha = modify_alpha_prime(alpha_prime, alpha_0, m, alpha_method, r)

    # A = np.array([[-1.4497627 ,  1.54421485,  1.68452936,  0.06263251],
    #    [ 1.41865314,  0.85449145, -0.77839685,  1.13978621],
    #    [-1.4317954 ,  5.14901026,  1.70741668,  0.8915957 ],
    #    [ 3.50273345,  1.47272678,  2.12818382,  3.78012353],
    #    [ 2.3596269 ,  1.3061396 ,  0.19338137,  0.22361989],
    #    [ 1.64838747,  1.50025336,  1.52502409,  1.45435589],
    #    [ 1.75598702,  2.37992202,  1.12801468,  2.70763009],
    #    [ 3.08712491,  2.98582916, -3.5356622 ,  1.90990457],
    #    [ 1.79069835,  0.64599947,  1.57269543, -0.51352852],
    #    [ 2.17839484,  2.11698363,  1.64278991, -1.31201104]])

    # Qyy =  np.diag(np.array([0.10542736, 7.83060205, 0.46125042, 0.26628967, 0.90305997,
    #    0.26617308, 0.48295989, 0.01148187, 0.89231722, 3.42611467]))

    # -------------------------END EXAMPLE 2 ----------------------------

    # ------------------------- Example 3 -------------------------------
    sigma = 5
    A = np.array(
        [
            [1.22904512, -3.54968514],
            [0.27355584, 3.12473409],
            [-0.02407907, 3.18461683],
            [-3.33091538, -2.90867776],
            [-1.93551813, 1.67792454],
            [-1.99147255, 0.44022716],
            [-2.47978365, -2.3722565],
        ]
    )
    m, n = A.shape
    r = m - n

    Qyy = np.diag(
        np.array(
            [
                14.31268823,
                2.2163244,
                0.85054479,
                1.18610731,
                13.86584003,
                1.19437266,
                2.59718674,
            ]
        )
        * sigma**2
    )

    alpha = modify_alpha_prime(alpha_prime, alpha_0, m, alpha_method, r)

    # ----------------------------END Example 3-------------------------

    # # ------------------------- Example 4 -------------------------------
    # ## This does not work because this function is called from within all functions almost, so we get a different A every time.
    # sigma=5

    # n = int(1+np.random.rand(1)*5)
    # r = int(np.random.rand(1)*10 + n)
    # m = n + r
    # A = np.random.normal(size=(m,n))

    # Qyy = np.diag(np.random.normal(loc=0.0, scale=2, size=m)**2)

    # alpha = modify_alpha_prime(alpha_prime, alpha_0, m, alpha_method, r)

    # # ----------------------------END Example 4-------------------------

    Bt = scipy.linalg.null_space(A.T).T
    Qtt = Bt @ Qyy @ Bt.T
    Qtt_inv = np.linalg.inv(Qtt)

    Qyy_inv = np.linalg.inv(Qyy)
    return m, n, r, A, alpha, sigma, Qyy, Qyy_inv, Bt, Qtt, Qtt_inv


# Define the problem
def setup(alpha_prime=0.01, sigma=1.0, alpha_0=0.01, alpha_method="Kok_IDS"):
    m = 4
    n = 1
    r = m - n
    A = np.ones(m).reshape(-1, 1)

    # np.random.seed(4)
    # A = np.random.rand(m).reshape(-1,1)

    alpha = modify_alpha_prime(None, alpha_0, m, alpha_method, r)

    sigma = 1.0
    Qyy = np.eye(m) * sigma**2
    Qyy_inv = np.linalg.inv(Qyy)

    Bt = scipy.linalg.null_space(A.T).T

    Qtt = Bt @ Qyy @ Bt.T
    Qtt_inv = np.linalg.inv(Qtt)
    return m, n, r, A, alpha, sigma, Qyy, Qyy_inv, Bt, Qtt, Qtt_inv


# Define an SPP GNSS example problem with a given sigma value


def setup_SPP_GNSS_example(alpha_0=None, alpha_prime=None, alpha_method=None):

    A = np.array(
        [
            [0.0225, 0.9951, -0.0966, 1],
            [0.6750, -0.6900, -0.2612, 1],
            [0.0723, -0.6601, -0.7477, 1],
            [-0.9398, 0.2553, -0.2269, 1],
            [-0.5907, -0.7539, -0.2877, 1],
            [-0.3236, -0.0354, -0.9455, 1],
            [-0.6748, 0.4356, -0.5957, 1],
        ]
    )

    Qyy = np.diag([3.5740, 1.1252, 0.5479, 1.3258, 1.0104, 0.5309, 0.5838])

    m, n = A.shape
    r = m - n
    Bt = np.loadtxt(
        r"C:\Users\bgvannoort\Documents\IDS\Sim Data\SPP_GNSS\B_transpose_matrix.txt",
        delimiter=",",
    )

    # # Carry out test to see if Bt_I @ A is really zero
    null_mat = Bt @ A
    if np.abs(null_mat.sum()) > 1e-13:
        raise Exception("Apparently, the Bt_I matrix @ A does not produce zeros")

    if alpha_0 == None:
        alpha_0 = 0.001
    if alpha_prime == None:
        alpha_prime = 0.001

    if alpha_method is not None:
        alpha_prime = modify_alpha_prime(
            None, alpha_0, m, method=alpha_method, df_chi2_new=r
        )

    Qyy_inv = np.linalg.inv(Qyy)
    Qtt = np.eye(r)
    Qtt_inv = np.eye(r)
    # the None value is 'sigma' as sigma is no longer
    return m, n, r, A, alpha_prime, None, Qyy, Qyy_inv, Bt, Qtt, Qtt_inv


def setup_GNSS_example_Sebastian(
    alpha_0=0.01, alpha_prime=0.01, alpha_method="Kok_IDS"
):
    dir_to_load = (
        r"C:\Users\bgvannoort\Documents\IDS\Code\GNSS_examples_paper\Setup 1 Sebastian"
    )

    A = np.loadtxt(os.path.join(dir_to_load, "A_mat.txt"), delimiter=",")
    B = np.loadtxt(os.path.join(dir_to_load, "B_mat.txt"), delimiter=",")
    Qyy = np.loadtxt(os.path.join(dir_to_load, "Qyy.txt"), delimiter=",")
    Qyy_inv = np.linalg.inv(Qyy)

    Qtt = B.T @ Qyy @ B
    m, n = A.shape
    r = m - n

    alpha = modify_alpha_prime(alpha_prime, alpha_0, m, alpha_method, r)

    if np.sum(Qtt) > 1.01 * r or np.sum(Qtt) < 0.99 * r:
        raise Exception(
            "It appears that Qtt is not the identity matrix using this B-matrix \n \
                        B = {}".format(B)
        )

    Qtt_inv = np.eye(r)
    return m, n, r, A, alpha, None, Qyy, Qyy_inv, B.T, Qtt, Qtt_inv


def setup_GNSS_example_ARAIM(alpha_0=0.01, alpha_prime=0.01, alpha_method="Kok_IDS"):

    A = np.array(
        [
            [0.0225, 0.9951, -0.0966, 1],
            [0.6750, -0.6900, -0.2612, 1],
            [0.0723, -0.6601, -0.7477, 1],
            [-0.9398, 0.2553, -0.2269, 1],
            [-0.5907, -0.7539, -0.2877, 1],
            [-0.3236, -0.0354, -0.9455, 1],
            [0.85232641089, -0.279702496, -0.441934614182, 1],
        ]
    )

    B = scipy.linalg.null_space(A.T)

    m, n = A.shape
    r = m - n
    sigma = 1.0
    Qyy = sigma**2 * np.eye(m)
    Qyy_inv = np.linalg.inv(Qyy)

    Qtt = B.T @ Qyy @ B

    alpha = modify_alpha_prime(alpha_prime, alpha_0, m, alpha_method, r)

    if np.sum(Qtt) > 1.01 * r or np.sum(Qtt) < 0.99 * r:
        raise Exception(
            "It appears that Qtt is not the identity matrix using this B-matrix \n \
                        B = {}".format(B)
        )
    Qtt_inv = np.eye(r)

    return m, n, r, A, alpha, sigma, Qyy, Qyy_inv, B.T, Qtt, Qtt_inv


def setup_GNSS_example_Safoora(alpha_0=0.01, alpha_prime=0.01, alpha_method="Kok_IDS"):
    dir_to_load = (
        r"C:\Users\bgvannoort\Documents\IDS\Code\GNSS_examples_paper\Setup 2 Safoora"
    )

    A = np.loadtxt(os.path.join(dir_to_load, "A_mat.txt"), delimiter=",")
    B = np.loadtxt(os.path.join(dir_to_load, "B_mat.txt"), delimiter=",")
    Qyy = np.loadtxt(os.path.join(dir_to_load, "Qyy.txt"), delimiter=",")
    Qyy_inv = np.linalg.inv(Qyy)

    Qtt = B.T @ Qyy @ B
    m, n = A.shape
    r = m - n

    alpha = modify_alpha_prime(alpha_prime, alpha_0, m, alpha_method, r)

    if np.sum(Qtt) > 1.01 * r or np.sum(Qtt) < 0.99 * r:
        raise Exception(
            "It appears that Qtt is not the identity matrix using this B-matrix \n \
                        B = {}".format(B)
        )

    Qtt_inv = np.eye(r)
    return m, n, r, A, alpha, None, Qyy, Qyy_inv, B.T, Qtt, Qtt_inv


def load_indices_partitions_GNSS_ex(boolean_sep_order, load_colors=False):
    # if boolean == true, then we would like to plot also the separate order
    # of the IDS partition, else it should be false.
    # This is for the example_type of 'SPP_GNSS'
    dirt = r"C:\Users\bgvannoort\Documents\IDS\Sim Data\SPP_GNSS"

    if boolean_sep_order:
        fname = "indices_partition_separate_order.json"

    else:
        fname = "indices_partition_no_separate_order.json"

    with open(os.path.join(dirt, fname), "r") as f:
        indices_partition = json.load(f)

    if load_colors:
        ## Write code that also loads the corresponding colors:
        fname_colors = "colors_partitions_dict.json"
        with open(os.path.join(dirt, fname_colors), "r") as f:
            colors_partitions = json.load(f)
        return indices_partition, colors_partitions

    return indices_partition


def IDS_typeA_mult_its(
    t,
    S,
    cti_list,
    idx_S,
    separate_order_IDS=False,
    inPlane=False,
    lastOMT=True,
    alpha_0=0.01,
    type_of_alpha="Kok_IDS",
    type_of_example="simple",
    qmax=5,
):
    """

    Parameters
    ----------
    t : Array of misclosure vectors (3, nsamples)
    S: the list of indices corresponding to the fault vectors.
        Note that S=[0,3,7] corresponds to H{1,4,8} due to python idxes starting at 0
    cti_list: None or list
        If None, S should be empty. Otherwise, it is a list of ci_vectors.
        First iteration, cti_list will be set to Bt @ np.eye(m), but afterwards
        it contains all the projected cti_vectors, i.e. cti_bar. It will remain to be rxm,
        as the at the next iteration, the corresponding removed w-test/observation for ci
        is simply set to the zero vector.
        Since we take the max of w-test values, we will never select this one.
    idx_S: int
        index of the current hypothesis, if S=empty, idx_S = 0
        otherwise, it is obtained with the get_idx_hypt function.
    separate_order_IDS : boolean, optional
        DESCRIPTION. Whether or not for IDS the partitionings P12 and P21 should be
        treated as 'different', such that they get a different color when plotting.
        The default is False, i.e. not different.
    inPlane : boolean, optional
        DESCRIPTION. If True, then the misclosure space will be sampled on a plane
        (as a cross section intersection of the 3D sphere). The default is False.
    lastOMT : boolean, optional
        DESCRIPTION. If true, then after identification of q=2 outliers, we carry out an additional OMT.
        The default is True.
    type_of_DS: str, optional
        DESCRIPTION. Determines the type of IDS specified in the report. Can be A, B or C.
    alpha_0: float, optional
        significance test of the w-test
    qmax: int, optional
        specifies how many outliers we can at most identify.

    Returns
    -------

    Pis : list, array
        Containing the indices corresponding to the identified hypothesis.

    """

    # print('At start of function')

    # -------------- Problem setup ---------------------------------

    m, n, r, A, alpha, sigma, Qyy, Qyy_inv, Bt, Qtt, Qtt_inv = load_setup_parameters(
        type_of_example=type_of_example, alpha_method=type_of_alpha, alpha_0=alpha_0
    )

    # ------------------- start iterative datasnooping -----------------

    q_S = len(S)

    # Threshold for w-tests
    w_alpha0 = scipy.stats.norm.isf(alpha_0 / 2)
    # print('w_alpha0', w_alpha0)

    if q_S == r - 1 or q_S >= qmax:
        # We can only carry out an additional OMT, but after rejection, we cannot identify.
        Pis = np.ones(t.shape[1]) * idx_S
        C_S = np.eye(m)[:, np.array(S)]
        ct_S = Bt @ C_S
        P_cts_perp = P_perp(P_mat(ct_S, Qtt))

        alpha_prime_final = modify_alpha_prime(
            alpha, alpha_0, m, method=type_of_alpha, df_chi2_new=r - q_S
        )
        OMTs = np.einsum("ij,jm,mi->i", (P_cts_perp @ t).T, Qtt_inv, P_cts_perp @ t)

        ## Where does the OMT fail because of redundancy issues?
        T_OMT_final = scipy.stats.chi2.isf(alpha_prime_final, df=r - q_S)
        OMT_failed = OMTs > T_OMT_final

        # If lastOMT is true: But in theory should always be done if we want to compare with RIDS.
        Pis[OMT_failed] = -1

        ## NOTE:
        ## If we get to q_S = qmax, we only do an OMT and for the rejected samples, we put them
        ## to the undecided region !!!

        return Pis

    elif q_S > r - 1:
        raise Exception(
            "Cannot exclude more than r-1 outliers, currently S={} and q_S={}".format(
                S, q_S
            )
        )

    # If S = [], it is the first function call. Set cti_
    elif len(S) == 0:
        if cti_list is not None:
            raise ValueError(
                "S is empty (first function call), so parameter cti_list should be None, not {}".format(
                    cti_list
                )
            )

        # print('in this loop?')
        Pis = np.zeros(t.shape[1])
        cti_list = Bt @ np.eye(m)

        # Start with first OMT
        threshold_OMT = scipy.stats.chi2.isf(alpha, df=r)
        t_OMT = np.einsum("mi,ij,jm->m", t.T, Qtt_inv, t)
        (P0_indices,) = np.where(t_OMT < threshold_OMT)
        # print('alpha', alpha)
        (P_outside_ind,) = np.where(t_OMT > threshold_OMT)

        nr_rejected = len(P_outside_ind)
        t_rej = t[:, P_outside_ind]
        w_tests = np.zeros((m, nr_rejected))
        for i in range(m):
            pcti = P_mat(cti_list[:, i].reshape(-1, 1), Qtt)
            pcti_t = pcti @ t_rej
            wi_squared = np.einsum("mi,ij,jm->m", pcti_t.T, Qtt_inv, pcti_t)
            w_tests[i, :] = wi_squared

        idx_max_wtests = np.argmax(w_tests, axis=0)
        max_wtests = np.max(w_tests, axis=0)

        # For type A IDS, there is an undecided region where OMT fails, but w-test is too small.
        where_undecided = max_wtests < w_alpha0**2
        Pis[P_outside_ind[where_undecided]] = -1
        for i in range(m):
            # Select points where w-test is max, and not smaller than w_alpha_0
            where_to_identify = np.logical_and(idx_max_wtests == i, ~where_undecided)
            if (where_to_identify).sum() == 0:
                continue

            # print("S,", S)

            S_new = S + [i]  # append i to S and go to next iteration
            # print('S_new', S_new)
            pcti = P_mat(cti_list[:, i].reshape(-1, 1), Qtt)
            pcti_perp = P_perp(pcti)
            cti_list_new = pcti_perp @ cti_list
            # print(cti_list)
            idx_S = get_idx_hypt_RIDS(m, subset=S_new)

            cti_list_new[:, i] = np.zeros(r)

            Pis[P_outside_ind[where_to_identify]] = IDS_typeA_mult_its(
                pcti_perp @ t_rej[:, where_to_identify],
                S_new,
                cti_list_new,
                idx_S=idx_S,
                separate_order_IDS=separate_order_IDS,
                inPlane=inPlane,
                lastOMT=lastOMT,
                alpha_0=alpha_0,
                type_of_alpha=type_of_alpha,
                type_of_example=type_of_example,
                qmax=qmax,
            )
        return Pis

    else:
        # create new Pis list to return
        Pis = np.ones(t.shape[1]) * idx_S
        # Compute the new OMT
        c_S = np.eye(m)[:, np.array(S)].reshape((m, len(S)))
        ct_S = Bt @ c_S
        pctS_perp = P_perp(P_mat(ct_S, Qtt))

        alpha_prime_new = modify_alpha_prime(
            alpha, alpha_0, m, method=type_of_alpha, df_chi2_new=r - q_S
        )
        T_OMT_new = scipy.stats.chi2.isf(alpha_prime_new, df=r - q_S)
        OMTnew = np.einsum("ij,jm,mi->i", (pctS_perp @ t).T, Qtt_inv, pctS_perp @ t)

        idx_where_OMT_passed = OMTnew < T_OMT_new

        nr_not_accepted = np.sum(~idx_where_OMT_passed)
        # If all are accepted
        if nr_not_accepted == 0:
            return Pis

        w_bar_tests = np.zeros((m, t.shape[1]))
        # loop over all projected cti_vectors ci
        for i in range(m):
            if i in S:
                continue
            cti = cti_list[:, i].reshape(-1, 1)
            pcti = P_mat(cti, Qtt)
            pcti_t = pcti @ t
            w_bar_squared = np.einsum("ij,jm,mi->i", pcti_t.T, Qtt_inv, pcti_t)
            w_bar_tests[i, :] = w_bar_squared

        # print(w_bar_tests)
        max_w_tests = np.max(w_bar_tests, axis=0)
        idx_max_wtests = np.argmax(w_bar_tests, axis=0)

        # Again, undecided region here for type A IDS
        smaller_w_alpha0 = max_w_tests < w_alpha0**2
        where_undecided = np.logical_and(smaller_w_alpha0, ~idx_where_OMT_passed)
        Pis[where_undecided] = -1

        for i in range(m):
            idx_to_identify = np.logical_and(idx_max_wtests == i, ~idx_where_OMT_passed)
            idx_to_identify = np.logical_and(idx_to_identify, ~smaller_w_alpha0)
            if np.sum(idx_to_identify) == 0:
                continue
            ## Create new cti_new_lsit and reproject to cti_bar_perp
            cti_bar = cti_list[:, i].reshape(-1, 1)
            Pcti_bar_perp = P_perp(P_mat(cti_bar, Qtt))
            cti_list_new = Pcti_bar_perp @ cti_list

            # print('S, ', S)
            S_new = S + [i]
            S_new.sort()
            # print("S_new", S_new)

            # make sure that cti_list_new contains zero vectors for indices in S_new
            cti_list_new[:, np.array(S_new)] = np.zeros((r, len(S_new)))
            idx_S_new = get_idx_hypt_RIDS(m, S_new)

            Pis[idx_to_identify] = IDS_typeA_mult_its(
                Pcti_bar_perp @ t[:, idx_to_identify],
                S_new,
                cti_list_new,
                idx_S=idx_S_new,
                separate_order_IDS=separate_order_IDS,
                inPlane=inPlane,
                lastOMT=lastOMT,
                alpha_0=alpha_0,
                type_of_alpha=type_of_alpha,
                type_of_example=type_of_example,
                qmax=qmax,
            )
        return Pis


def IDS_typeB_mult_its(
    t,
    S,
    cti_list,
    idx_S,
    separate_order_IDS=False,
    inPlane=False,
    lastOMT=True,
    alpha_0=0.01,
    type_of_alpha="Kok_IDS",
    type_of_example="simple",
    qmax=5,
):
    """

    Parameters
    ----------
    t : Array of misclosure vectors (3, nsamples)
    S: the list of indices corresponding to the fault vectors.
        Note that S=[0,3,7] corresponds to H{1,4,8} due to python idxes starting at 0
    cti_list: None or list
        If None, S should be empty. Otherwise, it is a list of ci_vectors.
        First iteration, cti_list will be set to Bt @ np.eye(m), but afterwards
        it contains all the projected cti_vectors, i.e. cti_bar. It will remain to be rxm,
        as the at the next iteration, the corresponding removed w-test/observation for ci
        is simply set to the zero vector.
        Since we take the max of w-test values, we will never select this one.
    idx_S: int
        index of the current hypothesis, if S=empty, idx_S = 0
        otherwise, it is obtained with the get_idx_hypt function.
    separate_order_IDS : boolean, optional
        DESCRIPTION. Whether or not for IDS the partitionings P12 and P21 should be
        treated as 'different', such that they get a different color when plotting.
        The default is False, i.e. not different.
    inPlane : boolean, optional
        DESCRIPTION. If True, then the misclosure space will be sampled on a plane
        (as a cross section intersection of the 3D sphere). The default is False.
    lastOMT : boolean, optional
        DESCRIPTION. If true, then after identification of q=2 outliers, we carry out an additional OMT.
        The default is True.
    type_of_DS: str, optional
        DESCRIPTION. Determines the type of IDS specified in the report. Can be A, B or C.
    alpha_0: float, optional
        significance test of the w-test
    qmax: int, optional
        specifies how many outliers we can at most identify.

    Returns
    -------

    Pis : list, array
        Containing the indices corresponding to the identified hypothesis.

    """

    # -------------- Problem setup ---------------------------------

    m, n, r, A, alpha, sigma, Qyy, Qyy_inv, Bt, Qtt, Qtt_inv = load_setup_parameters(
        type_of_example=type_of_example, alpha_method=type_of_alpha, alpha_0=alpha_0
    )

    # ------------------- start iterative datasnooping -----------------

    q_S = len(S)

    if q_S == r - 1 or q_S >= qmax:
        # We can only carry out an additional OMT, but after rejection, we cannot identify.
        Pis = np.ones(t.shape[1]) * idx_S
        C_S = np.eye(m)[:, np.array(S)]
        ct_S = Bt @ C_S
        P_cts_perp = P_perp(P_mat(ct_S, Qtt))

        alpha_prime_final = modify_alpha_prime(
            alpha, alpha_0, m, method=type_of_alpha, df_chi2_new=r - q_S
        )
        # print('IDS_mult_ids: alpha_prime_final ', alpha_prime_final)

        OMTs = np.einsum("ij,jm,mi->i", (P_cts_perp @ t).T, Qtt_inv, P_cts_perp @ t)

        ## Where does the OMT fail because of redundancy issues?
        T_OMT_final = scipy.stats.chi2.isf(alpha_prime_final, df=r - q_S)
        # print("IDS_mult_its: T_OMT_final", T_OMT_final)
        OMT_failed = OMTs > T_OMT_final

        # If lastOMT is true: But in theory should always be done if we want to compare with RIDS.
        Pis[OMT_failed] = -1

        ## NOTE:
        ## If we get to q_S = qmax, we only do an OMT and for the rejected samples, we put them
        ## to the undecided region !!!

        return Pis

    elif q_S > r - 1:
        raise Exception(
            "Cannot exclude more than r-1 outliers, currently S={} and q_S={}".format(
                S, q_S
            )
        )

    # If S = [], it is the first function call. Set cti_
    elif len(S) == 0:
        if cti_list is not None:
            raise ValueError(
                "S is empty (first function call), so parameter cti_list should be None, not {}".format(
                    cti_list
                )
            )

        # print('in this loop?')
        Pis = np.zeros(t.shape[1])
        cti_list = Bt @ np.eye(m)

        # Start with first OMT
        threshold_OMT = scipy.stats.chi2.isf(alpha, df=r)
        t_OMT = np.einsum("mi,ij,jm->m", t.T, Qtt_inv, t)
        (P0_indices,) = np.where(t_OMT < threshold_OMT)

        # print('IDS_mult_its: threshold_OMT', threshold_OMT)
        (P_outside_ind,) = np.where(t_OMT > threshold_OMT)

        nr_rejected = len(P_outside_ind)
        t_rej = t[:, P_outside_ind]
        w_tests = np.zeros((m, nr_rejected))
        for i in range(m):
            if i in S:
                continue
            pcti = P_mat(cti_list[:, i].reshape(-1, 1), Qtt)
            pcti_t = pcti @ t_rej
            wi_squared = np.einsum("mi,ij,jm->m", pcti_t.T, Qtt_inv, pcti_t)
            w_tests[i, :] = wi_squared

        idx_max_wtests = np.argmax(w_tests, axis=0)
        max_wtests = np.max(w_tests, axis=0)
        for i in range(m):
            where_to_identify = idx_max_wtests == i
            if (where_to_identify).sum() == 0:
                continue
            S_new = S + [i]  # append i to S and go to next iteration
            # print('S_new', S_new)
            pcti = P_mat(cti_list[:, i].reshape(-1, 1), Qtt)
            pcti_perp = P_perp(pcti)
            cti_list_new = pcti_perp @ cti_list
            # print(cti_list)
            idx_S = get_idx_hypt_RIDS(m, subset=S_new)

            Pis[P_outside_ind[where_to_identify]] = IDS_typeB_mult_its(
                pcti_perp @ t_rej[:, where_to_identify],
                S_new,
                cti_list_new,
                idx_S=idx_S,
                separate_order_IDS=separate_order_IDS,
                inPlane=inPlane,
                lastOMT=lastOMT,
                alpha_0=alpha_0,
                type_of_alpha=type_of_alpha,
                type_of_example=type_of_example,
                qmax=qmax,
            )
        return Pis

    else:
        # create new Pis list to return
        Pis = np.ones(t.shape[1]) * idx_S
        # Compute the new OMT
        c_S = np.eye(m)[:, np.array(S)].reshape((m, len(S)))
        ct_S = Bt @ c_S
        pctS_perp = P_perp(P_mat(ct_S, Qtt))

        # print("IDS_mult_its: S,", S)
        alpha_prime_new = modify_alpha_prime(
            alpha, alpha_0, m, method=type_of_alpha, df_chi2_new=r - q_S
        )
        T_OMT_new = scipy.stats.chi2.isf(alpha_prime_new, df=r - q_S)
        # print('IDS_mult_its: T_OMT_new', T_OMT_new)
        OMTnew = np.einsum("ij,jm,mi->i", (pctS_perp @ t).T, Qtt_inv, pctS_perp @ t)

        idx_where_OMT_passed = OMTnew < T_OMT_new

        nr_not_accepted = np.sum(~idx_where_OMT_passed)
        # If all are accepted
        if nr_not_accepted == 0:
            return Pis

        w_bar_tests = np.zeros((m, t.shape[1]))
        # loop over all projected cti_vectors ci
        for i in range(m):
            if i in S:
                continue
            cti = cti_list[:, i].reshape(-1, 1)
            pcti = P_mat(cti, Qtt)
            pcti_t = pcti @ t
            w_bar_squared = np.einsum("ij,jm,mi->i", pcti_t.T, Qtt_inv, pcti_t)
            w_bar_tests[i, :] = w_bar_squared

        # print(w_bar_tests)
        max_w_tests = np.max(w_bar_tests, axis=0)
        idx_max_wtests = np.argmax(w_bar_tests, axis=0)

        for i in range(m):
            idx_to_identify = idx_max_wtests == i
            if np.sum(idx_to_identify) == 0:
                continue
            ## Create new cti_new_lsit and reproject to cti_bar_perp
            cti_bar = cti_list[:, i].reshape(-1, 1)
            Pcti_bar_perp = P_perp(P_mat(cti_bar, Qtt))
            cti_list_new = Pcti_bar_perp @ cti_list

            # print('S, ', S)
            S_new = S + [i]
            S_new.sort()
            # print("S_new", S_new)
            idx_S_new = get_idx_hypt_RIDS(m, S_new)

            # the total combination of failing the current OMT and where wi=max.
            idx_to_identify = np.logical_and(idx_to_identify, ~idx_where_OMT_passed)

            Pis[idx_to_identify] = IDS_typeB_mult_its(
                Pcti_bar_perp @ t[:, idx_to_identify],
                S_new,
                cti_list_new,
                idx_S=idx_S_new,
                separate_order_IDS=separate_order_IDS,
                inPlane=inPlane,
                lastOMT=lastOMT,
                alpha_0=alpha_0,
                type_of_alpha=type_of_alpha,
                type_of_example=type_of_example,
                qmax=qmax,
            )
        return Pis


def IDS_typeC_mult_its(
    t,
    S,
    cti_list,
    idx_S,
    separate_order_IDS=False,
    inPlane=False,
    lastOMT=True,
    alpha_0=0.01,
    type_of_alpha="Kok_IDS",
    type_of_example="simple",
    qmax=5,
):
    """

    Parameters
    ----------
    t : Array of misclosure vectors (3, nsamples)
    S: the list of indices corresponding to the fault vectors.
        Note that S=[0,3,7] corresponds to H{1,4,8} due to python idxes starting at 0
    cti_list: None or list
        If None, S should be empty. Otherwise, it is a list of ci_vectors.
        First iteration, cti_list will be set to Bt @ np.eye(m), but afterwards
        it contains all the projected cti_vectors, i.e. cti_bar. It will remain to be rxm,
        as the at the next iteration, the corresponding removed w-test/observation for ci
        is simply set to the zero vector.
        Since we take the max of w-test values, we will never select this one.
    idx_S: int
        index of the current hypothesis, if S=empty, idx_S = 0
        otherwise, it is obtained with the get_idx_hypt function.
    separate_order_IDS : boolean, optional
        DESCRIPTION. Whether or not for IDS the partitionings P12 and P21 should be
        treated as 'different', such that they get a different color when plotting.
        The default is False, i.e. not different.
    inPlane : boolean, optional
        DESCRIPTION. If True, then the misclosure space will be sampled on a plane
        (as a cross section intersection of the 3D sphere). The default is False.
    lastOMT : boolean, optional
        DESCRIPTION. If true, then after identification of q=2 outliers, we carry out an additional OMT.
        The default is True.
    type_of_DS: str, optional
        DESCRIPTION. Determines the type of IDS specified in the report. Can be A, B or C.
    alpha_0: float, optional
        significance test of the w-test
    qmax: int, optional
        specifies how many outliers we can at most identify.

    Returns
    -------

    Pis : list, array
        Containing the indices corresponding to the identified hypothesis.

    """

    # -------------- Problem setup ---------------------------------

    m, n, r, A, alpha, sigma, Qyy, Qyy_inv, Bt, Qtt, Qtt_inv = load_setup_parameters(
        type_of_example=type_of_example, alpha_method=type_of_alpha, alpha_0=alpha_0
    )

    # print('in IDS C: alpha_0', alpha_0)
    # print('In IDS C: alpha prime ', alpha )
    # print('In IDS  C: type of alpha ', type_of_alpha )
    # ------------------- start iterative datasnooping -----------------

    q_S = len(S)

    # Threshold for w-tests
    w_alpha0 = scipy.stats.norm.isf(alpha_0 / 2)

    if q_S == r - 1 or q_S >= qmax:
        # We can only carry out an additional OMT, but after rejection, we cannot identify.
        Pis = np.ones(t.shape[1]) * idx_S

        w_tests = np.zeros((m, t.shape[1]))
        for i in range(m):
            pcti = P_mat(cti_list[:, i].reshape(-1, 1), Qtt)
            pcti_t = pcti @ t
            wi_squared = np.einsum("mi,ij,jm->m", pcti_t.T, Qtt_inv, pcti_t)
            w_tests[i, :] = wi_squared

        # There is no OMT for type C; check instead with w-tests
        # print('t.shape', t.shape)
        max_w_tests = np.max(w_tests, axis=0)
        # print(max_w_tests)
        where_undec = max_w_tests > w_alpha0**2

        # If lastOMT is 'true': But in theory should always be done if we want to compare with RIDS.
        # For type C IDS, we check all w-tests (which are all equal) and check if they are larger than w_alpha0
        Pis[where_undec] = -1

        ## NOTE:
        ## If we get to q_S = qmax, we only do an OMT and for the rejected samples, we put them
        ## to the undecided region !!!

        return Pis

    elif q_S > r - 1:
        raise Exception(
            "Cannot exclude more than r-1 outliers, currently S={} and q_S={}".format(
                S, q_S
            )
        )

    # If S = [], it is the first function call. Set cti_
    elif len(S) == 0:
        if cti_list is not None:
            raise ValueError(
                "S is empty (first function call), so parameter cti_list should be None, not {}".format(
                    cti_list
                )
            )

        # print('in this loop?')
        Pis = np.zeros(t.shape[1])
        cti_list = Bt @ np.eye(m)

        ## For type C there is no OMT, so we set everything to 'pass'
        w_tests = np.zeros((m, t.shape[1]))
        for i in range(m):
            if i in S:
                continue
            pcti = P_mat(cti_list[:, i].reshape(-1, 1), Qtt)
            pcti_t = pcti @ t
            wi_squared = np.einsum("mi,ij,jm->m", pcti_t.T, Qtt_inv, pcti_t)
            w_tests[i, :] = wi_squared

        idx_max_wtests = np.argmax(w_tests, axis=0)
        max_wtests = np.max(w_tests, axis=0)

        P_outside_ind = max_wtests > w_alpha0**2

        for i in range(m):
            # Select points where w-test is max, and not smaller than w_alpha_0
            where_to_identify = np.logical_and(idx_max_wtests == i, P_outside_ind)
            if (where_to_identify).sum() == 0:
                continue

            # print("S,", S):

            S_new = S + [i]  # append i to S and go to next iteration
            # print('S_new', S_new)
            pcti = P_mat(cti_list[:, i].reshape(-1, 1), Qtt)
            pcti_perp = P_perp(pcti)
            cti_list_new = pcti_perp @ cti_list
            # print(cti_list)
            idx_S = get_idx_hypt_RIDS(m, subset=S_new)

            # print("cti_list", np.round(cti_list_new, 3) )
            # print("S_new", S_new)

            Pis[where_to_identify] = IDS_typeC_mult_its(
                pcti_perp @ t[:, where_to_identify],
                S_new,
                cti_list_new,
                idx_S=idx_S,
                separate_order_IDS=separate_order_IDS,
                inPlane=inPlane,
                lastOMT=lastOMT,
                alpha_0=alpha_0,
                type_of_alpha=type_of_alpha,
                type_of_example=type_of_example,
                qmax=qmax,
            )
        return Pis

    else:
        # create new Pis list to return
        Pis = np.ones(t.shape[1]) * idx_S
        # # Compute the new OMT
        # c_S = np.eye(m)[:, np.array(S)].reshape((m,len(S)))
        # ct_S = Bt @ c_S
        # pctS_perp = P_perp(P_mat(ct_S, Qtt))

        # alpha_prime_new = modify_alpha_prime(alpha, alpha_0, m, method=type_of_alpha, df_chi2_new=r-q_S)
        # T_OMT_new = scipy.stats.chi2.isf(alpha_prime_new, df=r-q_S)
        # OMTnew = np.einsum('ij,jm,mi->i', (pctS_perp @ t).T, Qtt_inv, pctS_perp @ t)

        # idx_where_OMT_passed = OMTnew < T_OMT_new

        # nr_not_accepted = np.sum(~idx_where_OMT_passed)
        # # If all are accepted
        # if nr_not_accepted == 0:
        #     return Pis

        # print('t.shape', t.shape)
        w_bar_tests = np.zeros((m, t.shape[1]))
        # loop over all projected cti_vectors ci
        for i in range(m):
            if i in S:
                continue
            cti = cti_list[:, i].reshape(-1, 1)
            # print('i, cti.flatten()', i,  cti.flatten())
            pcti = P_mat_invQ(cti, Qtt_inv)
            pcti_t = pcti @ t
            w_bar_squared = np.einsum("ij,jm,mi->i", pcti_t.T, Qtt_inv, pcti_t)
            w_bar_tests[i, :] = w_bar_squared

        # print(w_bar_tests)
        max_w_tests = np.max(w_bar_tests, axis=0)
        idx_max_wtests = np.argmax(w_bar_tests, axis=0)

        # print('max_tests.shape', max_w_tests.shape, idx_max_wtests.shape)
        # Again, undecided region here for type A IDS
        idx_accept_S = max_w_tests < w_alpha0**2

        # print("max of w_bar_tests[:10]", max_w_tests[:10])
        # print('w_bar_tests[:10]', w_bar_tests[:10,:])
        if np.sum(~idx_accept_S) == 0:  # accept all current sets of S as H_S
            return Pis

        for i in range(m):
            idx_to_identify = np.logical_and(idx_max_wtests == i, ~idx_accept_S)
            if np.sum(idx_to_identify) == 0:
                continue
            ## Create new cti_new_lsit and reproject to cti_bar_perp
            cti_bar = cti_list[:, i].reshape(-1, 1)
            Pcti_bar_perp = P_perp(P_mat(cti_bar, Qtt))
            cti_list_new = Pcti_bar_perp @ cti_list

            # print('S, ', S)
            S_new = S + [i]
            S_new.sort()

            # print("cti_list_new", np.round(cti_list_new, 3) )
            # print("S_new", S_new)
            idx_S_new = get_idx_hypt_RIDS(m, S_new)

            Pis[idx_to_identify] = IDS_typeC_mult_its(
                Pcti_bar_perp @ t[:, idx_to_identify],
                S_new,
                cti_list_new,
                idx_S=idx_S_new,
                separate_order_IDS=separate_order_IDS,
                inPlane=inPlane,
                lastOMT=lastOMT,
                alpha_0=alpha_0,
                type_of_alpha=type_of_alpha,
                type_of_example=type_of_example,
                qmax=qmax,
            )
        return Pis


def ordinary_DIA_testing_only(
    t,
    Qtt_inv,
    B_T,
    P_region,
    type_of_example="simple",
    alpha_0=0.01,
    type_of_alpha="Kok_IDS",
    qmax=2,
):
    # this function is almost identical to the ordinary_DIA function, though here we only carry out the testing, not the plotting and saving of the data.
    # so we merely compute the Pis.

    m, n, r, A, alpha_prime, sigma, Qyy, Qyy_inv, Bt, Qtt, Qtt_inv = (
        load_setup_parameters(
            type_of_example=type_of_example, alpha_method=type_of_alpha, alpha_0=alpha_0
        )
    )

    c_vec_list = []
    idx_tot = 0
    q = 1
    while q <= qmax:
        for idxs in itertools.combinations(range(m), q):
            part_ = "P" + "".join(
                str(i + 1) for i in idxs
            )  # e.g., 'P13' for i=0 and j=2
            c_vec_ = np.eye(m)[:, list(idxs)]  # select columns
            c_vec_list.append([part_, c_vec_])

        q += 1

    k = len(c_vec_list)

    # print("P_region", P_region)
    # print("Warning: double check whether indices allign traditional DIA function")
    # Problem setup
    # t is an r x N dim array
    r = t.shape[0]
    Pis = np.zeros(t.shape[1])

    threshold_OMT = scipy.stats.chi2.isf(alpha_prime, df=r)
    print("ordinary_DIA: alpha_prime", alpha_prime)
    t_OMT = np.einsum("mi,ij,jm->m", t.T, Qtt_inv, t)
    (P0_indices,) = np.where(t_OMT < threshold_OMT)

    (P_outside_ind,) = np.where(t_OMT > threshold_OMT)

    # the P0 region for the normal DIA method is exactly identical to the one as before.

    # this stores the 1-cumulative density function, i.e. 1-S_i = CDF_{chi^2_q_i}(Tq_i) from DIA DS paper (SZ and PT)
    # note that we thus have to take argmin later!!
    S_array = np.zeros((k, len(P_outside_ind)))

    t_rej = t[:, P_outside_ind]  # rejected samples / t-grid points

    for i, (partition, c_vec) in enumerate(c_vec_list):
        q = c_vec.shape[1]

        cti = B_T @ c_vec
        Pcti = P_mat(cti, Qtt)
        t_proj = Pcti @ t_rej
        Tq_test = np.einsum("mi,ij,jm->m", t_proj.T, Qtt_inv, t_proj)
        Tq_test_alpha = scipy.stats.chi2.sf(Tq_test, df=q)

        S_array[i, :] = Tq_test_alpha

    Pijs = (
        np.argmin(S_array, axis=0) + 1
    )  # P0=0, P1=1, P2=2 etc.., matlab indexing starts at 1 as well.

    Pis[P_outside_ind] = Pijs

    return Pis


def compute_Pis(
    t_sample: np.ndarray,
    Qtt: np.ndarray,
    B_T: np.ndarray,
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
    separate_order_IDS: bool = False,
    inPlane: bool = False,
    bool_start_with_OMT: bool = False,
):
    """
    Compute Pi_s values based on the selected hypothesis testing approach (DS, IDS, R_IDS, or classical DIA).

    Parameters
    ----------
    t_sample : np.ndarray
        Matrix of t-sample vectors, assumed to be N x r in shape.
    Qtt : np.ndarray
        Covariance matrix of the t-sample vectors.
    B_T : np.ndarray
        Left null space of the A matrix (e.g., from model setup).
    alpha_prime : float
        Type I error probability. Level of significance. Depending on alpha_method
        it may not be used as it will use alpha_0 to determine the level of significance.
    S : list
        List of indices corresponding to the identified outliers
        It should be empty at the start of the function call for IDS and RIDS.
        Since these functions are iterative, these functions append indices and
        call the same function again.
    qmax : int, optional
        Maximum number of outliers allowed to be identified. Default is 5.
    idx_S : int, optional
        Current index in the iterative outlier identification process. Default is 0.
    cti_list : list, optional
        List of cti vectors used during the iterative procedures. Should start as None.
        Warning: the IDS functions take cti_list as input, while the RIDS functions
            take the ci_list as input, i.e. the 'ordinary' fault vectors ci rather than cti!
    type_of_testing : str, optional
        Specifies the hypothesis testing framework. Must be one of:
        'classical DIA', 'DS', 'IDS', or 'R_IDS'.
    P_region : list, optional
        List containing all the several regions / partitions / indices with colors
        for plotting purposes. The default is [].
    alpha_0 : float, optional
        Level of significance of the w-tests. The default is 0.01.
    type_of_DS : str, optional
        Data snooping type. Can be A, B, C, or D (only for data snooping).
        The default is 'B'.
    lastOMT : bool, optional
        Whether or not a final OMT is carried out. The default is True.
    alpha_method : str, optional
        Default "Kok_IDS". It specifies how to alter alpha' every iteration in IDS. Kok_IDS is equal to the B-method.
    type_of_example : str, optional
        Defines the example type. The default is 'simple'.
    separate_order_IDS: bool, optional
        Sets whether the order of identifications should be taken into account.
        If True, then P12 and P21 are given a different index. Only usefull for demonstration purposes
        when r=3. Default is False.
    inPlane: bool, optional
        specifies whether t_array is from just a sampled plane in R^3 or not. Default is
        not that, so inPlane=False. Relevant for some plotting and/or demonstration purposes.
    bool_start_with_OMT: bool, optional, Default False
        ONLY APPLICABLE FOR RIDS PROCEDURES
        If True, the RIDS test procedure will start with an OMT. In other words,
        the partitioning P0 will be identical to the standard OMT shaped P0. If True,
        the alpha_prime (=alpha_check) value taken will be the one that either
            - follows from the B-method of type_of_alpha='Kok_IDS'
            - or follows manually (given in the modify_alpha_prime method) if type_of_alpha='manual'


    Raises
    ------
    ValueError
        When 'type_of_testing' is not one of 'classical DIA', 'DS', or 'IDS' or 'R_IDS'

    Returns
    -------
    Pi_s: np.ndarray
        Array of the identifications of the different hypotheses c.f. the convention for indices as given in
        the variable indices_partition or using function get_idx_hypotheses

    """

    ## Old input:
    # t_sample: np.ndarray,
    #                 Qtt: np.ndarray,
    #                 B_T: np.ndarray,
    #                 alpha_prime: float,
    #                 P_region: list,
    #                 type_of_testing: str,
    #                 alpha_0: float = 0.01,
    #                 type_of_DS: str = 'B',
    #                 lastOMT: bool = False,
    #                 N: int = int(1e5),
    #                 alpha_method: str = 'Kok_IDS',
    #                 type_of_example: str = 'simple'

    m, n, r, A, alpha, sigma, Qyy, Qyy_inv, Bt, Qtt, Qtt_inv = load_setup_parameters(
        type_of_example=type_of_example, alpha_method=alpha_method, alpha_0=alpha_0
    )

    # print('Warning: the order of input parameters for the "compute_Pis" function has changed')
    # print('Check if the inputs still allign with this modification...')

    # N = int(1e5) # nr of samples to generate for y

    # y_mean = C_i @ b_i # we implicitly assume that x=0
    # y_sample = np.random.multivariate_normal(y_mean.flatten(), Qyy, size=(N,))

    # t_sample = B_T @ y_sample.T
    # print('t_sample.shape', t_sample.shape)
    # print("Compute_Pis: in this function ")

    if type_of_testing == "DS":
        Pi_s = ordinary_DS(
            t_sample,
            Qtt_inv,
            B_T,
            alpha_prime=alpha,  # IN DS function, we do not carry out loading the parameters
            P_region=P_region,
            alpha_0=alpha_0,
            type_of_DS=type_of_DS,
            type_of_example=type_of_example,
        )
    elif type_of_testing == "IDS":
        if type_of_DS == "A":
            Pi_s = IDS_typeA_mult_its(
                t_sample,
                S=S,
                cti_list=cti_list,
                idx_S=idx_S,
                separate_order_IDS=separate_order_IDS,
                inPlane=inPlane,
                lastOMT=lastOMT,
                alpha_0=alpha_0,
                type_of_alpha=alpha_method,
                type_of_example=type_of_example,
                qmax=qmax,
            )
            ## Leave old code commented, such that we can easily 'implement' it again
            ## for verification
            # _, _, Pi_s = IDS_typeA_2its(t_sample,
            #                             separate_order_IDS=separate_order_IDS,
            #                             inPlane=inPlane,
            #                             alpha_0=alpha_0,
            #                             lastOMT=lastOMT,
            #                             type_of_DS=type_of_DS,
            #                             type_of_alpha=alpha_method,
            #                             type_of_example=type_of_example)
        elif type_of_DS == "B":
            Pi_s = IDS_typeB_mult_its(
                t_sample,
                S=S,
                cti_list=cti_list,
                idx_S=idx_S,
                separate_order_IDS=separate_order_IDS,
                inPlane=inPlane,
                lastOMT=lastOMT,
                alpha_0=alpha_0,
                type_of_alpha=alpha_method,
                type_of_example=type_of_example,
                qmax=qmax,
            )
            # _, _, Pi_s = IDS_typeB_2its(t_sample,
            #                             separate_order_IDS=separate_order_IDS,
            #                             inPlane=inPlane,
            #                             alpha_0=alpha_0,
            #                             lastOMT=lastOMT,
            #                             type_of_DS=type_of_DS,
            #                             type_of_alpha=alpha_method,
            #                             type_of_example=type_of_example)
        elif type_of_DS == "C":
            Pi_s = IDS_typeC_mult_its(
                t_sample,
                S=S,
                cti_list=cti_list,
                idx_S=idx_S,
                separate_order_IDS=separate_order_IDS,
                inPlane=inPlane,
                lastOMT=lastOMT,
                alpha_0=alpha_0,
                type_of_alpha=alpha_method,
                type_of_example=type_of_example,
                qmax=qmax,
            )
            # _, _, Pi_s = IDS_typeC_2its(t_sample,
            #                             separate_order_IDS=separate_order_IDS,
            #                             inPlane=inPlane,
            #                             alpha_0=alpha_0,
            #                             lastOMT=lastOMT,
            #                             type_of_DS=type_of_DS,
            #                             type_of_alpha=alpha_method,
            #                             type_of_example=type_of_example)
    elif type_of_testing == "classical DIA":
        # print("compute Pis: type_of_example", type_of_example)
        Pi_s = ordinary_DIA_testing_only(
            t_sample,
            Qtt_inv,
            B_T,
            P_region,
            type_of_example=type_of_example,
            alpha_0=alpha_0,
            type_of_alpha=alpha_method,
            qmax=qmax,
        )

    elif type_of_testing == "R_IDS":
        if type_of_DS == "A":
            # print('compute_Pis: qmax=', qmax)
            Pi_s = RIDS_mult_its_type_A(
                t_sample,
                alpha_0=alpha_0,
                qmax=qmax,
                S=S,
                ci_list=cti_list,
                idx_S=idx_S,
                example_type=type_of_example,
                type_of_alpha=alpha_method,
                inPlane=inPlane,
                separate_order_IDS=separate_order_IDS,
                bool_start_with_OMT=bool_start_with_OMT,
            )

            # Pi_s = RIDS_2its_type_A(t_sample,
            #                         separate_order_IDS=separate_order_IDS,
            #                         inPlane=inPlane,
            #                         alpha_0=alpha_0,
            #                         example_type=type_of_example,
            #                         type_of_alpha=alpha_method)
        elif type_of_DS == "B":
            Pi_s = RIDS_mult_its_type_B(
                t_sample,
                alpha_0=alpha_0,
                qmax=qmax,
                S=S,
                ci_list=cti_list,
                idx_S=idx_S,
                example_type=type_of_example,
                type_of_alpha=alpha_method,
                inPlane=inPlane,
                separate_order_IDS=separate_order_IDS,
                bool_start_with_OMT=bool_start_with_OMT,
            )

            # Pi_s = RIDS_2its_type_B(t_sample,
            #                         separate_order_IDS=separate_order_IDS,
            #                         inPlane=inPlane,
            #                         alpha_0=alpha_0,
            #                         example_type=type_of_example,
            #                         type_of_alpha=alpha_method)
        elif type_of_DS == "C":
            Pi_s = RIDS_mult_its_type_C(
                t_sample,
                alpha_0=alpha_0,
                qmax=qmax,
                S=S,
                ci_list=cti_list,
                idx_S=idx_S,
                example_type=type_of_example,
                type_of_alpha=alpha_method,
                inPlane=inPlane,
                separate_order_IDS=separate_order_IDS,
                bool_start_with_OMT=bool_start_with_OMT,
            )

            # Pi_s = RIDS_2its_type_C(t_sample,
            #                         separate_order_IDS=separate_order_IDS,
            #                         inPlane=inPlane,
            #                         alpha_0=alpha_0,
            #                         example_type=type_of_example,
            #                         type_of_alpha=alpha_method)

    else:
        raise ValueError(
            "'type_of_testing' is not recognized for {}".format(type_of_testing)
        )

    return Pi_s


# CArry out hypothesis testing for Datasnooping (0 iterations), given the t-vector
# Returns the identification list of size t.shape[1] (i.e. number of points in t-space)
def ordinary_DS(
    t,
    Qtt_inv,
    B_T,
    alpha_prime,
    P_region,
    alpha_0=0.01,
    type_of_DS="B",
    alpha_i=None,
    type_of_example="simple",
):
    # P_region is a list of lists with first entries the partition (str) and the second the fault vector ci (array)
    # t is an r x N array
    Qtt = np.linalg.inv(Qtt_inv)
    identifications = np.zeros(t.shape[1])
    r, m = B_T.shape

    if type_of_DS == "D":
        bool_OMT, bool_w_test, alpha_i = import_DS_types(type_of_DS, alpha_i=alpha_i)
    else:
        bool_OMT, bool_w_test = import_DS_types(type_of_DS)

    # print('bool_OMT, bool_w_test', bool_OMT, bool_w_test)

    if bool_OMT:
        # Do one OMT
        threshold_OMT = scipy.stats.chi2.isf(alpha_prime, t.shape[0])
        OMT = np.einsum("mi,ij,jm->m", t.T, Qtt_inv, t)
        # Check which points < OMT_treshold
        (P0s,) = np.where(OMT < threshold_OMT)
        (P_outside,) = np.where(OMT > threshold_OMT)
        # print('alpha_prime', alpha_prime)
        # print('thresholdOMT, ', threshold_OMT)
    else:
        P0s = np.array([])
        P_outside = np.arange(len(t[1]))

    # Continue testing with the rejected t- samples or if type C data snooping
    T1_tests = np.zeros((m, len(P_outside)))
    t_rej = t[:, P_outside]  # rejected samples / t-grid points
    # Loop over the hypotheses
    for i in range(m):
        ci = np.eye(m)[:, i].reshape(-1, 1)
        cti = B_T @ ci
        Pcti = P_mat(cti, Qtt)
        t_proj = Pcti @ t_rej
        Tq_test = np.einsum("mi,ij,jm->m", t_proj.T, Qtt_inv, t_proj)
        T1_tests[i, :] = Tq_test

    # Carry out an actual w-test, where max(w_j) should be larger than Phi_alpha_0 (thus with P_omega)
    if bool_w_test:
        maxT_stats = np.max(T1_tests, axis=0)
        Pijs = np.argmax(T1_tests, axis=0) + 1
        phi_alpha_0 = scipy.stats.norm.isf(alpha_0 / 2) ** 2
        # samples rejected by OMT, but 'accepted' by w-test:
        (rejected_samples_w_test,) = np.where(maxT_stats < phi_alpha_0)

        if bool_OMT:
            Pijs[rejected_samples_w_test] = -1
        else:
            Pijs[rejected_samples_w_test] = 0
        identifications[P_outside] = Pijs
    else:
        # otherwise, simply select largest index (no P_omega) for type B or carry out type D
        # P0=0, P1=1, P2=2 etc.., matlab indexing starts at 1 as well.
        if type_of_DS == "B":
            Pijs = np.argmax(T1_tests, axis=0) + 1
            identifications[P_outside] = Pijs
        elif type_of_DS == "D":
            rejected_OMTs = OMT[P_outside]
            Pijs = np.argmax(T1_tests, axis=0) + 1
            threshold_2nd_OMT = scipy.stats.chi2.isf(alpha_i, df=t.shape[0] - 1)
            # print('alpha_i', alpha_i)
            passed_2nd_OMT = (
                rejected_OMTs - np.max(T1_tests, axis=0) < threshold_2nd_OMT
            )
            Pijs[~passed_2nd_OMT] = -1
            identifications[P_outside] = Pijs
    return identifications


def import_DS_types(type_of_DS, alpha_i=None):
    """


    Returns
        Booleans
        bool_OMT: boolean specifying whether OMT should be carried out.
        bool_w_test: boolean specifying whether w-test should be carried out,
                if true, automatically done with the alpha_0 level of significance definition.

    """

    error_message_str = (
        "Not possible to obtain parameters for this type of data snooping\n"
        "Parameter 'type_of_DS' should be 'A', 'B', or 'C', not {} ".format(type_of_DS)
    )

    if type_of_DS == "A":
        bool_OMT = True
        bool_w_test = True
    elif type_of_DS == "B":
        bool_OMT = True
        bool_w_test = False
    elif type_of_DS == "C":
        bool_OMT = False
        bool_w_test = True
    elif type_of_DS == "D":
        bool_OMT = True
        bool_w_test = False
        if alpha_i == None:
            alpha_i = 0.01

        return bool_OMT, bool_w_test, alpha_i
    else:
        raise ValueError(error_message_str)
    return bool_OMT, bool_w_test


@timing
def write_data_for_matlab(
    x,
    y,
    z,
    identifications,
    partition,
    DIA="IDS",
    write_grid_to_file=True,
    separate_order_IDS=False,
    inPlane=False,
    factor=None,
    workingDir=None,
    lastOMT=False,
    type_of_DS="B",
    type_of_alpha="iteration",
    a1=None,
    a2=None,
    removeFiles=False,
    alpha_0="default",
    manualPath=False,
    angle=None,
    type_of_example="simple",
    indices_partitions=None,
):
    # and returns the partitioned grid xx, yy, zz
    # Inputs
    # all speaks for itself.
    # DIA is the type of partitioning/testing procedure used. IDS, DS_DIA or ordinary DIA
    # write_grid_to_file, if true then the xx,yy,zz arrays are written to a file.
    # separate order IDS, is a boolean. Specify False (default) if we do not want to be able to show P12 and P21
    # If False and write_grid_to_file =True, then P21 and P12 files are identical.
    # if lastOMT = True, then we carry out an additional OMT at the end after removing two outliers (for our example.)
    # Note that this needs to be specified as this data will be written in a different directory.
    # type_of_DS specifies which type of DS should be carried out.
    # a1 and a2 are vectors. If inPlane is true, these vectors are span the plane in which we sample t
    # the boolean removeFiles is added to specify whether files should first be removed from the directory.
    # this is especially important to be set to True when plotting a partitioning inPlane (i.e. when inPlane=True).
    # If alpha_0=='default', it means that we take the default directory, i.e. where alpha_0=0.01 and the alpha' follows from B-method (beta=0.2).
    # if manualPath = True, we assign a directory to write data in manually.
    # angle is the rotation of the plane, only relevant for manualPath = True, and for inplane
    # type of example describes which example to use; simple is 4 measurements one unknown
    # and SPP_GNSS is a SPP gnss example with n=4 and m=7. Thus much more alt hypotheses
    # indices_partitions: Should be given as input if type_of_example == SPP_GNSS
    # because then we have many more alternative hypotheses than the simple example.

    factor = np.round(factor, 2)
    if DIA == "IDS":
        DIA_type = "\\IDS"
    elif DIA == "ordinary DIA":
        DIA_type = "\\ordinary_DIA"
    elif DIA == "DS_DIA":
        DIA_type = "\\DS_DIA"

    if type_of_example == "SPP_GNSS" or type_of_example == "ARAIM_UNDEC_GNSS":
        if indices_partitions == None:
            str_to_print = (
                "If we are running / saving data for the SPP GNSS example, we should provide"
                + ' the parameter "indices_partition"'
            )
            raise Exception(str_to_print)
        pass

    elif DIA == "IDS" and separate_order_IDS:
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

    index = indices_partitions[partition]

    if partition == "P99":
        print("Watch out: hardcoded to set index for P99 to -1")
        index = -1

    xx = x.flatten()
    yy = y.flatten()
    zz = z.flatten()

    indices = identifications == index

    xx[~indices] = np.nan * np.ones(sum(~indices))
    yy[~indices] = np.nan * np.ones(sum(~indices))
    zz[~indices] = np.nan * np.ones(sum(~indices))

    xx = xx.reshape(x.shape)
    yy = yy.reshape(x.shape)
    zz = zz.reshape(x.shape)

    if write_grid_to_file:
        if workingDir is None:
            cwd = os.getcwd()
        else:
            cwd = workingDir
        cwd = cwd.replace("\\Code", "") + "\\Sim Data"

        if type_of_example != "simple":
            cwd += "\\{}".format(type_of_example)

        if DIA == "ordinary DIA":
            startString = cwd + DIA_type
        else:
            startString = cwd + DIA_type + "\\" + type_of_DS

        if separate_order_IDS:
            startString = startString + "\\separate_partitionings"
        else:
            startString += "\\no_separate_partitionings"
        if inPlane:
            startString = startString + "\\inPlane"

        if lastOMT:
            startString = startString + "\\lastOMT"
        if DIA_type == "\\IDS":
            startString += "\\alpha_type_" + type_of_alpha

        if alpha_0 != "default":
            startString += "\\alpha_0=" + str(alpha_0)

        if factor is not None:
            directory = startString + "\\Partitioned_grid\\factor_" + str(factor) + "\\"
        else:
            directory = startString + "\\Partitioned_grid\\"

        if manualPath == True:
            if inPlane:
                print("Warning, we hardcode the directory if inPlane = True")
                directory = r"C:\Users\bgvannoort\Documents\IDS\Sim Data\SPP_GNSS\IDS\C\separate_partitionings\inPlane\alpha_type_Kok_IDS\Rotate_Along_c1c2_vectors"
                run = angle
                directory = os.path.join(directory, "Partitioned_grid", str(run)) + "\\"
            else:
                directory = os.path.join(
                    workingDir, "factor_{}".format(str(factor)), "Partitioned_grid"
                )

        if not os.path.exists(directory):
            os.makedirs(directory)
        # ## store the grid

        # first clean the directory if there are files in the directory.
        if removeFiles:
            for fname in os.listdir(directory):
                # Check if the file has a .txt extension
                if fname.endswith(".txt"):
                    # Construct the full file path
                    file_path = os.path.join(directory, fname)
                    # Remove the file
                    os.remove(file_path)

        print("Writing partitioned grid")
        print("Directory and file", directory + partition + "_xx.txt")
        np.savetxt(directory + "\\" + partition + "_xx.txt", xx)
        np.savetxt(directory + "\\" + partition + "_yy.txt", yy)
        np.savetxt(directory + "\\" + partition + "_zz.txt", zz)

    if inPlane:
        np.savetxt(directory + "\\a1_vec.txt", a1, delimiter=",")
        np.savetxt(directory + "\\a2_vec.txt", a2, delimiter=",")
    return xx, yy, zz


@timing
def write_data_for_matlab_RIDS(
    x,
    y,
    z,
    identifications,
    partition,
    DIA="IDS",
    write_grid_to_file=True,
    factor=None,
    lastOMT=False,
    type_of_RIDS="B",
    type_of_alpha="iteration",
    alpha_0="default",
    manualPath=False,
    angle=None,
    removeFiles=False,
    a1=None,
    a2=None,
    inPlane=False,
    type_of_example="simple",
    indices_partitions=None,
):
    # and returns the partitioned grid xx, yy, zz
    # Inputs
    # all speaks for itself.
    # DIA is the type of partitioning/testing procedure used. IDS, DS_DIA or ordinary DIA
    # write_grid_to_file, if true then the xx,yy,zz arrays are written to a file.
    # separate order IDS, is a boolean. Specify False (default) if we do not want to be able to show P12 and P21
    # If False and write_grid_to_file =True, then P21 and P12 files are identical.
    # if lastOMT = True, then we carry out an additional OMT at the end after removing two outliers (for our example.)
    # Note that this needs to be specified as this data will be written in a different directory.
    # type_of_DS specifies which type of DS should be carried out.
    # a1 and a2 are vectors. If inPlane is true, these vectors are span the plane in which we sample t
    # the boolean removeFiles is added to specify whether files should first be removed from the directory.
    # this is especially important to be set to True when plotting a partitioning inPlane (i.e. when inPlane=True).
    # if manualPath = True, we assign a directory to write data in manually.
    # angle is the rotation of the plane, only relevant for manualPath = True, and for inplane
    # type of example describes which example to use; simple is 4 measurements one unknown
    # and SPP_GNSS is a SPP gnss example with n=4 and m=7. Thus much more alt hypotheses
    # indices_partitions: Should be given as input if type_of_example == SPP_GNSS
    # because then we have many more alternative hypotheses than the simple example.

    factor = np.round(factor, 3)
    if type_of_example == "simple" or type_of_example == "Simple":
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
        if indices_partitions == None:
            str_to_print = (
                "If we are running / saving data for the SPP GNSS example, we should provide"
                + ' the parameter "indices_partition"'
            )
            raise Exception(str_to_print)
    else:
        raise Exception("No valid entry of type_of_example: ", type_of_example)

    index = indices_partitions[partition]
    if partition == "P99":
        index = -1

    xx = x.flatten()
    yy = y.flatten()
    zz = z.flatten()

    indices = identifications == index

    xx[~indices] = np.nan * np.ones(sum(~indices))
    yy[~indices] = np.nan * np.ones(sum(~indices))
    zz[~indices] = np.nan * np.ones(sum(~indices))

    xx = xx.reshape(x.shape)
    yy = yy.reshape(x.shape)
    zz = zz.reshape(x.shape)

    if write_grid_to_file:
        if alpha_0 == "default":
            directory = (
                os.path.join(manualPath, "Partitioned_grid", "factor_" + str(factor))
                + "\\"
            )
        else:
            directory = (
                os.path.join(
                    manualPath,
                    "alpha_0=" + str(alpha_0),
                    "Partitioned_grid",
                    "factor_" + str(factor),
                )
                + "\\"
            )

        if not os.path.exists(directory):
            os.makedirs(directory)
        # ## store the grid

        # first clean the directory if there are files in the directory.
        if removeFiles:
            for fname in os.listdir(directory):
                # Check if the file has a .txt extension
                if fname.endswith(".txt"):
                    # Construct the full file path
                    file_path = os.path.join(directory, fname)
                    # Remove the file
                    os.remove(file_path)

        print("Writing partitioned grid")
        print("Directory and file", directory + partition + "_xx.txt")
        np.savetxt(directory + partition + "_xx.txt", xx)
        np.savetxt(directory + partition + "_yy.txt", yy)
        np.savetxt(directory + partition + "_zz.txt", zz)

    if inPlane:
        np.savetxt(directory + "\\a1_vec.txt", a1, delimiter=",")
        np.savetxt(directory + "\\a2_vec.txt", a2, delimiter=",")

    return xx, yy, zz


@timing
def write_data_for_matlab_MHSS(
    x,
    y,
    z,
    identifications,
    partition_list,
    alpha_prime,
    alpha_per_hypt,
    write_grid_to_file=True,
    separate_order_IDS=False,
    inPlane=False,
    factor=None,
    workingDir=None,
    lastOMT=False,
    type_of_alpha="iteration",
    a1=None,
    a2=None,
    manualPath=False,
    angle=None,
    example_type="Simple",
    ARAIM_type="Zhai",
):
    # and returns the partitioned grid xx, yy, zz
    # Inputs
    # all speaks for itself.
    # DIA is the type of partitioning/testing procedure used. IDS, DS_DIA or ordinary DIA
    # write_grid_to_file, if true then the xx,yy,zz arrays are written to a file.
    # separate order IDS, is a boolean. Specify False (default) if we do not want to be able to show P12 and P21
    # If False and write_grid_to_file =True, then P21 and P12 files are identical.
    # if lastOMT = True, then we carry out an additional OMT at the end after removing two outliers (for our example.)
    # Note that this needs to be specified as this data will be written in a different directory.
    # type_of_DS specifies which type of DS should be carried out.
    # a1 and a2 are vectors. If inPlane is true, these vectors are span the plane in which we sample t
    # the boolean removeFiles is added to specify whether files should first be removed from the directory.
    # this is especially important to be set to True when plotting a partitioning inPlane (i.e. when inPlane=True).
    # if manualPath = True, we assign a directory to write data in manually.
    # angle is the rotation of the plane, only relevant for manualPath = True, and for inplane

    factor = np.round(factor, 2)

    DIA_type = "\\MHSS"

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

    if write_grid_to_file:
        if workingDir is None:
            cwd = os.getcwd()
        else:
            cwd = workingDir
        cwd = cwd.replace("\\Code", "")

        startString = cwd + "\\Sim Data" + DIA_type + "\\" + ARAIM_type + "\\"

        if example_type == "SPP_GNSS":
            startString += "SPP_GNSS\\"

        if separate_order_IDS:
            startString = startString + "\\separate_partitionings"
        else:
            startString += "\\no_separate_partitionings"

        if inPlane:
            startString = startString + "\\inPlane"

        if lastOMT:
            startString = startString + "\\lastOMT"

        startString += (
            "_alpha_prime_"
            + str(np.round(alpha_prime, 3))
            + "_alpha_per_hypt_"
            + str(np.round(alpha_per_hypt, 3))
        )

        if factor is not None:
            directory = startString + "\\Partitioned_grid\\factor_" + str(factor) + "\\"
        else:
            directory = startString + "\\Partitioned_grid\\"

        if manualPath == True and inPlane:
            directory = r"C:\Users\bgvannoort\Documents\IDS\Sim Data\IDS\B\separate_partitionings\inPlane\alpha_type_Kok_IDS\Rotate_Along_c1c2_vectors"
            run = angle
            directory = os.path.join(directory, "Partitioned_grid", str(run)) + "\\"

        if not os.path.exists(directory):
            os.makedirs(directory)
        # ## store the grid

        # In this function, we do the looping over the partitions in this function!
        removeFiles = True
        for idx_part, partition in enumerate(partition_list):
            index = indices_partitions[partition]

            xx = x.flatten()
            yy = y.flatten()
            zz = z.flatten()

            indices = identifications == index
            if indices.sum() == 0:  # there exists no data points for this partition
                continue  # go to next partition

            xx[~indices] = np.nan * np.ones(sum(~indices))
            yy[~indices] = np.nan * np.ones(sum(~indices))
            zz[~indices] = np.nan * np.ones(sum(~indices))

            xx = xx.reshape(x.shape)
            yy = yy.reshape(x.shape)
            zz = zz.reshape(x.shape)

            # first clean the directory if there are files in the directory.
            if removeFiles:
                for fname in os.listdir(directory):
                    # Check if the file has a .txt extension
                    if fname.endswith(".txt"):
                        # Construct the full file path
                        file_path = os.path.join(directory, fname)
                        # Remove the file
                        os.remove(file_path)
                removeFiles = False  # only delete the files in the directory once!

            print("Writing partitioned grid")
            print("Directory and file", directory + partition + "_xx.txt")
            np.savetxt(directory + partition + "_xx.txt", xx)
            np.savetxt(directory + partition + "_yy.txt", yy)
            np.savetxt(directory + partition + "_zz.txt", zz)

    if inPlane:
        np.savetxt(directory + "\\a1_vec.txt", a1, delimiter=",")
        np.savetxt(directory + "\\a2_vec.txt", a2, delimiter=",")
    return xx, yy, zz


################ REVERSE IDS PROCEDURES HERE! ####################


def make_partition_string(arr_of_idxes):
    str0 = "P"
    for ix in arr_of_idxes:
        str0 += str(ix + 1) + ","
    str0 = str0[:-1]
    return str0


def get_idx_hypt_RIDS(m, subset, qmax=5):

    if len(subset) == 0:  # We are interested in the whole list of indices
        array_of_idxes = []
        array_of_partitions = []
        counter = 0
        for qs in np.arange(1, qmax + 1):
            for _, el in enumerate(itertools.combinations(np.arange(m), qs)):
                # print('el:', el)
                counter += 1
                sequence = np.array(el)
                array_of_idxes.append(counter)
                array_of_partitions.append(make_partition_string(sequence))

        return array_of_idxes, array_of_partitions
    else:
        array_of_idxes = []
        array_of_partitions = []
        counter = 0
        for qs in np.arange(1, qmax + 1):
            for _, el in enumerate(itertools.combinations(np.arange(m), qs)):
                # print('el:', el)
                counter += 1
                sequence = np.array(el)
                if len(subset) == len(sequence) and np.all(sequence == subset):
                    # print('int his loop')
                    return counter

    raise Exception(
        "In 'get_idx_hypt_RIDS' function, cannot get an index corresponding to S. "
    )


def RIDS_mult_its_type_A(
    t_det,
    alpha_0,
    qmax,
    S,
    ci_list,
    idx_S,  ## this is the index corresponding to H_S
    example_type="simple",
    type_of_alpha="Kok_IDS",
    inPlane=False,
    separate_order_IDS=False,
    bool_start_with_OMT=False,
):
    """
    Perform multiple iterations of Reverse IDS (RIDS) using Type A logic.

    This function carries out type A RIDS as described in the IDS and RIDS report. For a given
    value of t (t_det), it computes to which misclosure space partition (Pis) it belongs.


    Parameters:
    ----------
    t_det : array-like or ndarray
        Array of misclosure vector 'samples' (or can be grid); dimension r times n_samples
    alpha_0 : float
        Level of significance of one w-test.
    qmax : int
        Maximum number of initial outliers present in the initialization of S
    S : list or array
        List of indices corresponding to the (initialized) hypothesis.
    ci_list: list or np.ndarray
        contains the fault vectors ci as columns corresponding to H_S

    example_type : str, optional
        Type of test scenario. Options might include 'simple', 'SPP_GNSS' or others (in the future)
    type_of_alpha : str, optional
        Method to distribute the significance level `alpha_0` of the w-test. Default is 'Kok_IDS', using the B-method
        Other options include, 'Bonferroni', or 'iteration'.
    inPlane : bool, optional
        If True, the misclosure vector t_det consists of a planar region.
    separate_order_IDS : bool, optional
        If True, hypotheses are highlighted based on indices corresponding to the
        order of how they are identified. Not effective in RIDS.
    bool_start_with_OMT: bool, optional, Default False
        If True, the RIDS procedure will start with an OMT. In other words,
        the partitioning P0 will be identical to the standard OMT shaped OMT. If True,
        the alpha_prime (=alpha_check) value taken will be the one that either
            - follows from the B-method of type_of_alpha='Kok_IDS'
            - or follows manually (given in the modify_alpha_prime method) if type_of_alpha='manual'

    Returns:
    -------
    Pis: list or ndarray
        Contains the corresponding list of indices for the identification of the t-vectors in
        t_det.

    Notes:
    -----

    """
    # print("RIDS_type_A: in this function")
    # -------------------------- Problem Setup --------------------------------

    m, n, r, A, alpha, sigma, Qyy, Qyy_inv, Bt, Qtt, Qtt_inv = load_setup_parameters(
        type_of_example=example_type, alpha_method=type_of_alpha, alpha_0=alpha_0
    )
    # -------------------------------------------------------------------------

    # ------------------------ Start of iterative procedures ------------------

    q_S = len(S)

    ## Check if there is enough redundancy at all to exclude qmax outliers.
    if qmax > r - 1:
        raise ValueError(
            "No more than {} outliers can be excluded, qmax={} requested".format(
                r - 1, qmax
            )
        )

    ## Define the global threshold w_alpha0
    w_alpha_0 = scipy.stats.norm.isf(alpha_0 / 2)
    T_OMT_P0 = scipy.stats.chi2.isf(alpha, r)  # threshold for P0

    if bool_start_with_OMT:
        T_OMT = scipy.stats.chi2.isf(alpha, r)
        OMT = np.einsum("ij,jm,mi->i", t_det.T, Qtt_inv, t_det)

        where_larger_TOMT = OMT > T_OMT
        Pis = np.zeros(t_det.shape[1])

        Pis[where_larger_TOMT] = RIDS_mult_its_type_A(
            t_det[:, where_larger_TOMT],
            alpha_0,
            qmax=qmax,
            idx_S=0,
            S=[],
            ci_list=None,
            example_type=example_type,
            type_of_alpha=type_of_alpha,
            bool_start_with_OMT=False,
        )  # we have now carried out an OMT essentially!!
        return Pis

    # print("RIDS_A: q_s = ", q_S)
    # print("RIDS_A: S = ", S, 'type_of_example=', example_type)

    ## Check if S is empty, if so, it is the first function call.
    elif (
        q_S == 0
    ):  ## MAKE SURE THAT ONCE FIRST CALLED, ci_list == None and NOT empytlist!!!
        if ci_list is not None:
            raise IOError("Ci_list should be none if RIDS function is the first call.")

        ## Formulate the ci-list
        # ci_list = []

        # print('RIDS_A: alpha_check=', alpha, 'qmax=', qmax)
        alpha_prime_r_minus_qmax = modify_alpha_prime(
            alpha, alpha_0, m, type_of_alpha, m - n - qmax
        )  # qmax is maximum r-1
        # OMT threshold at first step, i.e. for 2 outliers
        # print('RIDS_A: alpha_prime_r_minus_qmax=', alpha_prime_r_minus_qmax)

        T_OMT = scipy.stats.chi2.isf(alpha_prime_r_minus_qmax, m - n - qmax)
        # print("RIDS_A: T_OMT", T_OMT)
        # print("RIDS_type_A: Right before starting OMTs")
        OMTs = np.zeros((math.comb(m, qmax), t_det.shape[1]))

        nr_unique_sets_S = int(scipy.special.binom(m, qmax))
        arr_order_of_idxes = []

        I_m = np.ascontiguousarray(np.eye(m))
        I_t = np.ascontiguousarray(np.eye(r))
        Bt = np.ascontiguousarray(Bt)
        Qtt_inv = np.ascontiguousarray(Qtt_inv)

        for i, ci_ind in enumerate(itertools.combinations(np.arange(m), qmax)):
            idxes_arr = np.ascontiguousarray(ci_ind)
            arr_order_of_idxes.append(idxes_arr)

            ci = I_m[:, idxes_arr]
            cti = Bt @ ci
            P_perp_cti = I_t - P_mat_invQ(cti, Qtt_inv)

            proj_t = P_perp_cti @ t_det
            OMT = np.einsum("ij,jm,mi->i", proj_t.T, Qtt_inv, proj_t)
            OMTs[i, :] = OMT

        Pis = np.zeros(t_det.shape[1])
        idxes_min_OMT = np.argmin(OMTs, axis=0)
        min_OMT_larger_thresh = np.min(OMTs, axis=0) > T_OMT

        arr_order_of_idxes = np.array(arr_order_of_idxes)
        # arr_order_of_idxes = np.array(combinations)

        # print('arr_order_of_idxes', arr_order_of_idxes)

        for initialization in np.arange(nr_unique_sets_S):
            (idxes_where_initialized,) = np.where(idxes_min_OMT == initialization)
            # print('initialization', initialization)
            t_select = t_det[:, idxes_where_initialized]
            S = arr_order_of_idxes[initialization, :]
            # iteration of w-tests
            w_squared_tests = np.zeros((len(S), t_select.shape[1]))
            S_subsets = []
            OMT2 = np.zeros((len(S), t_select.shape[1]))
            # print("RIDS_A: Initialization Subset S", S)

            # print('RIDS_A, line 2123, S: ', S)
            for idx_S, el in enumerate(S):
                # print("Check if the right indices are then removed")
                # print('RIDS_A:idx_S, el ', idx_S, el)
                # print('RIDS_A: S! = el', S!=el )
                ci = np.eye(m)[:, S[S != el]]
                S_subsets.append(S[S != el])
                # print("RIDS_A: S[S!=el]", S[S!=el])
                cti = Bt @ ci
                P_cti_perp = P_perp(P_mat_invQ(cti, Qtt_inv))
                ct_el_bar = P_cti_perp @ Bt @ np.eye(m)[:, el].reshape(-1, 1)
                P_ct_el_bar_t = P_mat_invQ(ct_el_bar, Qtt_inv) @ t_select

                # compute the projected w-tests
                w_bar_el_squared = np.einsum(
                    "ij,jm,mi->i", P_ct_el_bar_t.T, Qtt_inv, P_ct_el_bar_t
                )
                w_squared_tests[idx_S, :] = w_bar_el_squared

                # Compute the OMT after bias el is removed
                OMT2[idx_S, :] = np.einsum(
                    "ij,jm,mi->i",
                    (P_cti_perp @ t_select).T,
                    Qtt_inv,
                    (P_cti_perp @ t_select),
                )

            idx_min_w_tests = np.argmin(w_squared_tests, axis=0)
            identified_samples = np.min(w_squared_tests, axis=0) > w_alpha_0**2

            # the OMTs after removal of the bias with minimum w-test
            OMT2s_min_wtest = OMT2[idx_min_w_tests, np.arange(len(idx_min_w_tests))]

            # New OMT is with r-qmax+1 dofs
            alpha_prime_new = modify_alpha_prime(
                alpha, alpha_0, m, method=type_of_alpha, df_chi2_new=r - qmax + 1
            )
            # print("RIDS A: alpha_prime_new", alpha_prime_new)

            T_OMT_alpha_prime_new = scipy.stats.chi2.isf(
                alpha_prime_new, df=r - qmax + 1
            )
            # print("RIDS A: alpha_prime_new", alpha_prime_new, 'T_OMT_alpha_prime_new', T_OMT_alpha_prime_new)
            OMT2_where_failed = OMT2s_min_wtest > T_OMT_alpha_prime_new

            ## Undecided region is where min w-test is removed and OMT fails afterwards
            bool_where_OMT2_fails_min_w_smaller = np.logical_and(
                OMT2_where_failed, ~identified_samples
            )
            Pis[idxes_where_initialized[bool_where_OMT2_fails_min_w_smaller]] = -1

            idx_of_set_S = get_idx_hypt_RIDS(m, S, qmax=qmax)
            Pis[idxes_where_initialized[identified_samples]] = idx_of_set_S
            for idx_S_in, S_in in enumerate(S_subsets):
                idx_of_set_S_in = get_idx_hypt_RIDS(m, S_in, qmax=qmax)

                # This is for type C:
                idx_of_samples_to_continue = np.logical_and(
                    ~identified_samples, idx_min_w_tests == idx_S_in
                )
                ## Include an OMT for type A:
                idx_of_samples_to_continue = np.logical_and(
                    idx_of_samples_to_continue, ~OMT2_where_failed
                )
                # print('np.sum(idx_of_samples_to_continue)', np.sum(idx_of_samples_to_continue))

                if qmax == 1:
                    # we cannot provide these values to the function again, because it will fail.
                    # For these samples, the OMT is passing
                    Pis[idxes_where_initialized[idx_of_samples_to_continue]] = 0
                elif np.sum(idx_of_samples_to_continue) == 0:
                    pass
                else:
                    Pis[idxes_where_initialized[idx_of_samples_to_continue]] = (
                        RIDS_mult_its_type_A(
                            t_select[:, idx_of_samples_to_continue],
                            alpha_0,
                            qmax=qmax,
                            idx_S=idx_of_set_S_in,
                            S=S_in,
                            ci_list=np.eye(m)[:, S_in].reshape((m, len(S_in))),
                            example_type=example_type,
                            type_of_alpha=type_of_alpha,
                        )
                    )

        ## Check here at the end if the initialization was sound at all, i.e. if not OMT > chi2_thresh
        Pis[min_OMT_larger_thresh] = -1

        return Pis

    else:  # the function is called from within this function, i.e. for the next iteration.
        Pis = np.zeros(
            t_det.shape[1]
        )  # only t-samples are provided for the corresponding S set.

        if len(ci_list) == 0:
            if len(S) == 0:  # accept the null hypothesis
                return Pis
            else:
                raise Exception(
                    "Ci_list is empty in RIDS function while S is not empty"
                )

        w_squared_tests = np.zeros((len(S), t_det.shape[1]))
        OMT2 = np.zeros((len(S), t_det.shape[1]))
        S_subsets = []

        ## INCLUDE HERE AN IF STATEMENT WHEN S CONTAINS ONLY 1 IDX
        if q_S == 1:
            # print("in q_S=1 statement: ci_list", ci_list)
            ci = ci_list
            cti = Bt @ ci
            # print("Pis before", Pis)
            # it is just the 'ordinary' w-test
            P_ct_i_t = P_mat_invQ(cti, Qtt_inv) @ t_det

            w_i_squared = np.einsum("ij,jm,mi->i", P_ct_i_t.T, Qtt_inv, P_ct_i_t)

            idx_identified_Hi = w_i_squared > w_alpha_0**2
            trueOMT = np.einsum("ij,jm,mi->i", t_det.T, Qtt_inv, t_det)
            # print("RIDS_A: subset S", S)
            # print("RIDS_A: trueOMT_threshold", T_OMT_P0)

            # Pis is initialized as zeros, so no need to 'update' the array for identification of
            # H0 with index of 0.
            # Get idxes of Pomega points/samples
            idx_identified_Homega = np.logical_and(
                ~idx_identified_Hi, trueOMT > T_OMT_P0
            )
            # print('idx_identified_Homega', idx_identified_Homega)

            idx_S_new = get_idx_hypt_RIDS(m, S)
            Pis[idx_identified_Hi] = idx_S_new
            # print('identified_Hi:', idx_identified_Hi)
            Pis[idx_identified_Homega] = -1

            # for i__, j__, k__ in zip(Pis, ~idx_identified_Hi, idx_identified_Homega):
            #     print("Pis[i], ~idx_identifiedi[i], idx_identified_Homega")
            #     print(i__, j__, k__)
            # print('sum(idx_identified_Homega)', np.sum(idx_identified_Homega))
            # sys.exit()
            # The other indices were set to 0 anyway.
            return Pis
        else:
            S = np.array(S).astype(int)
            # print('RIDS_A: S = ', S)

            # print('ci_list', ci_list)
            # print("RIDS_A: S", S)
            for idx_subset, el in enumerate(S):
                # print("Check if the right indices are then removed")
                # print('RIDS_A:idx_subset, el ', idx_subset, el)
                # print('RIDS_A: S! = el', S!=el )
                ci = ci_list[:, S != el]
                S_subsets.append(S[S != el])
                cti = Bt @ ci
                P_cti_perp = P_perp(P_mat_invQ(cti, Qtt_inv))
                ct_el_bar = P_cti_perp @ Bt @ np.eye(m)[:, el].reshape(-1, 1)
                P_ct_el_bar_t = P_mat_invQ(ct_el_bar, Qtt_inv) @ t_det

                w_bar_el_squared = np.einsum(
                    "ij,jm,mi->i", P_ct_el_bar_t.T, Qtt_inv, P_ct_el_bar_t
                )
                w_squared_tests[idx_subset, :] = w_bar_el_squared

                # Compute the OMT after bias el is removed
                OMT2[idx_subset, :] = np.einsum(
                    "ij,jm,mi->i", (P_cti_perp @ t_det).T, Qtt_inv, (P_cti_perp @ t_det)
                )

            idx_min_w_tests = np.argmin(w_squared_tests, axis=0)
            identified_samples = np.min(w_squared_tests, axis=0) > w_alpha_0**2

            OMT2s_min_wtest = OMT2[idx_min_w_tests, np.arange(len(idx_min_w_tests))]

            Pis[identified_samples] = idx_S

            ## Include additional OMT after removal of bias bi
            # New OMT is with r-q_S+1 dofs
            alpha_prime_new = modify_alpha_prime(
                alpha, alpha_0, m, method=type_of_alpha, df_chi2_new=r - q_S + 1
            )
            # print("RIDS A: alpha_prime_new", alpha_prime_new)
            T_OMT_alpha_prime_new = scipy.stats.chi2.isf(
                alpha_prime_new, df=r - q_S + 1
            )
            OMT2_where_failed = OMT2s_min_wtest > T_OMT_alpha_prime_new
            # print("RIDS A: alpha_prime_new", alpha_prime_new, 'T_OMT_alphaprime_new', T_OMT_alpha_prime_new)

            ## Undecided region is where min w-test is removed and OMT fails afterwards
            bool_where_OMT2_fails_min_w_smaller = np.logical_and(
                OMT2_where_failed, ~identified_samples
            )
            Pis[bool_where_OMT2_fails_min_w_smaller] = -1

            for idx_subset, S_subset in enumerate(S_subsets):
                # for type C:
                idx_t_next_iter = np.logical_and(
                    ~identified_samples, idx_min_w_tests == idx_subset
                )
                # for type A include OMT:
                idx_t_next_iter = np.logical_and(idx_t_next_iter, ~OMT2_where_failed)

                # print("RIDS_A: Line 2304 Subset_new", S_subset)

                if (
                    np.sum(idx_t_next_iter) == 0
                ):  # there are no samples that belong to this subset
                    continue
                else:
                    t_next_iter = t_det[:, idx_t_next_iter]
                    idx_S_new = get_idx_hypt_RIDS(m, S_subset, qmax=qmax)
                    Pis[idx_t_next_iter] = RIDS_mult_its_type_A(
                        t_next_iter,
                        alpha_0,
                        qmax=qmax,
                        S=S_subset,
                        ci_list=np.eye(m)[:, S_subset].reshape((m, len(S_subset))),
                        idx_S=idx_S_new,
                        example_type=example_type,
                        type_of_alpha=type_of_alpha,
                    )
        # print('exeting in line 2286')
        # sys.exit()
        return Pis


def RIDS_mult_its_type_B(
    t_det,
    alpha_0,
    qmax,
    S,
    ci_list,
    idx_S,  ## this is the index corresponding to H_S
    example_type="simple",
    type_of_alpha="Kok_IDS",
    inPlane=False,
    separate_order_IDS=False,
    bool_start_with_OMT=False,
):
    """
    Perform multiple iterations of Reverse IDS (RIDS) using Type B logic.

    This function carries out type B RIDS as described in the IDS and RIDS report. For a given
    value of t (t_det), it computes to which misclosure space partition (Pis) it belongs.


    Parameters:
    ----------
    t_det : array-like or ndarray
        Array of misclosure vector 'samples' (or can be grid); dimension r times n_samples
    alpha_0 : float
        Level of significance of one w-test. -- only relevant if type_of_alpha=Kok_IDS (B-method)
        Otherwise, for type B RIDS, it does not do anything.
    qmax : int
        Maximum number of initial outliers present in the initialization of S
    S : list or array
        List of indices corresponding to the (initialized) hypothesis.
    ci_list: list or np.ndarray
        contains the fault vectors corresponding to H_S as columns
    example_type : str, optional
        Type of test scenario. Options might include 'simple', 'SPP_GNSS' or others (in the future)
    type_of_alpha : str, optional
        Method to distribute the significance level `alpha_0` of the w-test. Default is 'Kok_IDS', using the B-method
        Other options include, 'Bonferroni', or 'iteration'.
    inPlane : bool, optional
        If True, the misclosure vector t_det consists of a planar region.
    separate_order_IDS : bool, optional
        If True, hypotheses are highlighted based on indices corresponding to the
        order of how they are identified. Not effective in RIDS.
    bool_start_with_OMT: bool, optional, Default False
        If True, the RIDS procedure will start with an OMT. In other words,
        the partitioning P0 will be identical to the standard OMT shaped OMT. If True,
        the alpha_prime (=alpha_check) value taken will be the one that either
            - follows from the B-method of type_of_alpha='Kok_IDS'
            - or follows manually (given in the modify_alpha_prime method) if type_of_alpha='manual'

    Returns:
    -------
    Pis: list or ndarray
        Contains the corresponding list of indices for the identification of the t-vectors in
        t_det.

    Notes:
    -----

    """
    # -------------------------- Problem Setup --------------------------------

    m, n, r, A, alpha, sigma, Qyy, Qyy_inv, Bt, Qtt, Qtt_inv = load_setup_parameters(
        type_of_example=example_type, alpha_method=type_of_alpha, alpha_0=alpha_0
    )

    # -------------------------------------------------------------------------

    # ------------------------ Start of iterative procedures ------------------

    q_S = len(S)

    ## Check if there is enough redundancy at all to exclude qmax outliers.
    if qmax > r - 1:
        raise ValueError(
            "No more than {} outliers can be excluded, qmax={} requested".format(
                r - 1, qmax
            )
        )

    ## Define the global threshold w_alpha0
    w_alpha_0 = scipy.stats.norm.isf(alpha_0 / 2)
    T_OMT_P0 = scipy.stats.chi2.isf(alpha, r)  # threshold for P0

    if bool_start_with_OMT:
        T_OMT = scipy.stats.chi2.isf(alpha, df=r)
        OMT = np.einsum("ij,jm,mi->i", t_det.T, Qtt_inv, t_det)

        where_larger_TOMT = OMT > T_OMT
        Pis = np.zeros(t_det.shape[1])

        Pis[where_larger_TOMT] = RIDS_mult_its_type_B(
            t_det[:, where_larger_TOMT],
            alpha_0,
            qmax=qmax,
            idx_S=0,
            ci_list=None,
            example_type=example_type,
            type_of_alpha=type_of_alpha,
            bool_start_with_OMT=False,
        )  # we have now carried out an OMT essentially
        return Pis

    ## Check if S is empty, if so, it is the first function call.
    elif (
        q_S == 0
    ):  ## MAKE SURE THAT ONCE FIRST CALLED, ci_list == None and NOT empytlist!!!
        if ci_list is not None:
            raise IOError("Ci_list should be none if RIDS function is the first call.")

        ## Formulate the ci-list
        ci_list = []
        indices_of_ci = []  # contains the indices of the hypotheses in string format

        alpha_prime_r_minus_qmax = modify_alpha_prime(
            alpha, alpha_0, m, type_of_alpha, m - n - qmax
        )  # qmax is maximum r-1
        # OMT threshold at first step, i.e. for 2 outliers
        T_OMT = scipy.stats.chi2.isf(alpha_prime_r_minus_qmax, m - n - qmax)
        OMTs = np.zeros((math.comb(m, qmax), t_det.shape[1]))

        nr_unique_sets_S = int(scipy.special.binom(m, qmax))
        arr_order_of_idxes = []

        I_m = np.ascontiguousarray(np.eye(m))
        I_t = np.ascontiguousarray(np.eye(r))
        Bt = np.ascontiguousarray(Bt)
        Qtt_inv = np.ascontiguousarray(Qtt_inv)

        # print('OMTs.shape', OMTs.shape)
        # print('nr_of_unique_sets', nr_unique_sets_S)
        for i, ci_ind in enumerate(itertools.combinations(np.arange(m), qmax)):
            idxes_arr = np.ascontiguousarray(ci_ind)
            arr_order_of_idxes.append(idxes_arr)

            ci = I_m[:, idxes_arr]
            cti = Bt @ ci
            P_perp_cti = I_t - P_mat_invQ(cti, Qtt_inv)

            proj_t = P_perp_cti @ t_det
            OMT = np.einsum("ij,jm,mi->i", proj_t.T, Qtt_inv, proj_t)
            OMTs[i, :] = OMT
            ci_list.append(ci)

        Pis = np.zeros(t_det.shape[1])
        idxes_min_OMT = np.argmin(OMTs, axis=0)
        min_OMT_larger_thresh = np.min(OMTs, axis=0) > T_OMT

        arr_order_of_idxes = np.array(arr_order_of_idxes)

        # print('arr_order_of_idxes', arr_order_of_idxes)

        for initialization in np.arange(nr_unique_sets_S):
            where_initialized = idxes_min_OMT == initialization
            (idxes_where_initialized,) = np.where(where_initialized)
            # print('initialization', initialization)
            t_select = t_det[:, where_initialized]
            S = arr_order_of_idxes[initialization, :]
            # iteration of w-tests
            w_squared_tests = np.zeros((len(S), t_select.shape[1]))
            S_subsets = []
            OMT2 = np.zeros((len(S), t_select.shape[1]))

            for idx_S, el in enumerate(S):
                ci = np.eye(m)[:, S[S != el]]
                S_subsets.append(S[S != el])
                cti = Bt @ ci
                P_cti_perp = P_perp(P_mat(cti, Qtt))
                ct_el_bar = P_cti_perp @ Bt @ np.eye(m)[:, el].reshape(-1, 1)
                P_ct_el_bar_t = P_mat(ct_el_bar, Qtt) @ t_select

                # compute the projected w-tests
                w_bar_el_squared = np.einsum(
                    "ij,jm,mi->i", P_ct_el_bar_t.T, Qtt_inv, P_ct_el_bar_t
                )
                w_squared_tests[idx_S, :] = w_bar_el_squared

                # Compute the OMT after bias el is removed
                OMT2[idx_S, :] = np.einsum(
                    "ij,jm,mi->i",
                    (P_cti_perp @ t_select).T,
                    Qtt_inv,
                    (P_cti_perp @ t_select),
                )

            idx_min_w_tests = np.argmin(w_squared_tests, axis=0)

            # The OMTs after removal of the bias with minimum w-test
            OMT2s_min_wtest = OMT2[idx_min_w_tests, np.arange(len(idx_min_w_tests))]

            # New OMT is with r-qmax+1 dofs
            alpha_prime_new = modify_alpha_prime(
                alpha, alpha_0, m, method=type_of_alpha, df_chi2_new=r - qmax + 1
            )
            T_OMT_alpha_prime_new = scipy.stats.chi2.isf(
                alpha_prime_new, df=r - qmax + 1
            )
            # For type B RIDS, identify the current S if OMT fails after removal of min w-test.
            identified_samples = OMT2s_min_wtest > T_OMT_alpha_prime_new

            idx_of_set_S = get_idx_hypt_RIDS(m, S, qmax=qmax)
            Pis[idxes_where_initialized[identified_samples]] = idx_of_set_S
            for idx_S, S_in in enumerate(S_subsets):
                idx_of_set_S_in = get_idx_hypt_RIDS(m, S_in, qmax=qmax)

                # print('np.logical_and(~identified_samples, idx_min_w_tests==idx_S)',
                #       np.sum(np.logical_and(~identified_samples, idx_min_w_tests==idx_S)))

                # print('Length of output:', RIDS_mult_its_type_C(t_select[:, idx_min_w_tests==idx_S], alpha_0, qmax = qmax,
                #                      idx_S = idx_of_set_S_in,
                #                      S=S_in, ci_list=np.eye(m)[:, S_in].reshape((m,len(S_in)))))
                # This is for type B: min w-test removed, and OMT passes afterwards
                idx_of_samples_to_continue = np.logical_and(
                    ~identified_samples, idx_min_w_tests == idx_S
                )

                if qmax == 1:
                    Pis[idxes_where_initialized[idx_of_samples_to_continue]] = 0
                elif np.sum(idx_of_samples_to_continue) == 0:
                    pass
                else:
                    # print("LIne 2541")
                    # print("S_subset, qmax, ", S_in, qmax)
                    Pis[idxes_where_initialized[idx_of_samples_to_continue]] = (
                        RIDS_mult_its_type_B(
                            t_select[:, idx_of_samples_to_continue],
                            alpha_0,
                            qmax=qmax,
                            idx_S=idx_of_set_S_in,
                            S=S_in,
                            ci_list=np.eye(m)[:, S_in].reshape((m, len(S_in))),
                            example_type=example_type,
                            type_of_alpha=type_of_alpha,
                        )
                    )

        ## Check here at the end if the initialization was sound at all, i.e. if not OMT > chi2_thresh
        Pis[min_OMT_larger_thresh] = -1

        return Pis

    else:  # the function is called from within this function, i.e. for the next iteration.
        Pis = np.zeros(
            t_det.shape[1]
        )  # only t-samples are provided for the corresponding S set.

        if len(ci_list) == 0:
            if len(S) == 0:  # accept the null hypothesis
                return Pis
            else:
                raise Exception(
                    "Ci_list is empty in RIDS function while S is not empty"
                )

        w_squared_tests = np.zeros((len(S), t_det.shape[1]))
        OMT2 = np.zeros((len(S), t_det.shape[1]))
        S_subsets = []

        if q_S == 1:
            ci = ci_list
            cti = Bt @ ci

            # it is just the 'ordinary' w-test
            P_ct_i_t = P_mat(cti, Qtt) @ t_det

            trueOMT = np.einsum("ij,jm,mi->i", t_det.T, Qtt_inv, t_det)
            # Pis is initialized as zeros, so no need to 'update' the array for identification of
            # H0 with index of 0.
            # Get idxes of Pomega points/samples
            idx_identified_Hi = trueOMT > T_OMT_P0

            Pis[idx_identified_Hi] = idx_S

        else:
            S = np.array(S)
            for idx_subset, el in enumerate(S):
                ci = ci_list[:, S != el]
                S_subsets.append(S[S != el])
                cti = Bt @ ci
                P_cti_perp = P_perp(P_mat(cti, Qtt))
                ct_el_bar = P_cti_perp @ Bt @ np.eye(m)[:, el].reshape(-1, 1)
                P_ct_el_bar_t = P_mat(ct_el_bar, Qtt) @ t_det

                w_bar_el_squared = np.einsum(
                    "ij,jm,mi->i", P_ct_el_bar_t.T, Qtt_inv, P_ct_el_bar_t
                )
                w_squared_tests[idx_subset, :] = w_bar_el_squared

                # Compute the OMT after bias el is removed
                OMT2[idx_subset, :] = np.einsum(
                    "ij,jm,mi->i", (P_cti_perp @ t_det).T, Qtt_inv, (P_cti_perp @ t_det)
                )

            idx_min_w_tests = np.argmin(w_squared_tests, axis=0)

            # The OMTs after removal of the bias with minimum w-test
            OMT2s_min_wtest = OMT2[idx_min_w_tests, np.arange(len(idx_min_w_tests))]

            ## Include additional OMT after removal of bias bi
            # New OMT is with r-q_S+1 dofs
            alpha_prime_new = modify_alpha_prime(
                alpha, alpha_0, m, method=type_of_alpha, df_chi2_new=r - q_S + 1
            )
            T_OMT_alpha_prime_new = scipy.stats.chi2.isf(
                alpha_prime_new, df=r - q_S + 1
            )
            # For type B RIDS, identify the current S if OMT fails after removal of min w-test.
            identified_samples = OMT2s_min_wtest > T_OMT_alpha_prime_new

            Pis[identified_samples] = idx_S

            for idx_subset, S_subset in enumerate(S_subsets):
                # for type C:
                idx_t_next_iter = np.logical_and(
                    ~identified_samples, idx_min_w_tests == idx_subset
                )

                if (
                    np.sum(idx_t_next_iter) == 0
                ):  # there are no samples that belong to this subset
                    continue
                else:
                    t_next_iter = t_det[:, idx_t_next_iter]
                    idx_S_new = get_idx_hypt_RIDS(m, S_subset, qmax=qmax)

                    print("LIne 2631")
                    print("S_subset, qmax, ", S_subset, idx_S_new)
                    Pis[idx_t_next_iter] = RIDS_mult_its_type_B(
                        t_next_iter,
                        alpha_0,
                        qmax=qmax,
                        S=S_subset,
                        ci_list=np.eye(m)[:, S_subset].reshape((m, len(S_subset))),
                        idx_S=idx_S_new,
                        example_type=example_type,
                        type_of_alpha=type_of_alpha,
                    )
        return Pis


def RIDS_mult_its_type_C(
    t_det,
    alpha_0,
    qmax,
    S,
    ci_list,
    idx_S,  ## this is the index corresponding to H_S
    example_type="simple",
    type_of_alpha="Kok_IDS",
    inPlane=False,
    separate_order_IDS=False,
    bool_start_with_OMT=False,
):
    """
    Perform multiple iterations of Reverse IDS (RIDS) using Type C logic.

    This function carries out type C RIDS as described in the IDS and RIDS report. For a given
    value of t (t_det), it computes to which misclosure space partition (Pis) it belongs.


    Parameters:
    ----------
    t_det : array-like or ndarray
        Array of misclosure vector 'samples' (or can be grid); dimension r times n_samples
    alpha_0 : float
        Level of significance of one w-test.
    qmax : int
        Maximum number of initial outliers present in the initialization of S
    S : list or array
        List of indices corresponding to the (initialized) hypothesis.
    example_type : str, optional
        Type of test scenario. Options might include 'simple', 'SPP_GNSS' or others (in the future)
    type_of_alpha : str, optional
        Method to distribute the significance level `alpha_0` of the w-test. Default is 'Kok_IDS', using the B-method
        Other options include, 'Bonferroni', or 'iteration'.
    inPlane : bool, optional
        If True, the misclosure vector t_det consists of a planar region.
    separate_order_IDS : bool, optional
        If True, hypotheses are highlighted based on indices corresponding to the
        order of how they are identified. Not effective in RIDS.
    bool_start_with_OMT: bool, optional, Default False
        If True, the RIDS procedure will start with an OMT. In other words,
        the partitioning P0 will be identical to the standard OMT shaped OMT. If True,
        the alpha_prime (=alpha_check) value taken will be the one that either
            - follows from the B-method of type_of_alpha='Kok_IDS'
            - or follows manually (given in the modify_alpha_prime method) if type_of_alpha='manual'



    Returns:
    -------
    Pis: list or ndarray
        Contains the corresponding list of indices for the identification of the t-vectors in
        t_det.

    Notes:
    -----

    """
    # -------------------------- Problem Setup --------------------------------

    m, n, r, A, alpha, sigma, Qyy, Qyy_inv, Bt, Qtt, Qtt_inv = load_setup_parameters(
        type_of_example=example_type, alpha_method=type_of_alpha, alpha_0=alpha_0
    )

    # -------------------------------------------------------------------------

    # ------------------------ Start of iterative procedures ------------------

    q_S = len(S)

    ## Check if there is enough redundancy at all to exclude qmax outliers.
    if qmax > r - 1:
        raise ValueError(
            "No more than {} outliers can be excluded, qmax={} requested".format(
                r - 1, qmax
            )
        )

    ## Define the global threshold w_alpha0
    w_alpha_0 = scipy.stats.norm.isf(alpha_0 / 2)

    if bool_start_with_OMT:
        T_OMT = scipy.stats.chi2.isf(alpha, df=r)
        OMT = np.einsum("ij,jm,mi->i", t_det.T, Qtt_inv, t_det)

        where_larger_TOMT = OMT > T_OMT
        Pis = np.zeros(t_det.shape[1])

        Pis[where_larger_TOMT] = RIDS_mult_its_type_C(
            t_det[:, where_larger_TOMT],
            alpha_0,
            qmax=qmax,
            S=[],
            ci_list=None,
            idx_S=0,
            example_type=example_type,
            type_of_alpha=type_of_alpha,
            bool_start_with_OMT=False,
        )
        return Pis

    ## Check if S is empty, if so, it is the first function call.
    elif (
        q_S == 0
    ):  ## MAKE SURE THAT ONCE FIRST CALLED, ci_list == None and NOT empytlist!!!
        if ci_list is not None:
            raise IOError("Ci_list should be none if RIDS function is the first call.")

        ## Formulate the ci-list
        ci_list = []

        alpha_prime_r_minus_qmax = modify_alpha_prime(
            alpha, alpha_0, m, type_of_alpha, m - n - qmax
        )  # qmax is maximum r-1
        # OMT threshold at first step, i.e. for 2 outliers
        T_OMT = scipy.stats.chi2.isf(alpha_prime_r_minus_qmax, m - n - qmax)
        OMTs = np.zeros((math.comb(m, qmax), t_det.shape[1]))

        nr_unique_sets_S = int(scipy.special.binom(m, qmax))
        arr_order_of_idxes = []

        I_m = np.ascontiguousarray(np.eye(m))
        I_t = np.ascontiguousarray(np.eye(r))
        Bt = np.ascontiguousarray(Bt)
        Qtt_inv = np.ascontiguousarray(Qtt_inv)
        # print('OMTs.shape', OMTs.shape)
        # print('nr_of_unique_sets', nr_unique_sets_S)
        for i, ci_ind in enumerate(itertools.combinations(np.arange(m), qmax)):
            idxes_arr = np.ascontiguousarray(np.array(ci_ind))
            arr_order_of_idxes.append(idxes_arr)

            ci = I_m[:, idxes_arr]
            cti = Bt @ ci
            P_perp_cti = I_t - P_mat_invQ(cti, Qtt_inv)

            proj_t = P_perp_cti @ t_det
            OMT = np.einsum("ij,jm,mi->i", proj_t.T, Qtt_inv, proj_t)
            OMTs[i, :] = OMT
            ci_list.append(ci)

        Pis = np.zeros(t_det.shape[1])
        idxes_min_OMT = np.argmin(OMTs, axis=0)
        min_OMT_larger_thresh = np.min(OMTs, axis=0) > T_OMT

        arr_order_of_idxes = np.ascontiguousarray(arr_order_of_idxes)

        # print('arr_order_of_idxes', arr_order_of_idxes)

        for initialization in np.arange(nr_unique_sets_S):
            where_initialized = idxes_min_OMT == initialization
            (idxes_where_initialized,) = np.where(where_initialized)
            # print('initialization', initialization)
            t_select = t_det[:, where_initialized]
            S = arr_order_of_idxes[initialization, :]
            # iteration of w-tests
            w_squared_tests = np.zeros((len(S), t_select.shape[1]))
            S_subsets = []

            ## INCLUDE HERE AN IF STATEMENT WHEN S CONTAINS ONLY 1 IDX --> cannot happen here..?
            ## AS THE ROUTINES ARE OTHERWISE FAULT

            for idx_S, el in enumerate(S):
                ci = np.eye(m)[:, S[S != el]]
                S_subsets.append(S[S != el])
                cti = Bt @ ci
                P_cti_perp = P_perp(P_mat(cti, Qtt))
                ct_el_bar = P_cti_perp @ Bt @ np.eye(m)[:, el].reshape(-1, 1)
                P_ct_el_bar_t = P_mat(ct_el_bar, Qtt) @ t_select

                w_bar_el_squared = np.einsum(
                    "ij,jm,mi->i", P_ct_el_bar_t.T, Qtt_inv, P_ct_el_bar_t
                )
                w_squared_tests[idx_S, :] = w_bar_el_squared

            idx_min_w_tests = np.argmin(w_squared_tests, axis=0)
            identified_samples = np.min(w_squared_tests, axis=0) > w_alpha_0**2

            idx_of_set_S = get_idx_hypt_RIDS(m, S, qmax=qmax)
            Pis[idxes_where_initialized[identified_samples]] = idx_of_set_S
            for idx_S, S_in in enumerate(S_subsets):
                idx_of_set_S_in = get_idx_hypt_RIDS(m, S_in, qmax=qmax)
                # print('np.logical_and(~identified_samples, idx_min_w_tests==idx_S)',
                #       np.sum(np.logical_and(~identified_samples, idx_min_w_tests==idx_S)))

                # print('Length of output:', RIDS_mult_its_type_C(t_select[:, idx_min_w_tests==idx_S], alpha_0, qmax = qmax,
                #                      idx_S = idx_of_set_S_in,
                #                      S=S_in, ci_list=np.eye(m)[:, S_in].reshape((m,len(S_in)))))
                idx_of_samples_to_continue = np.logical_and(
                    ~identified_samples, idx_min_w_tests == idx_S
                )

                if qmax == 1:
                    Pis[idxes_where_initialized[idx_of_samples_to_continue]] = 0
                elif np.sum(idx_of_samples_to_continue) == 0:
                    pass
                else:
                    Pis[idxes_where_initialized[idx_of_samples_to_continue]] = (
                        RIDS_mult_its_type_C(
                            t_select[:, idx_of_samples_to_continue],
                            alpha_0,
                            qmax=qmax,
                            idx_S=idx_of_set_S_in,
                            S=S_in,
                            ci_list=np.eye(m)[:, S_in].reshape((m, len(S_in))),
                            example_type=example_type,
                            type_of_alpha=type_of_alpha,
                        )
                    )

        ## Check here at the end if the initialization was sound at all, i.e. if not OMT > chi2_thresh
        Pis[min_OMT_larger_thresh] = -1

        return Pis
    else:  # the function is called from within this function, i.e. for the next iteration.
        Pis = np.zeros(
            t_det.shape[1]
        )  # only t-samples are provided for the corresponding S set.

        if len(ci_list) == 0:
            if len(S) == 0:  # accept the null hypothesis
                return Pis
            else:
                raise Exception(
                    "Ci_list is empty in RIDS function while S is not empty"
                )

        w_squared_tests = np.zeros((len(S), t_det.shape[1]))
        S_subsets = []

        ## INCLUDE HERE AN IF STATEMENT WHEN S CONTAINS ONLY 1 IDX
        if q_S == 1:
            ci = ci_list
            cti = Bt @ ci

            # it is just the 'ordinary' w-test
            P_ct_i_t = P_mat(cti, Qtt) @ t_det

            w_i_squared = np.einsum("ij,jm,mi->i", P_ct_i_t.T, Qtt_inv, P_ct_i_t)

            idx_identified_Hi = w_i_squared > w_alpha_0**2

            idx_S_new = get_idx_hypt_RIDS(m, S)
            Pis[idx_identified_Hi] = idx_S_new
        else:
            S = np.array(S)
            for idx_subset, el in enumerate(S):
                ci = ci_list[:, S != el]
                S_subsets.append(S[S != el])
                cti = Bt @ ci
                P_cti_perp = P_perp(P_mat(cti, Qtt))
                ct_el_bar = P_cti_perp @ Bt @ np.eye(m)[:, el].reshape(-1, 1)
                P_ct_el_bar_t = P_mat(ct_el_bar, Qtt) @ t_det

                w_bar_el_squared = np.einsum(
                    "ij,jm,mi->i", P_ct_el_bar_t.T, Qtt_inv, P_ct_el_bar_t
                )
                w_squared_tests[idx_subset, :] = w_bar_el_squared

            idx_min_w_tests = np.argmin(w_squared_tests, axis=0)
            identified_samples = np.min(w_squared_tests, axis=0) > w_alpha_0**2

            Pis[identified_samples] = idx_S
            for idx_subset, S_subset in enumerate(S_subsets):
                idx_t_next_iter = np.logical_and(
                    ~identified_samples, idx_min_w_tests == idx_subset
                )
                if (
                    np.sum(idx_t_next_iter) == 0
                ):  # there are no samples that belong to this subset
                    continue
                else:
                    t_next_iter = t_det[:, idx_t_next_iter]
                    idx_S_new = get_idx_hypt_RIDS(m, S_subset, qmax=qmax)
                    Pis[idx_t_next_iter] = RIDS_mult_its_type_C(
                        t_next_iter,
                        alpha_0,
                        qmax=qmax,
                        S=S_subset,
                        ci_list=np.eye(m)[:, S_subset].reshape((m, len(S_subset))),
                        idx_S=idx_S_new,
                        example_type=example_type,
                        type_of_alpha=type_of_alpha,
                    )
        return Pis
