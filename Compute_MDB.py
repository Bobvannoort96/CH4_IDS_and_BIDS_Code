"""
Description of code:
    Compute the MDBs for a given method, IDS type
    and example type
Author:
    Bob van Noort
Date:
    27 April

"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
from Functions import *
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox, TransformedBbox
import matplotlib.lines as mlines


def compute_P_CD(
    b_mag,
    cti_model,
    d_vector,
    Qtt,
    N_samples,
    compute_Pis,
    B_T,
    alpha_prime,
    type_of_testing,
    type_of_DS,
    type_of_example,
    qmax,
):
    """
    Compute P_CD for a given bias magnitude.
    """
    t_mean = (b_mag * cti_model @ d_vector).flatten()
    t_samples = scipy.stats.multivariate_normal.rvs(
        mean=t_mean, cov=Qtt, size=N_samples
    ).T

    Pis = compute_Pis(
        t_samples,
        Qtt,
        B_T,
        alpha_prime,
        type_of_testing,
        type_of_DS=type_of_DS,
        type_of_example=type_of_example,
        qmax=qmax,
    )

    # Compute the probability of correct detection, i.e. not accepting H0
    PCD_computed = np.sum(Pis != 0) / N_samples
    return PCD_computed


def find_bias_b_MDB(
    PCD_goal,
    b_min,
    b_max,
    tol,
    max_iter,
    cti_model,
    d_vector,
    Qtt,
    N_samples,
    compute_Pis,
    B_T,
    alpha_prime,
    type_of_testing,
    type_of_DS,
    type_of_example,
    qmax,
):
    """
    Bisection search for b_mag such that P_CD ≈ PCD_goal.
    """

    for _ in range(max_iter):
        b_mid = 0.5 * (b_min + b_max)

        PCD_mid = compute_P_CD(
            b_mid,
            cti_model,
            d_vector,
            Qtt,
            N_samples,
            compute_Pis,
            B_T,
            alpha_prime,
            type_of_testing,
            type_of_DS,
            type_of_example,
            qmax,
        )

        diff = PCD_mid - PCD_goal

        if abs(diff) < tol:
            return b_mid, PCD_mid

        PCD_min = compute_P_CD(
            b_min,
            cti_model,
            d_vector,
            Qtt,
            N_samples,
            compute_Pis,
            B_T,
            alpha_prime,
            type_of_testing,
            type_of_DS,
            type_of_example,
            qmax,
        )

        # Check which side to keep
        if (PCD_min - PCD_goal) * diff < 0:
            b_max = b_mid
        else:
            b_min = b_mid

        print("b_min, b_mid, b_max", b_min, b_mid, b_max)
        print("PCI_mid, PCI_min", PCD_mid, PCD_min)

    b_mag = 0.5 * (b_min + b_max)
    PCD_bmag = compute_P_CD(
        b_mag,
        cti_model,
        d_vector,
        Qtt,
        N_samples,
        compute_Pis,
        B_T,
        alpha_prime,
        type_of_testing,
        type_of_DS,
        type_of_example,
    )
    return b_mag, PCD_bmag


# Code that functions well with the supercomputer.
# It takes d_i as input, ranging from 0 to 999. So, the supercomputer
# can be run with sbatch --array=0-999
def compute_MDB_for_d_i(
    d_i,
    type_of_testing,
    type_of_DS,
    type_of_example,
    alpha_0,
    alpha_prime,
    type_of_alpha,
    for_partition,
    qmax,
    PCD_goal,
):
    """
    Computes MDB and PCD_computed for one direction index d_i.

    Parameters
    ----------
    d_i : int
        Index of direction vector to process.
    type_of_testing : str
    type_of_DS : str
    type_of_example : str
    alpha_0 : float
    alpha_prime : float
    type_of_alpha : str
    for_partition : str
    qmax: int # specifies the total / max nr of outliers to detect for.
    PCD_goal: float, specifies the desired probability of correct detection.

    Returns
    -------
    MIB_vector : ndarray
        Computed MIB (bias vector).
    PCI_computed : float
        Probability of correct identification achieved.
    """

    # Load problem setup
    if type_of_example == "simple":
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
        m, n, r, A, alpha, sigma, Qyy, Qyy_inv, B_T, Qtt, Qtt_inv = setup(
            alpha_0=alpha_0, alpha_method=type_of_alpha, alpha_prime=alpha_prime
        )
    elif type_of_example == "SPP_GNSS":
        indices_partitions = load_indices_partitions_GNSS_ex(False)
        m, n, r, A, alpha, sigma, Qyy, Qyy_inv, B_T, Qtt, Qtt_inv = (
            setup_SPP_GNSS_example(
                alpha_0=alpha_0, alpha_method=type_of_alpha, alpha_prime=alpha_prime
            )
        )
    else:
        raise ValueError("Unknown type_of_example: {}".format(type_of_example))

    q_i = len(for_partition.replace("P", ""))
    idx_under_hypt = indices_partitions[for_partition]
    n_points = 1000
    print("idx_under_hypt", idx_under_hypt)

    if q_i == 1:
        d_vectors = np.array([[1]])
        idxes = int(for_partition.replace("P", ""))
        ci_model = np.eye(m)[:, idxes - 1].reshape(-1, 1)
    elif q_i == 2:
        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        d_vectors = np.stack((np.cos(theta), np.sin(theta)), axis=0)
        idxes = np.array([int(zeta) for zeta in for_partition.replace("P", "")])
        ci_model = np.eye(m)[:, idxes - 1]
    elif q_i == 3:
        _, _, _, d_vectors = generate_t_grid(31, 32)
        idxes = np.array([int(zeta) for zeta in for_partition.replace("P", "")])
        ci_model = np.eye(m)[:, idxes - 1]
    else:
        raise ValueError("Partition dimension not supported.")

    if d_i >= d_vectors.shape[1]:
        raise ValueError(
            f"Requested d_i={d_i}, but only {d_vectors.shape[1]} directions available!"
        )

    d_vector = d_vectors[:, d_i].reshape(-1, 1)

    # Start simulation
    cti_model = B_T @ ci_model
    N_samples = int(1e6)

    # Initialize
    a_, b_, c_ = 0, 10, 3  # update these parameter if we cannot find a good mdb
    MDB_range = np.arange(a_, b_) * c_
    PCD_initialized = np.zeros(len(MDB_range))
    while True:
        for i_mdb, Mdb in enumerate(MDB_range):
            t_mean = Mdb * cti_model @ d_vector
            t_samples = scipy.stats.multivariate_normal.rvs(
                mean=t_mean.flatten(), cov=Qtt, size=(N_samples,)
            ).T
            Pis = compute_Pis(
                t_samples,
                Qtt,
                B_T,
                alpha_prime,
                type_of_testing,
                type_of_DS=type_of_DS,
                type_of_example=type_of_example,
                qmax=qmax,
            )
            PCD_initialized[i_mdb] = np.sum(Pis != 0) / N_samples
        print("PCD_initialized", PCD_initialized)
        where_larger_than_PCD_goal = PCD_initialized > PCD_goal
        if np.sum(where_larger_than_PCD_goal) == 0:
            # Cannot find good MDB-range
            a_ = np.copy(b_)
            b_ += 10
        else:
            break

    b_max = MDB_range[where_larger_than_PCD_goal][0]
    b_min = MDB_range[~where_larger_than_PCD_goal][-1]
    print("Where_larger_PCD", where_larger_than_PCD_goal)
    print("bmin, bmax", b_min, b_max)
    # if b_mag > 150:
    #     d_b_mag = 0.5

    b_mag, PCD_computed = find_bias_b_MDB(
        PCD_goal,
        b_min=b_min,
        b_max=b_max,
        tol=1e-3,
        max_iter=50,
        cti_model=cti_model,
        d_vector=d_vector,
        Qtt=Qtt,
        N_samples=N_samples,
        compute_Pis=compute_Pis,
        B_T=B_T,
        alpha_prime=alpha_prime,
        type_of_testing=type_of_testing,
        type_of_DS=type_of_DS,
        type_of_example=type_of_example,
        qmax=qmax,
    )

    MDB_vector = b_mag * d_vector.flatten()
    return MDB_vector, PCD_computed


def plot_MDB_surface(
    type_of_testing,
    type_of_DS,
    type_of_example,
    alpha_0,
    alpha_prime,
    type_of_alpha,
    for_partition,
    PCD_goal,
    q_i,
    ax=None,
    linestyle="-",
    legend_label=None,
    color_lines=None,
):

    print(
        "Type_of_testing, type_of_DS, PCD_goal", type_of_testing, type_of_DS, PCD_goal
    )
    ## Load the original d_vectors array

    if type_of_testing == "classical DIA":
        resDir = os.path.join(
            r"C:\Users\bgvannoort\Documents\IDS\Results\MDBs",
            type_of_example,
            for_partition.replace("P", "H"),
            type_of_testing,
            "PCD_goal={}".format(PCD_goal),
        )
        if legend_label is None:
            legend_label = "{}".format(type_of_testing)
    else:
        resDir = os.path.join(
            r"C:\Users\bgvannoort\Documents\IDS\Results\MDBs",
            type_of_example,
            for_partition.replace("P", "H"),
            "PCD_goal={}".format(PCD_goal),
            type_of_testing,
            type_of_DS,
        )
        if legend_label is None:
            legend_label = "{} {}".format(type_of_testing, type_of_DS)

    print("resdir", resDir)
    # Load MIB data
    MDBs = np.loadtxt(os.path.join(resDir, "MDB_vectors.txt"))
    print("Mdbs.shape", MDBs.shape)
    # Load PCI_computed_data
    PCD_computed = np.loadtxt(os.path.join(resDir, "PCD_computed.txt"))
    print("PCD_computed.shape", PCD_computed.shape)

    for idx_PCD in range(PCD_computed.shape[0]):
        di, PCD_comp = PCD_computed[idx_PCD, :]
        # print(PCI_comp)
        if PCD_comp is not np.nan:
            if np.abs(PCD_comp - PCD_goal) > 0.01:
                print("At di = {}, the computed PCI = {}".format(di, PCD_comp))

    orders_PCD = PCD_computed[:, 0].astype(int)
    order_MDB = MDBs[:, 0].astype(int)
    # sys.exit()

    argsorted_PCD = np.argsort(orders_PCD)
    argsorted_MDB = np.argsort(order_MDB)

    PCD_computed = PCD_computed[
        argsorted_PCD, 1
    ]  # only last column really corresponds to PCI

    MDBs_ordered = MDBs[argsorted_MDB, 1:]

    if len(MDBs_ordered) != 1000:
        raise Exception("Length of Array is not 1000")

    if q_i == 2:
        if ax is not None:
            pass
        else:
            fig, ax = plt.subplots()

        # if type_of_DS == 'B' and PCI_goal == 0.8 and type_of_testing == 'R_IDS':
        #     ## For positive b1 comps
        #     c_MIBs = np.copy(MIBs_ordered)
        #     bool_tot = np.logical_and(c_MIBs[:,0] > 50, c_MIBs[:, 0] < 60 )
        #     bool_tot2 = np.logical_and(bool_tot, c_MIBs[:, 1] > 50)
        #     arrays_to_sort = np.linalg.norm(c_MIBs[bool_tot2, :], axis=1)
        #     argsort_typeB = np.argsort(arrays_to_sort)
        #     c_MIBs[bool_tot2, :] = c_MIBs[bool_tot2, :][argsort_typeB]
        #     # print('In this loop')
        #     MIBs_ordered = c_MIBs

        #     # For negative b1
        #     c_MIBs = np.copy(MIBs_ordered)
        #     bool_tot = np.logical_and(c_MIBs[:,0] < - 50, c_MIBs[:, 0] > -60 )
        #     bool_tot2 = np.logical_and(bool_tot, c_MIBs[:, 1] < -50)
        #     arrays_to_sort = np.linalg.norm(c_MIBs[bool_tot2, :], axis=1)
        #     argsort_typeB = np.argsort(arrays_to_sort)
        #     c_MIBs[bool_tot2, :] = c_MIBs[bool_tot2, :][argsort_typeB]
        #     MIBs_ordered = c_MIBs

        if color_lines is not None:
            ax.plot(
                MDBs_ordered[:, 0],
                MDBs_ordered[:, 1],
                label=legend_label,
                linestyle=linestyle,
                color=color_lines,
                linewidth=2.0,
            )
        else:
            ax.plot(
                MDBs_ordered[:, 0],
                MDBs_ordered[:, 1],
                label=legend_label,
                linestyle=linestyle,
                linewidth=2.0,
            )
        # ax.scatter(MIBs_ordered[:, 0], MIBs_ordered[:,1], label=legend_label.replace('_', ''), marker='.')

    return ax, MDBs_ordered, PCD_computed


# %%
if __name__ == "__main__":
    type_of_testing = "DS"
    type_of_DS = "A"
    type_of_example = "simple"
    # type_of_example = 'SPP_GNSS'
    alpha_0 = 0.01
    alpha_prime = 0.038
    type_of_alpha = "Kok_IDS"

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
        m, n, r, A, alpha, sigma, Qyy, Qyy_inv, B_T, Qtt, Qtt_inv = setup(
            alpha_0=alpha_0, alpha_method=type_of_alpha, alpha_prime=alpha_prime
        )

    elif type_of_example == "SPP_GNSS":
        indices_partitions = load_indices_partitions_GNSS_ex(
            False
        )  # IDS_sep_order = False
        m, n, r, A, alpha, sigma, Qyy, Qyy_inv, B_T, Qtt, Qtt_inv = (
            setup_SPP_GNSS_example(
                alpha_0=alpha_0, alpha_method=type_of_alpha, alpha_prime=alpha_prime
            )
        )

    elif type_of_example == "Safoora_GNSS":
        m, n, r, A, alpha, sigma, Qyy, Qyy_inv, B_T, Qtt, Qtt_inv = (
            load_setup_parameters(
                type_of_example=type_of_example,
                alpha_method=type_of_alpha,
                alpha_prime=alpha_prime,
                alpha_0=alpha_0,
            )
        )

    elif type_of_example == "Sebastian_GNSS":
        m, n, r, A, alpha, sigma, Qyy, Qyy_inv, B_T, Qtt, Qtt_inv = (
            load_setup_parameters(
                type_of_example=type_of_example,
                alpha_method=type_of_alpha,
                alpha_prime=alpha_prime,
                alpha_0=alpha_0,
            )
        )
    elif type_of_example == "ARAIM_UNDEC_GNSS":
        m, n, r, A, alpha, sigma, Qyy, Qyy_inv, B_T, Qtt, Qtt_inv = (
            load_setup_parameters(
                type_of_example=type_of_example,
                alpha_method=type_of_alpha,
                alpha_prime=alpha_prime,
                alpha_0=alpha_0,
            )
        )

    # for_partition = 'P13'
    for_partition = "P24"
    q_i = len(for_partition.replace("P", ""))
    idx_under_hypt = indices_partitions[for_partition]

    n_points = 1000  # number of points on a 2D or 3D sphere for the unit vector d.

    if q_i == 1:  # the MDB is 1D, simply a value
        d_vectors = np.array([[1]])

        # Gather the indices from 'for_partition'
        idxes = int(for_partition.replace("P", ""))

        ci_model = np.eye(m)[:, idxes - 1].reshape(-1, 1)

    elif q_i == 2:  # the MIB is is a 2D figure, i.e. ellipse or circle
        # Number of vectors to generate (resolution)

        # Angles from 0 to 2π
        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

        # Create unit vectors
        d_vectors = np.stack(
            (np.cos(theta), np.sin(theta)), axis=0
        )  # shape: (2, n_points)

        idxes = for_partition.replace("P", "")
        idxes = np.array([int(zeta) for zeta in idxes])
        ci_model = np.eye(m)[:, idxes - 1]

    elif q_i == 3:  # the MDB is a 3D figure, i.e. an ellipsoid or spheroid.
        _, _, _, d_vectors = generate_t_grid(31, 32)
        idxes = np.array([int(zeta) for zeta in for_partition.replace("P", "")])
        ci_model = np.eye(m)[:, idxes - 1]

    PCD_goal = 0.7
    cti_model = B_T @ ci_model
    N_samples = int(1e6)
    max_its = 500
    store_MDB = np.zeros(d_vectors.shape)
    store_PCD_computed = np.zeros(d_vectors.shape[1])

    if type_of_DS == "C" or type_of_testing == "R_IDS":
        # these regions do not have 'spherically' shaped OMT regions. I.e. no simple integration regions.
        for d_i in range(d_vectors.shape[1]):
            d_vector = d_vectors[:, d_i].reshape(-1, 1)
            b_mag = 1.0
            d_b_mag = 0.01
            it_counter = 0
            PCD_computed = 1.1 * PCD_goal  # Reset for every d_vector

            while PCD_computed < PCD_goal and it_counter < max_its:
                # Update bias
                print("At it, PCD computed", it_counter, PCD_computed)
                print("b_mag", b_mag)
                b_mag += d_b_mag

                # Simulate test statistics
                t_mean = b_mag * cti_model @ d_vector
                t_samples = scipy.stats.multivariate_normal.rvs(
                    mean=t_mean.flatten(), cov=Qtt, size=(N_samples,)
                ).T

                # Compute PCI
                Pis = compute_Pis(
                    t_samples,
                    Qtt,
                    B_T,
                    alpha_prime,
                    type_of_testing,
                    type_of_DS=type_of_DS,
                    type_of_example=type_of_example,
                )

                PCD_computed = np.sum(Pis != 0) / len(Pis)

                it_counter += 1

            if it_counter == max_its:
                store_MDB[:, d_i] = np.nan
                store_PCD_computed[d_i] = np.nan
            else:
                store_MDB[:, d_i] = b_mag * d_vector.flatten()
                store_PCD_computed[d_i] = PCD_computed

    else:
        ## We can compute the MDB ellipse analytically
        beta_0 = 1 - PCD_goal
        T_alpha = scipy.stats.chi2.isf(alpha, df=r)
        lambda_0 = np.linspace(0.01, 100, 10000)
        beta_0_calculated = scipy.stats.ncx2.cdf(T_alpha, df=r, nc=lambda_0)
        (indBeta0,) = np.where(beta_0_calculated < beta_0)
        # print(beta_0_calculated)

        if len(indBeta0) == 0:
            raise Exception(
                "The lambda_0 parameter is not found, increase the maximum of its array"
            )
        indx = indBeta0[0]
        lambda_0 = lambda_0[indx]
        for d_i in range(d_vectors.shape[1]):
            d_vector = d_vectors[:, d_i].reshape(-1, 1)
            cti_di_magSquared = (
                (cti_model @ d_vector).T @ Qtt_inv @ (cti_model @ d_vector)
            )
            b_mag = np.sqrt(lambda_0) / np.sqrt(cti_di_magSquared[0, 0])

            store_MDB[:, d_i] = b_mag * d_vector.flatten()
            store_PCD_computed[d_i] = PCD_goal

        resDir = os.path.join(
            r"C:\Users\bgvannoort\Documents\IDS\Results\MDBs",
            type_of_example,
            for_partition.replace("P", "H"),
            "PCD_goal={}".format(PCD_goal),
            type_of_testing,
            type_of_DS,
        )

        tosaveMDB = np.hstack((np.arange(1000).reshape(-1, 1), store_MDB.T))
        tosavePCD = np.hstack(
            (np.arange(1000).reshape(-1, 1), store_PCD_computed.reshape(-1, 1))
        )
        np.savetxt(os.path.join(resDir, "MDB_vectors.txt"), tosaveMDB)
        np.savetxt(os.path.join(resDir, "PCD_computed.txt"), tosavePCD)

    sys.exit()
