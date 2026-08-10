"""
Description of code:
    The code that computes the table of cos(angle) between the
    different fault vectors and planes.
Author:
    Bob van Noort
Date:
    12 May 2025
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt

from Functions import load_setup_parameters, P_mat, P_perp
import itertools


type_of_example = "simple"
m, n, r, A, alpha, sigma, Qyy, Qyy_inv, Bt, Qtt, Qtt_inv = load_setup_parameters(
    type_of_example, "Kok_IDS", alpha_0=0.01
)


qmax = 2
combinations = int(scipy.special.binom(m, qmax))

table = np.zeros((combinations, m))
order = []

for i, comb in enumerate(itertools.combinations(np.arange(m), qmax)):
    combs = np.array(comb)
    cij = np.eye(m)[:, combs]
    ctij = Bt @ cij
    Pctij = P_mat(ctij, Qtt_inv, inverse=True)

    for j in np.arange(m):
        ctj = Bt @ np.eye(m)[:, j].reshape(-1, 1)
        sqrd_norm = np.einsum("ij,jm,mi->i", ctj.T, Qtt_inv, ctj)

        ctj_proj = Pctij @ ctj
        sqrd_norm_proj = np.einsum("ij,jm,mi->i", ctj_proj.T, Qtt_inv, ctj_proj)

        cos_angle = (sqrd_norm_proj / sqrd_norm) ** 0.5
        table[i, j] = cos_angle

    order.append(combs)
