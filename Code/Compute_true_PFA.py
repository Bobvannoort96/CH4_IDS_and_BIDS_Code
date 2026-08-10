"""
Description of code:

Author:
    Bob van Noort
Date:
    ${DATE}

"""

import time
from Functions import *


t0 = time.process_time()
type_of_DS = "A"
type_of_testing = "IDS"
type_of_example = "SPP_GNSS"
# type_of_example='ARAIM_UNDEC_GNSS'
# type_of_example = 'simple'
type_of_alpha = "Kok_IDS"
alpha_0 = 0.01
bool_start_with_OMT = True
separate_order_IDS = False

m, n, r, A, alpha, sigma, Qyy, Qyy_inv, Bt, Qtt, Qtt_inv = load_setup_parameters(
    type_of_example=type_of_example, alpha_method=type_of_alpha, alpha_0=alpha_0
)

t_new = np.random.multivariate_normal(np.zeros(r), cov=Qtt, size=int(1e4)).T

S_0 = []
idx_S = 0

qmax = 2
PFA_goal = 1 - 0.993021
PFA = 0

while PFA < PFA_goal:
    # alpha_0 += +.0001
    Pis = compute_Pis(
        t_new,
        Qtt,
        B_T=Bt,
        alpha_prime=alpha,
        type_of_testing=type_of_testing,
        type_of_DS=type_of_DS,
        alpha_method=type_of_alpha,
        alpha_0=alpha_0,
        S=[],
        cti_list=None,
        idx_S=idx_S,
        qmax=qmax,
        type_of_example=type_of_example,
        separate_order_IDS=separate_order_IDS,
        bool_start_with_OMT=bool_start_with_OMT,
    )
    PFA = np.sum(Pis != 0) / len(Pis)
    print("PFA true = ", PFA)
    break
print("alpha0", alpha_0)
