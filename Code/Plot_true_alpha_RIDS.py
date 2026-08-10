"""
Description of code:
    For reverse IDS, it is difficult to specify the true alpha", i.e. the probability of
    a type I error.
    We can specify alpha_0, the "significance" of the w-test, and the alpha_check, which is the
    significance of the OMT at that iteration.
    Here, we play around with alpha_0 and alpha_check to see what the final alpha is that we obtain.
Author:
    Bob van Noort
Date:
    07 05 2025

"""

import numpy as np
from Functions import (
    compute_Pis,
    RIDS_mult_its_type_A,
    RIDS_mult_its_type_B,
    RIDS_mult_its_type_C,
    load_setup_parameters,
)
import scipy
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import os
# %%


N_sims = 10
N_samples = int(1e6)
type_of_testing = "R_IDS"
type_of_DS = "A"
type_of_example = "RIDS_EXAMPLE_ALPHA_PRIME"
alpha_method = "manual"
alpha_0 = 0.01

#### ------------------- get setup params
m, n, r, A, alpha, sigma, Qyy, Qyy_inv, Bt, Qtt, Qtt_inv = load_setup_parameters(
    type_of_example=type_of_example, alpha_method=alpha_method, alpha_0=alpha_0
)

print("alpha", alpha)
qmax_list = np.arange(2, 6)
alpha_0_list = np.array(
    [0.1, 0.05, 0.01, 0.005, 0.001]
)  # make sure thesea re teh same as in the original file
# Store the alpha' parameters
alpha_prime_list = np.zeros((len(qmax_list), N_sims, len(alpha_0_list)))

plot_together = True  # If we want to plot both type C and A RIDS together.
Example_String = "Example1"
fig, ax = plt.subplots(figsize=(16, 10))
plt.rcParams["font.size"] = 16

plt.rcParams.update(
    {
        "font.size": 20,  # General font size
        "axes.titlesize": 20,  # Title font
        "axes.labelsize": 20,  # X/Y label font
        "xtick.labelsize": 20,  # X tick labels
        "ytick.labelsize": 20,  # Y tick labels
        "legend.fontsize": 20,  # Legend font
        "figure.titlesize": 20,  # Figure title (if used)
    }
)
linestyles = ["-.", "--", "-"]
colors_default = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# %% Load the data again

combined_alpha = [0.1, 0.05, 0.01]  # remove all but the one that you need.
# combined_alpha = [0.01] # we ran this configuration for type C, but really it does not depend on this value of alpha_check.
# alpha=0.1
for idxalphacheck, alpha in enumerate(combined_alpha):
    for idx_alpha0, alpha_0 in enumerate(alpha_0_list):
        cc = r"C:\Users\bgvannoort\Documents\IDS\Results\RIDS_alpha"
        if type_of_DS == "A" and Example_String == "Example1":
            cc = os.path.join(
                cc, Example_String, "alpha_0={}_alpha_check={}".format(alpha_0, alpha)
            )

        else:
            cc = os.path.join(
                cc,
                Example_String,
                "alpha_0={}_alpha_check={}".format(alpha_0, alpha),
                "type_of_DS_{}".format(type_of_DS),
            )

        dat = np.loadtxt(os.path.join(cc, "alpha_prime_computed.txt"))
        alpha_prime_list[:, :, idx_alpha0] = dat

    # %% Plot the results

    for idx_qmax, qmax in enumerate(qmax_list):
        results = alpha_prime_list[idx_qmax, :, :]
        means = np.mean(results, axis=0)
        stds = np.std(results, axis=0)
        # if idxalphacheck == 2:
        ax.errorbar(
            alpha_0_list,
            means,
            yerr=stds,
            label=r"$q_{\text{max}}$ = " + str(qmax),
            linestyle=linestyles[idxalphacheck],
            color=colors_default[idx_qmax],
            linewidth=2,
        )
        # else:

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim(0.5e-2, 0.6)

    ax.set_xlabel(r"$\alpha_0$ [-]")
    ax.set_ylabel(r"$\alpha_{\text{true}}'$ [-]")

if plot_together:
    # combined_alpha = [0.1, 0.05, 0.01] # remove all but the one that you need.
    combined_alpha_typeC = [
        0.01
    ]  # we ran this configuration for type C, but really it does not depend on this value of alpha_check.
    # alpha=0.1
    for idxalphacheck, alpha in enumerate(combined_alpha_typeC):
        for idx_alpha0, alpha_0 in enumerate(alpha_0_list):
            cc = r"C:\Users\bgvannoort\Documents\IDS\Results\RIDS_alpha"
            cc = os.path.join(
                cc,
                Example_String,
                "alpha_0={}_alpha_check={}".format(alpha_0, alpha),
                "type_of_DS_C",
            )

            dat = np.loadtxt(os.path.join(cc, "alpha_prime_computed.txt"))
            alpha_prime_list[:, :, idx_alpha0] = dat

        # %% Plot the results

        for idx_qmax, qmax in enumerate(qmax_list):
            results = alpha_prime_list[idx_qmax, :, :]
            means = np.mean(results, axis=0)
            stds = np.std(results, axis=0)
            # if idxalphacheck == 2:
            ax.errorbar(
                alpha_0_list,
                means,
                yerr=stds,
                label=r"$q_{\text{max}}$ = " + str(qmax),
                linestyle=":",
                color=colors_default[idx_qmax],
                linewidth=2,
            )
        ax.text()
        # else:


# Define custom legend for qmax (colored lines)
qmax_colors = [line.get_color() for line in ax.get_lines()[: len(qmax_list)]]
qmax_legend = [
    Line2D([0], [0], color=color, lw=2, label=rf"$q_{{\text{{max}}}}$ = {q}")
    for color, q in zip(qmax_colors, qmax_list)
]

# Define custom legend for alpha_check (black lines with different linestyles)

if type_of_DS == "C":
    alpha_legend = []
else:
    alpha_legend = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=ls,
            lw=2,
            label=rf"$\widehat{{\alpha}}$ = {a}",
        )
        for a, ls in zip(combined_alpha, linestyles)
    ]

if plot_together:
    alpha_legend += [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=":",
            lw=2,
            label=r"$\widehat{{\alpha}}$ = 0",
        )
    ]

ax.set_aspect(1)
# Combine legends and add to the plot
custom_legend = qmax_legend  # + alpha_legend
ax.legend(handles=custom_legend, loc="lower right")
# ax.grid('on')
ax.set_title(
    "Example {}".format(
        Example_String.replace("Example", "") + " for {} RIDS".format(type_of_DS)
    )
)
ax.minorticks_on()
ax.grid(which="both", linestyle="-")
plt.tight_layout()
