import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm
import scipy

# Degrees of freedom
dfs = [1, 2, 3, 4, 5, 6]

# x values for the chi-squared plots
x_chi2 = np.linspace(0.001, 20, 1000)
# x values for the normal plots
x_norm = np.linspace(-4, 4, 1000)

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# Plot PDFs of central chi-squared distributions
for df in dfs:
    pdf_chi2 = chi2.pdf(x_chi2, df)
    ax1.plot(x_chi2, pdf_chi2, label=f"chi2 df={df}")

# Plot PDF of the standard normal distribution
pdf_norm = norm.pdf(x_norm)
ax1.plot(x_norm, pdf_norm, label="Standard Normal", linestyle="--")

ax1.set_title("PDF of Central Chi-Squared and Standard Normal Distributions")
ax1.set_xlabel("x")
ax1.set_ylabel("Probability Density Function")
ax1.legend()
ax1.grid(True)

# Plot CDFs of central chi-squared distributions
for df in dfs:
    cdf_chi2 = chi2.cdf(x_chi2, df)
    ax2.plot(x_chi2, cdf_chi2, label=f"chi2 df={df}")

# Plot CDF of the standard normal distribution
cdf_norm = norm.cdf(x_norm)
ax2.plot(x_norm, cdf_norm, label="Standard Normal", linestyle="--")

ax2.set_title("CDF of Central Chi-Squared and Standard Normal Distributions")
ax2.set_xlabel("x")
ax2.set_ylabel("Cumulative Distribution Function")
ax2.legend()
ax2.grid(True)

# Show the plots
plt.tight_layout()
plt.show()


# %% test for transformations

alpha_is = chi2.sf(x_chi2, df=2)
w_is = scipy.stats.norm.isf(alpha_is)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def plot_histogram_and_abs_normal_pdf(w_is):
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # Plot histogram of the values in w_is
    ax1.hist(w_is, bins=30, density=True, alpha=0.6, color="g")
    ax1.set_title("Histogram of w_is")
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Density")
    ax1.grid(True)

    # Values for the x-axis for the PDF plot
    x = np.linspace(0, 5, 1000)

    # PDF of the absolute value of a standard normally distributed random variable
    pdf_abs_normal = 2 * norm.pdf(
        x, loc=0, scale=1
    )  # Multiply by 2 to account for the symmetry of the normal distribution

    # Plot the PDF of the absolute value of a standard normal distribution
    ax2.plot(x, pdf_abs_normal, label="$|Z|$ where $Z \\sim N(0, 1)$")
    ax2.set_title("PDF of |Z| where Z ~ N(0, 1)")
    ax2.set_xlabel("Value")
    ax2.set_ylabel("Probability Density Function")
    ax2.legend()
    ax2.grid(True)

    # Show the plots
    plt.tight_layout()
    plt.show()


def compare_w_and_transformed_w(x):
    """
    inputs:
        x=array on which to plot, x-axis of w-values
    """
    w_normal = x
    fig, ax = plt.subplots()
    ax.plot(x, w_normal, label="Normal $w$-test, q=1")
    return_arr = []
    ## Tq = w_i^2 + \bar{w}_j^2 for q=2
    for wj in range(0, 5):
        Tq = x**2 + wj**2
        cdf_val = chi2.cdf(Tq, df=2)
        w_i_trans = norm.ppf(cdf_val)
        ax.plot(
            x, w_i_trans, label="$w_i(T_q)$ with $\overline{w}_j=" + "{}$".format(wj)
        )
        return_arr.append([cdf_val, w_i_trans])
    # ax.set_xlim([0,8])
    # ax.set_ylim([-5,8])
    ax.set_xlabel("$w_i$ value")
    ax.set_ylabel("Transformed $w_i$ value")
    ax.legend()
    return return_arr


zz = np.linspace(0.001, 20, 1000)
# plot_histogram_and_abs_normal_pdf(w_is)
values = compare_w_and_transformed_w(zz)

# %%
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


# Function to generate chi-squared distributed values, transform and plot histograms
def chi2_normal_transformation(
    degrees_of_freedom_list, num_samples=int(1e7), num_bins=50
):
    fig, axes = plt.subplots(
        len(degrees_of_freedom_list) + 1,
        1,
        figsize=(10, 5 * len(degrees_of_freedom_list) + 1),
    )

    # Collect all transformed values to determine common x-axis limits
    all_transformed_values = []

    for i, df in enumerate(degrees_of_freedom_list):
        # Step 1: Generate chi-squared distributed values
        chi2_values = np.random.chisquare(df, num_samples)

        # Step 2: Compute the CDF values of the chi-squared values
        cdf_values = stats.chi2.cdf(chi2_values, df)

        # Step 3: Transform CDF values using the inverse CDF (ppf) of a standard normal distribution
        w_i_transformed = stats.norm.ppf(cdf_values)
        all_transformed_values.append(w_i_transformed)

        # Plot histogram of the transformed values
        axes[i].hist(w_i_transformed, bins=num_bins, density=True, alpha=0.6, color="g")
        axes[i].set_title(f"Transformed Histogram for chi2 df={df}")
        axes[i].set_xlabel("Transformed Values")
        axes[i].set_ylabel("Frequency")

    # Flatten the list of all transformed values to determine the common x-axis limits
    all_transformed_values = np.concatenate(all_transformed_values)
    xlim = (np.min(all_transformed_values), np.max(all_transformed_values))

    # Apply the same x-axis limits to all subplots
    for ax in axes:
        ax.set_xlim(xlim)

    # Step 4: Plot the histogram of a standard normal distribution
    normal_values = np.random.normal(0, 1, num_samples)
    axes[-1].hist(normal_values, bins=num_bins, density=True, alpha=0.6, color="b")
    axes[-1].set_title("Histogram for Standard Normal Distribution")
    axes[-1].set_xlabel("Standard Normal Values")
    axes[-1].set_ylabel("Frequency")
    axes[-1].set_xlim(xlim)

    plt.tight_layout()
    plt.show()


# Define degrees of freedom to be tested
degrees_of_freedom_list = [1, 2, 3, 4, 5]
chi2_normal_transformation(degrees_of_freedom_list)

chi2_normal_transformation(degrees_of_freedom_list)
