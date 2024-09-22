import numpy as np
import matplotlib.pyplot as plt
from utils import plot_OU_tau, plot_OU2

n = 10000
delta = 0.05
length_scales = [0.05, 0.5, 5]
pess_bounds = []
data_bounds_biased = []
data_bounds_unbiased = []
for l in length_scales:
    # pess_bound, data_bound_biased_cov_est, data_bound_unbiased_cov_est, taus = Covariance_Estimation(data_points, n, delta, l)
    # np.save(f'pess_bound_n_{n}_delta_{delta}_l_{l}.npy', pess_bound)
    # np.save(f'data_bound_biased_cov_est_n_{n}_delta_{delta}_l_{l}.npy', data_bound_biased_cov_est)
    # np.save(f'data_bound_unbiased_cov_est_n_{n}_delta_{delta}_l_{l}.npy', data_bound_unbiased_cov_est)

    # pess_bounds.append(pess_bound) 
    # data_bounds_biased.append(data_bound_biased_cov_est)
    # data_bounds_unbiased.append(data_bound_unbiased_cov_est)

    pess_bounds.append(np.load(f'pess_bound_n_{n}_delta_{delta}_l_{l}.npy'))
    data_bounds_biased.append(np.load(f'data_bound_biased_cov_est_n_{n}_delta_{delta}_l_{l}.npy'))
    data_bounds_unbiased.append(np.load(f'data_bound_unbiased_cov_est_n_{n}_delta_{delta}_l_{l}.npy'))

# Create a figure with 3 subplots in a row, single-column width (3.25 inches)
fig, axes = plt.subplots(1, 3, figsize=(3.25 * 3, 4.5))  # Adjust height as needed for visibility

# Plot each subplot and collect lines for the legend
lines = []
for i, length_scale in enumerate(length_scales):
    show_ylabel = (i == 0)  # Only show y-axis label on the first subplot
    lines += plot_OU_tau(axes[i], pess_bounds[i], data_bounds_biased[i], data_bounds_unbiased[i], taus, n=n, delta=delta, length_scale=length_scale, show_ylabel=show_ylabel)

# Create a common legend
labels = ["Emp. bound (unbiased cov. est.)", "Emp. bound (biased cov. est.)", "Pessimistic bound"]
fig.legend(lines[:3], labels, loc='upper center', fontsize=10, ncol=3, frameon=False)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.85])
plt.subplots_adjust(top=0.85)  # Add more space between title and legend
plt.savefig(f"OU_Exp_tau_n_{n}_delta_{delta}.pdf", format="pdf", dpi=600)
plt.show()