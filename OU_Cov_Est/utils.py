import collections
import itertools
import numpy as np
import matplotlib.pyplot as plt

def prime_factors(n):
    i = 2
    while i * i <= n:
        if n % i == 0:
            n /= i
            yield i
        else:
            i += 1

    if n > 1:
        yield n


def prod(iterable):
    result = 1
    for i in iterable:
        result *= i
    return result


def get_divisors(n):
    pf = prime_factors(n)

    pf_with_multiplicity = collections.Counter(pf)

    powers = [
        [factor ** i for i in range(count + 1)]
        for factor, count in pf_with_multiplicity.items()
    ]

    for prime_power_combo in itertools.product(*powers):
        yield prod(prime_power_combo)

def plot_OU_tau_lenght_scale(ax, Pinelis_bound, Pinelis_emp_bound_biased_cov_est, Pinelis_emp_bound_unbiased_cov_est, M_bound, M_emp_bound_biased_cov_est, M_emp_bound_unbiased_cov_est, taus, length_scale, show_ylabel=False):
    Pinelis_bound_mean = np.mean(Pinelis_bound, axis=-1)
    Pinelis_bound_std = np.std(Pinelis_bound, axis=-1)
    Pinelis_emp_bound_biased_cov_est_mean = np.mean(Pinelis_emp_bound_biased_cov_est, axis=-1)
    Pinelis_emp_bound_biased_cov_est_std = np.std(Pinelis_emp_bound_biased_cov_est, axis=-1)
    Pinelis_emp_bound_unbiased_cov_est_mean = np.mean(Pinelis_emp_bound_unbiased_cov_est, axis=-1)
    Pinelis_emp_bound_unbiased_cov_est_std = np.std(Pinelis_emp_bound_unbiased_cov_est, axis=-1)

    M_bound_mean = np.mean(M_bound, axis=-1)
    M_bound_std = np.std(M_bound, axis=-1)
    M_emp_bound_biased_cov_est_mean = np.mean(M_emp_bound_biased_cov_est, axis=-1)
    M_emp_bound_biased_cov_est_std = np.std(M_emp_bound_biased_cov_est, axis=-1)    
    M_emp_bound_unbiased_cov_est_mean = np.mean(M_emp_bound_unbiased_cov_est, axis=-1)
    M_emp_bound_unbiased_cov_est_std = np.std(M_emp_bound_unbiased_cov_est, axis=-1)

    # Plot with appropriate figure size and font sizes
    line1, = ax.loglog(taus, Pinelis_emp_bound_unbiased_cov_est_mean, marker='o', label="Pinelis bound", linewidth=1)
    ax.fill_between(taus, Pinelis_emp_bound_unbiased_cov_est_mean - Pinelis_emp_bound_unbiased_cov_est_std,
                    Pinelis_emp_bound_unbiased_cov_est_mean + Pinelis_emp_bound_unbiased_cov_est_std, alpha=0.2)

    line2, = ax.loglog(taus, Pinelis_emp_bound_biased_cov_est_mean, marker='s', label="Pinelis emp. bound (biased cov. est.)", linewidth=1)
    ax.fill_between(taus, Pinelis_emp_bound_biased_cov_est_mean - Pinelis_emp_bound_biased_cov_est_std,
                    Pinelis_emp_bound_biased_cov_est_mean + Pinelis_emp_bound_biased_cov_est_std, alpha=0.2)
    

    line3, = ax.loglog(taus, Pinelis_emp_bound_unbiased_cov_est_mean, marker='s', label="Pinelis emp. bound (unbiased cov. est.)", linewidth=1)
    ax.fill_between(taus, Pinelis_emp_bound_unbiased_cov_est_mean - Pinelis_emp_bound_unbiased_cov_est_std,
                    Pinelis_emp_bound_unbiased_cov_est_mean + Pinelis_emp_bound_unbiased_cov_est_std, alpha=0.2)
    
    
    line4 = ax.loglog(taus, M_bound_mean, marker='^', label="M bound", linewidth=1)
    ax.fill_between(taus, M_bound_mean - M_bound_std,
                    M_bound_mean + M_bound_std, alpha=0.2)
        
    line5 = ax.loglog(taus, M_emp_bound_biased_cov_est_mean, marker='P', label="M emp. bound (biased cov. est.)", linewidth=1)
    ax.fill_between(taus, M_emp_bound_biased_cov_est_mean - M_emp_bound_biased_cov_est_std,
                    M_emp_bound_biased_cov_est_mean + M_emp_bound_biased_cov_est_std, alpha=0.2)
    
    line6 = ax.loglog(taus, M_emp_bound_unbiased_cov_est_mean, marker='P', label="M emp. bound (unbiased cov. est.)", linewidth=1)
    ax.fill_between(taus, M_emp_bound_unbiased_cov_est_mean - M_emp_bound_unbiased_cov_est_std,
                    M_emp_bound_unbiased_cov_est_mean + M_emp_bound_unbiased_cov_est_std, alpha=0.2)
    

    ax.set_xlabel("Block size", fontsize=10)
    if show_ylabel:
        ax.set_ylabel("Covariance upper bound", fontsize=10)
    ax.set_title(f"length scale ={length_scale}", fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.grid(True)
    
    return line1, line2, line3, line4, line5, line6

def plot_OU_tau(configs, length_scales, Pinelis_bound, Pinelis_emp_bound_biased_cov_est, Pinelis_emp_bound_unbiased_cov_est, M_bound, M_emp_bound_biased_cov_est, M_emp_bound_unbiased_cov_est, taus, length_scale):
    # Create a figure with 3 subplots in a row, single-column width (3.25 inches)
    fig, axes = plt.subplots(1, len(length_scales), figsize=(3.25 * 3, 4.5))  # Adjust height as needed for visibility

    # Plot each subplot and collect lines for the legend
    lines = []
    for i, length_scale in enumerate(length_scales):
        show_ylabel = (i == 0)  # Only show y-axis label on the first subplot
        lines += plot_OU_tau(axes[i],Pinelis_bound, Pinelis_emp_bound_biased_cov_est, Pinelis_emp_bound_unbiased_cov_est, M_bound, M_emp_bound_biased_cov_est, M_emp_bound_unbiased_cov_est, taus, length_scale, show_ylabel=show_ylabel)

    # Create a common legend
    labels = ["Pinelis bound", "Pinelis emp. bound (biased cov. est.)", "Pinelis emp. bound (unbiased cov. est.)", "M bound", "M emp. bound (biased cov. est.)", "M emp. bound (unbiased cov. est.)"]
    n_labels = len(labels)
    fig.legend(lines[:n_labels], labels, loc='upper center', fontsize=10, ncol=len(length_scales), frameon=False)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.85])
    plt.subplots_adjust(top=0.85)  # Add more space between title and legend
    plt.savefig(f"OU_Exp_tau_n_{configs.n_plot_tau}_delta_{configs.delta}.pdf", format="pdf", dpi=600)
    plt.show()



# def plot_OU2(pess_bound, data_bound_biased_cov_est, data_bound_unbiased_cov_est,True_value, Ns, delta, length_scale):
#     pess_bound_mean = np.mean(pess_bound, axis=-1)
#     pess_bound_std = np.std(pess_bound, axis=-1)

#     data_bound_biased_est_mean = np.mean(data_bound_biased_cov_est, axis=-1)
#     data_bound_biased_est_std = np.std(data_bound_biased_cov_est, axis=-1)

#     data_bound_unbiased_est_mean = np.mean(data_bound_unbiased_cov_est, axis=-1)
#     data_bound_unbiased_est_std = np.std(data_bound_unbiased_cov_est, axis=-1)

#     true_value_mean = np.mean(True_value, axis=-1)
#     true_value_std = np.std(True_value, axis=-1)

#     # Plot with larger figure size and font sizes
#     plt.figure(figsize=(12, 8))  # Adjust figure size as needed

#     plt.loglog(Ns, data_bound_unbiased_est_mean, marker='o', label="Emp. bound (unbiased cov. est.)", linewidth=2)
#     plt.fill_between(Ns, data_bound_unbiased_est_mean - data_bound_unbiased_est_std, 
#                      data_bound_unbiased_est_mean + data_bound_unbiased_est_std, alpha=0.2)

#     plt.loglog(Ns, data_bound_biased_est_mean, marker='s', label="Emp. bound (biased cov. est.)", linewidth=2)
#     plt.fill_between(Ns, data_bound_biased_est_mean - data_bound_biased_est_std, 
#                      data_bound_biased_est_mean + data_bound_biased_est_std, alpha=0.2)

#     plt.loglog(Ns, pess_bound_mean, marker='^', label="Pessimistic bound", linewidth=2)
#     plt.fill_between(Ns, pess_bound_mean - pess_bound_std, 
#                      pess_bound_mean + pess_bound_std, alpha=0.2)
    
#     plt.loglog(Ns, true_value_mean, marker='P', label= "Estimated True value", linewidth=2)
#     plt.fill_between(Ns, true_value_mean - true_value_std, 
#                      true_value_mean + true_value_std, alpha=0.2)

#     plt.xlabel("Number of training samples", fontsize=18)
#     plt.ylabel("Covariance upper bound", fontsize=18)
#     # plt.title(f"Covariance Upper Bounds (delta = {delta}) of OU Process with Gaussian Kernel (l={length_scale})", fontsize=16)

#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)
#     plt.legend(fontsize=16)
#     plt.grid(True)
    
#     plt.savefig(f"OU_Exp_delta_{delta}_l_{length_scale}.pdf", format="pdf", dpi=900)
#     plt.show()

# def plot_OU2(ax, pess_bound, data_bound_biased_cov_est, data_bound_unbiased_cov_est, True_value, Ns, delta, length_scale, show_ylabel=False):
#     pess_bound_mean = np.mean(pess_bound, axis=-1)
#     pess_bound_std = np.std(pess_bound, axis=-1)

#     data_bound_biased_est_mean = np.mean(data_bound_biased_cov_est, axis=-1)
#     data_bound_biased_est_std = np.std(data_bound_biased_cov_est, axis=-1)

#     data_bound_unbiased_est_mean = np.mean(data_bound_unbiased_cov_est, axis=-1)
#     data_bound_unbiased_est_std = np.std(data_bound_unbiased_cov_est, axis=-1)

#     true_value_mean = np.mean(True_value, axis=-1)
#     true_value_std = np.std(True_value, axis=-1)

#     # Plot with appropriate figure size and font sizes
#     line1, = ax.loglog(Ns, data_bound_unbiased_est_mean, marker='o', label="Emp. bound (unbiased cov. est.)", linewidth=1)
#     ax.fill_between(Ns, data_bound_unbiased_est_mean - data_bound_unbiased_est_std, 
#                     data_bound_unbiased_est_mean + data_bound_unbiased_est_std, alpha=0.2)

#     line2, = ax.loglog(Ns, data_bound_biased_est_mean, marker='s', label="Emp. bound (biased cov. est.)", linewidth=1)
#     ax.fill_between(Ns, data_bound_biased_est_mean - data_bound_biased_est_std, 
#                     data_bound_biased_est_mean + data_bound_biased_est_std, alpha=0.2)

#     line3, = ax.loglog(Ns, pess_bound_mean, marker='^', label="Pessimistic bound", linewidth=1)
#     ax.fill_between(Ns, pess_bound_mean - pess_bound_std, 
#                     pess_bound_mean + pess_bound_std, alpha=0.2)
    
#     line4, = ax.loglog(Ns, true_value_mean, marker='P', label="Estimated True value", linewidth=1)
#     ax.fill_between(Ns, true_value_mean - true_value_std, 
#                     true_value_mean + true_value_std, alpha=0.2)

#     ax.set_xlabel("Number of training samples", fontsize=10)
#     if show_ylabel:
#         ax.set_ylabel("Covariance upper bound", fontsize=10)
#     ax.set_title(f"length scale={length_scale}", fontsize=10)
#     ax.tick_params(axis='both', which='major', labelsize=10)
#     ax.grid(True)
    
#     return line1, line2, line3, line4