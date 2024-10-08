import collections
import itertools
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
main_path = Path(__file__).parent

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

def plot_risk_bound_N_length_scale(ax, emp_risk, upper_bound, lower_bound, test_emp_risk, Ns, length_scale, show_ylabel=False):
    emp_risk_mean = np.mean(emp_risk, axis=-1)
    emp_risk_std = np.std(emp_risk, axis=-1)
    upper_bound_mean = np.mean(upper_bound, axis=-1)
    upper_bound_std = np.std(upper_bound, axis=-1)
    lower_bound_mean = np.mean(lower_bound, axis=-1)
    lower_bound_std = np.std(lower_bound, axis=-1)
    test_emp_risk_mean = np.mean(test_emp_risk, axis=-1)
    test_emp_risk_std = np.std(test_emp_risk, axis=-1)

    # Plot with larger figure size and font sizes
    line1 = ax.loglog(Ns, emp_risk_mean, marker='o', label="Empirical risk", linewidth=1)
    ax.fill_between(Ns, emp_risk_mean - emp_risk_std,
                        emp_risk_mean + emp_risk_std, alpha=0.2)
    
    line2 = ax.loglog(Ns, upper_bound_mean, marker='s', label="Upper Bound", linewidth=1)
    ax.fill_between(Ns, upper_bound_mean - upper_bound_std,
                        upper_bound_mean + upper_bound_std, alpha=0.2)
    
    line3 = ax.loglog(Ns, lower_bound_mean, marker='^', label="Lower Bound", linewidth=1)
    ax.fill_between(Ns, lower_bound_mean - lower_bound_std,
                        lower_bound_mean + lower_bound_std, alpha=0.2)

    line4 = ax.loglog(Ns, test_emp_risk_mean, marker='x', label="Test empirical risk", linewidth=1)
    ax.fill_between(Ns, test_emp_risk_mean - test_emp_risk_std,
                        test_emp_risk_mean + test_emp_risk_std, alpha=0.2)
    
    ax.set_xlabel("Number of training samples", fontsize=10)
    if show_ylabel:
        ax.set_ylabel("Risk", fontsize=10)
    ax.set_title(f"length scale ={length_scale}", fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True)

    return line1, line2, line3, line4

def plot_risk_bound_N(configs, emp_risk, risk_bound, test_emp_risk, Ns, length_scales, labels, show_ylabel=False):

    # Create a figure with 3 subplots in a row, single-column width (3.25 inches)
    fig, axes = plt.subplots(1, len(length_scales), figsize=(3.25 * 3, 4.5))  # Adjust height as needed for visibility

    # Plot each subplot and collect lines for the legend
    lines = []
    for i, length_scale in enumerate(length_scales):
        show_ylabel = (i == 0)  # Only show y-axis label on the first subplot
        lines += plot_risk_bound_N_length_scale(axes[i], emp_risk[i], emp_risk[i] + risk_bound[i], emp_risk[i] - risk_bound[i], test_emp_risk[i], Ns, length_scale, show_ylabel=show_ylabel)

    # Create a common legend
    n_labels = len(labels)
    import matplotlib.lines as mlines

    # Flatten the list of lines in case it's nested (contains lists)
    flattened_lines = []
    for line_group in lines[:n_labels]:
        if isinstance(line_group, list):  # If it's a list of lines, flatten it
            flattened_lines.extend(line_group)
        else:
            flattened_lines.append(line_group)

    # Create proxy Line2D objects for the flattened lines
    legend_lines = [mlines.Line2D([], [], color=line.get_color()) for line in flattened_lines]

    # Pass the proxy Line2D objects to the legend
    fig.legend(legend_lines, labels, loc='upper center', fontsize=10, ncol=len(length_scales), frameon=False)
    # fig.legend(lines[:n_labels], labels, loc='upper center', fontsize=10, ncol=n_labels, frameon=False)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.85])
    plt.subplots_adjust(top=0.85)  # Add more space between title and legend
    plt.savefig(str(main_path) + f"/results/risk_bound_Exp_N_n_{configs.n_plot_tau}_delta_{configs.delta}_lamda_{configs.lamda}.pdf", format="pdf", dpi=600)