import importlib
import subprocess
import sys
for module in ['kooplearn', 'datasets', 'matplotlib', 'ml-confs']: # !! Add here any additional module that you need to install on top of kooplearn
    try:
        importlib.import_module(module)
    except ImportError:
        if module == 'kooplearn':
            module = 'kooplearn[full]'
        # pip install -q {module}
        subprocess.check_call([sys.executable, "-m", "pip", "install", module])

from pathlib import Path
import random
import ml_confs
import numpy as np
import matplotlib.pyplot as plt
from data_pipeline import make_dataset
from cov_estimation import Covariance_Estimation_tau, Cov_Est_N
from utils import plot_OU_tau, plot_OU2
CV = True

# Load configs
main_path = Path(__file__).parent
data_path = main_path / "__data__"
configs = ml_confs.from_file(main_path / "configs.yaml")

# Set the seed
random.seed(configs.rng_seed)
np.random.seed(configs.rng_seed)

# Lad data
data_points = make_dataset(configs)

n = 10000
delta = 0.05
length_scales = [0.05, 0.5, 5]
for l in length_scales:
    Pinelis_bound, Pinelis_emp_bound_biased_cov_est, Pinelis_emp_bound_unbiased_cov_est, M_bound, M_emp_bound_biased_cov_est, M_emp_bound_unbiased_cov_est, taus = Covariance_Estimation_tau(data_points, n, delta, l)
    np.save(f'pess_bound_n_{n}_delta_{delta}_l_{l}.npy', pess_bound)
    np.save(f'data_bound_biased_cov_est_n_{n}_delta_{delta}_l_{l}.npy', data_bound_biased_cov_est)
    np.save(f'data_bound_unbiased_cov_est_n_{n}_delta_{delta}_l_{l}.npy', data_bound_unbiased_cov_est)


delta = 0.05
length_scale= 0.5
Ns = [100,200,500,1000,2000,5000,10000] #20000,40000]

pess_bound, data_bound_biased_cov_est, data_bound_unbiased_cov_est, True_value = Cov_Est(data_points, Ns, delta, length_scale)
plot_OU2(pess_bound, data_bound_biased_cov_est, data_bound_unbiased_cov_est,True_value, Ns, delta, length_scale)
np.save(f'pess_bound_delta_{delta}_l_{length_scale}.npy', pess_bound)
np.save(f'data_bound_biased_cov_est_delta_{delta}_l_{length_scale}.npy', data_bound_biased_cov_est)
np.save(f'data_bound_unbiased_cov_est_delta_{delta}_l_{length_scale}.npy', data_bound_unbiased_cov_est)
np.save(f'True_value_delta_{delta}_l_{length_scale}.npy', True_value)

delta = 0.05
length_scale= 5
Ns = [100,200,500,1000,2000,5000,10000]#,20000,40000]

pess_bound, data_bound_biased_cov_est, data_bound_unbiased_cov_est, True_value = Cov_Est(data_points, Ns, delta, length_scale)
plot_OU2(pess_bound, data_bound_biased_cov_est, data_bound_unbiased_cov_est,True_value, Ns, delta, length_scale)
np.save(f'pess_bound_delta_{delta}_l_{length_scale}.npy', pess_bound)
np.save(f'data_bound_biased_cov_est_delta_{delta}_l_{length_scale}.npy', data_bound_biased_cov_est)
np.save(f'data_bound_unbiased_cov_est_delta_{delta}_l_{length_scale}.npy', data_bound_unbiased_cov_est)
np.save(f'True_value_delta_{delta}_l_{length_scale}.npy', True_value)

delta = 0.05
length_scale= 0.05
Ns = [100,200,500,1000,2000,5000,10000]#,20000,40000]

pess_bound, data_bound_biased_cov_est, data_bound_unbiased_cov_est, True_value = Cov_Est(data_points, Ns, delta, length_scale)
plot_OU2(pess_bound, data_bound_biased_cov_est, data_bound_unbiased_cov_est,True_value, Ns, delta, length_scale)
np.save(f'pess_bound_delta_{delta}_l_{length_scale}.npy', pess_bound)
np.save(f'data_bound_biased_cov_est_delta_{delta}_l_{length_scale}.npy', data_bound_biased_cov_est)
np.save(f'data_bound_unbiased_cov_est_delta_{delta}_l_{length_scale}.npy', data_bound_unbiased_cov_est)
np.save(f'True_value_delta_{delta}_l_{length_scale}.npy', True_value)


delta = 0.05
Ns = [100,200,500,1000,2000,5000,10000]#,20000,40000]
length_scales = [0.05, 0.5, 5]
pess_bounds = []
data_bounds_biased = []
data_bounds_unbiased = []
true_values = []
for l in length_scales:
    pess_bounds.append(np.load(f'pess_bound_delta_{delta}_l_{l}.npy'))
    data_bounds_biased.append(np.load(f'data_bound_biased_cov_est_delta_{delta}_l_{l}.npy'))
    data_bounds_unbiased.append(np.load(f'data_bound_unbiased_cov_est_delta_{delta}_l_{l}.npy'))
    true_values.append(np.load(f'True_value_delta_{delta}_l_{l}.npy'))

# Create a figure with 3 subplots in a row, single-column width (3.25 inches)
fig, axes = plt.subplots(1, 3, figsize=(3.25 * 3, 4.5))  # Adjust height as needed for visibility

# Plot each subplot and collect lines for the legend
lines = []
for i, length_scale in enumerate(length_scales):
    show_ylabel = (i == 0)  # Only show y-axis label on the first subplot
    lines += plot_OU2(axes[i], pess_bounds[i], data_bounds_biased[i], data_bounds_unbiased[i], true_values[i], Ns, delta, length_scale, show_ylabel=show_ylabel)

# Create a common legend
labels = ["Emp. bound (unbiased cov. est.)", "Emp. bound (biased cov. est.)", "Pessimistic bound", "Estimated True value"]
fig.legend(lines[:4], labels, loc='upper center', fontsize=10, ncol=4, frameon=False)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.85])
plt.subplots_adjust(top=0.85)  # Add more space between title and legend
plt.savefig("OU_Exp_different_length_scales_NeurIPS.pdf", format="pdf", dpi=900)
plt.show()