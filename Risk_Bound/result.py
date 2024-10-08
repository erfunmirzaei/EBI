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
import ml_confs
import numpy as np
from utils import plot_risk_bound_N

# Load configs
main_path = Path(__file__).parent
data_path = main_path / "__data__"
configs = ml_confs.from_file(main_path / "configs.yaml")

# First Experiment: Plot the risk bound for different values of N
Ns = [500,1000,2000, 5000, 10000]
length_scales = [0.05, 0.15, 0.25]
lamda = 1e-4

emp_risk = []
risk_bound = []
test_emp_risk = []


for l in length_scales:
    emp_risk.append(np.load(str(main_path) + f'/results/emp_risk_delta_{configs.delta}_l_{l}_reg_{lamda}.npy'))
    risk_bound.append(np.load(str(main_path) + f'/results/risk_bound_delta_{configs.delta}_l_{l}_reg_{lamda}.npy'))
    test_emp_risk.append(np.load(str(main_path) + f'/results/test_emp_risk_delta_{configs.delta}_l_{l}_reg_{lamda}.npy'))
    print(risk_bound[-1])

# Plot the bounds for different values of N
print('Plotting bounds for different values of N')
labels = ["Empirical risk", "Upper Bound", "Lower Bound", "Test Empirical risk"]
plot_risk_bound_N(configs, emp_risk, risk_bound, test_emp_risk, Ns, length_scales, labels)
          