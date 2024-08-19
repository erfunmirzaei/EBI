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

import data_pipeline
import ml_confs
from datasets import load_from_disk
from sklearn.gaussian_process.kernels import RBF
from kooplearn.models import Linear, Nonlinear, Kernel
import numpy as np
import random
import torch
from kooplearn.data import traj_to_contexts
from tqdm import tqdm
from sklearn.model_selection import  ParameterGrid
from pathlib import Path

# Load configs
main_path = Path(__file__).parent
configs = ml_confs.from_file(main_path / 'configs.yaml') 
device = 'gpu' if torch.cuda.is_available() else 'cpu'

# Set the seed
random.seed(configs.rng_seed)
np.random.seed(configs.rng_seed)
torch.manual_seed(configs.rng_seed)

# Define the parameter grid
tikhonov_regs = np.geomspace(1e-8, 1e-1, 20)
length_scales = np.geomspace(1e-8, 1e2, 20)
params = list(
    ParameterGrid(
        {
            'tikhonov_reg': tikhonov_regs,
            'length_scale': length_scales,
        }
    )
)

def parameter_tuning(params, configs):
    """
    Perform parameter tuning for the kernel model on the Noisy Ordered MNIST dataset
    """

    # Run data pipeline
    data_pipeline.main(configs) # Run data download and preprocessing
    ordered_MNIST = load_from_disk('__data__') # Load dataset (torch)
    Noisy_ordered_MNIST = load_from_disk('__data__Noisy') # Load dataset (torch)

    n = configs.train_samples
    new_train_dataset = Noisy_ordered_MNIST['train'].select(list(range(n)))
    new_val_dataset = Noisy_ordered_MNIST['validation'].select(range(int(n*configs.val_ratio)))
    train_data = traj_to_contexts(new_train_dataset['image'], backend='numpy')
    val_data = traj_to_contexts(new_val_dataset['image'], backend='numpy')

    test_data = traj_to_contexts(Noisy_ordered_MNIST['test']['image'], backend='numpy')
    test_labels = np.take(Noisy_ordered_MNIST['test']['label'], np.squeeze(test_data.idx_map.lookback(1))).detach().cpu().numpy()

    error = np.empty((len(params), 2))
    for iter_idx, iterate in tqdm(enumerate(params), total=len(params)):
        _err = []
        for i in range(configs.n_repits):

            try :
                model = Kernel(RBF(length_scale= iterate['length_scale']), reduced_rank=True, rank = configs.classes, tikhonov_reg = iterate['tikhonov_reg']).fit(train_data)
                _err.append(model.risk(val_data))
            except:
                _err.append(np.inf)

        _err = np.array(_err)
        error[iter_idx, 0] = np.mean(_err)
        error[iter_idx, 1] = np.std(_err)

    best_idx = np.argmin(error[:,0])
    best_params = params[best_idx]
    print(f"Best length scale is {best_params['length_scale']} and the best tikhonov reg is {best_params['tikhonov_reg']}")
    print(f"Error: {error[best_idx]}")
    print("Testing...")
    model = Kernel(RBF(length_scale= best_params['length_scale']), reduced_rank=True, rank = configs.classes, tikhonov_reg = best_params['tikhonov_reg']).fit(train_data)
    test_error = model.risk(test_data)
    print(f"Test error: {test_error}")

    return best_params, error, test_error

best_params, error, test_error = parameter_tuning(params, configs)