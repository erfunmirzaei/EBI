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
import numpy as np
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import lightning
from kooplearn.data import traj_to_contexts
from kooplearn.models import Linear, Nonlinear, Kernel
from sklearn.gaussian_process.kernels import RBF
from tqdm import tqdm
from sklearn.model_selection import  ParameterGrid
from oracle_net import ClassifierFeatureMap

# Load configs
main_path = Path(__file__).parent
data_path = main_path / "__data__"
noisy_data_path = main_path / "__data__Noisy"
configs = ml_confs.from_file(main_path / "configs.yaml")
device = 'gpu' if torch.cuda.is_available() else 'cpu'

# Set the seed
random.seed(configs.rng_seed)
np.random.seed(configs.rng_seed)
torch.manual_seed(configs.rng_seed)

Ns = np.arange(configs.n_train_first, configs.train_samples, configs.n_train_step) # Ns = [500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000]
n_0 = len(Ns)
delta = configs.delta
hparam_tuning = True 

# Gaussian_RRR_length_scale = 784
# Gausian_RRR_tikhonov_reg = 1e-7
# CNN_RRR_tikhonov_reg = 1e-7

# Hyperparameters tuning for the Gaussian RRR
print("Hyperparameter tuning for the Gaussian RRR model")
length_scales = np.geomspace(1e-8, 1e3, 2)
tikhonov_regs = np.geomspace(1e-8, 1e-1, 2)
params = list(
    ParameterGrid(
        {
            'tikhonov_reg': tikhonov_regs,
            'length_scale': length_scales,
        }
    )
)
error = np.empty((len(params), 2, len(Ns)))
for iter_idx, iterate in tqdm(enumerate(params), total=len(params)):
    _err = np.empty((len(Ns), configs.n_repits))
    for i in range(configs.n_repits):
        # Load the dataset
        data_pipeline.main(configs, data_path, noisy_data_path) # Run data download and preprocessing
        ordered_MNIST = load_from_disk(data_path) # Load dataset (torch)
        Noisy_ordered_MNIST = load_from_disk(noisy_data_path) # Load dataset (torch)
        
        for j in range(len(Ns)):
            n = Ns[j]

            for tau in range(1,n):
                if delta >= 2*(n/(2*tau) - 1)*np.exp(-(np.exp(1) -  1)/np.exp(1)*tau) and (n / tau) % 2 == 0 :
                    min_tau = tau
                    break
            tau = min_tau # tau = 25

            oracle_train_dl = DataLoader(ordered_MNIST['train'].select(range(n)), batch_size=configs.oracle_batch_size, shuffle=True)
            oracle_val_dl = DataLoader(ordered_MNIST['validation'].select(range(int(n*configs.val_ratio))), batch_size=len(ordered_MNIST['validation']), shuffle=True)

            trainer_kwargs = {
                'accelerator': device,
                'max_epochs': configs.oracle_epochs,
                'log_every_n_steps': 2,
                'enable_progress_bar': False,
                'devices': 1
            }

            trainer = lightning.Trainer(**trainer_kwargs)

            oracle = ClassifierFeatureMap(configs,configs.classes,configs.oracle_lr,trainer,seed=configs.rng_seed)

            oracle.fit(train_dataloaders=oracle_train_dl, val_dataloaders=oracle_val_dl)
            print(oracle.lightning_module.metrics.val_acc[-1])

            new_train_dataset = Noisy_ordered_MNIST['train'].select(list(range(n)))
            new_val_dataset = Noisy_ordered_MNIST['validation'].select(range(int(n*configs.val_ratio)))

            train_data = traj_to_contexts(new_train_dataset['image'], backend='numpy')
            val_data = traj_to_contexts(new_val_dataset['image'], backend='numpy')
            test_data = traj_to_contexts(Noisy_ordered_MNIST['test']['image'], backend='numpy')
            test_labels = np.take(Noisy_ordered_MNIST['test']['label'], np.squeeze(test_data.idx_map.lookback(1))).detach().cpu().numpy()

            try :
                model = Kernel(RBF(length_scale= iterate['length_scale']), reduced_rank=True, rank = configs.classes, tikhonov_reg = iterate['tikhonov_reg']).fit(train_data)
                _err[j][i] = model.risk(val_data)
            except:
                _err[j][i] = np.inf

    error[iter_idx, 0, :] = np.mean(_err, axis = -1)
    error[iter_idx, 1, :] = np.std(_err, axis = -1)

best_idx = [np.argmin(error[:,0,j]) for j in range(len(Ns))] 
best_params = [params[best_idx[j]] for j in range(len(Ns))]
print(f"Best length scale is {[best_params[j]['length_scale']for j in range(len(Ns))]} and the best tikhonov reg is {[best_params[j]['tikhonov_reg']for j in range(len(Ns))]}")
Errors = [error[best_idx[j],0,j] for j in range(len(Ns))]
print(f"Errors: {Errors}")

print("Testing...")
test_errors = np.empty((len(Ns), configs.n_repits))
for i in range(configs.n_repits):
    # Load the dataset
    data_pipeline.main(configs, data_path, noisy_data_path) # Run data download and preprocessing
    ordered_MNIST = load_from_disk(data_path) # Load dataset (torch)
    Noisy_ordered_MNIST = load_from_disk(noisy_data_path) # Load dataset (torch)
   
    for j in range(len(Ns)):
        n = Ns[j]

        for tau in range(1,n):
            if delta >= 2*(n/(2*tau) - 1)*np.exp(-(np.exp(1) -  1)/np.exp(1)*tau) and (n / tau) % 2 == 0 :
                min_tau = tau
                break
        tau = min_tau # tau = 25

        oracle_train_dl = DataLoader(ordered_MNIST['train'].select(range(n)), batch_size=configs.oracle_batch_size, shuffle=True)
        oracle_val_dl = DataLoader(ordered_MNIST['validation'].select(range(int(n*configs.val_ratio))), batch_size=len(ordered_MNIST['validation']), shuffle=True)

        trainer_kwargs = {
            'accelerator': device,
            'max_epochs': configs.oracle_epochs,
            'log_every_n_steps': 2,
            'enable_progress_bar': False,
            'devices': 1
        }

        trainer = lightning.Trainer(**trainer_kwargs)

        oracle = ClassifierFeatureMap(configs,configs.classes,configs.oracle_lr,trainer,seed=configs.rng_seed)

        oracle.fit(train_dataloaders=oracle_train_dl, val_dataloaders=oracle_val_dl)
        print(oracle.lightning_module.metrics.val_acc[-1])

        new_train_dataset = Noisy_ordered_MNIST['train'].select(list(range(n)))
        new_val_dataset = Noisy_ordered_MNIST['validation'].select(range(int(n*configs.val_ratio)))

        train_data = traj_to_contexts(new_train_dataset['image'], backend='numpy')
        val_data = traj_to_contexts(new_val_dataset['image'], backend='numpy')
        test_data = traj_to_contexts(Noisy_ordered_MNIST['test']['image'], backend='numpy')
        test_labels = np.take(Noisy_ordered_MNIST['test']['label'], np.squeeze(test_data.idx_map.lookback(1))).detach().cpu().numpy()

        model = Kernel(RBF(length_scale= best_params[j]['length_scale']), reduced_rank=True, rank = configs.classes, tikhonov_reg = best_params[j]['tikhonov_reg']).fit(train_data)
        test_errors[j][i] = model.risk(test_data)

print(f"Test errors: {[np.mean(test_errors, axis = -1)[j] for j in range(len(Ns))]}")
Gaussian_RRR_length_scales = np.array([best_params[j]['length_scale'] for j in range(len(Ns))])
Gaussian_RRR_tikhonov_regs = np.array([best_params[j]['tikhonov_reg'] for j in range(len(Ns))])

# Hyperparameters tuning for the Nonlinear reduced-rank regression model
print("Hyperparameter tuning for the CNN RRR model")
tikhonov_regs = np.geomspace(1e-8, 1e-1, 2)
params = list(
    ParameterGrid(
        {
            'tikhonov_reg': tikhonov_regs,
        }
    )
)
error = np.empty((len(params), 2, len(Ns)))
for iter_idx, iterate in tqdm(enumerate(params), total=len(params)):
    _err = np.empty((len(Ns), configs.n_repits))
    for i in range(configs.n_repits):
        # Load the dataset
        data_pipeline.main(configs, data_path, noisy_data_path) # Run data download and preprocessing
        ordered_MNIST = load_from_disk(data_path) # Load dataset (torch)
        Noisy_ordered_MNIST = load_from_disk(noisy_data_path) # Load dataset (torch)
        
        for j in range(len(Ns)):
            n = Ns[j]

            for tau in range(1,n):
                if delta >= 2*(n/(2*tau) - 1)*np.exp(-(np.exp(1) -  1)/np.exp(1)*tau) and (n / tau) % 2 == 0 :
                    min_tau = tau
                    break
            tau = min_tau # tau = 25

            oracle_train_dl = DataLoader(ordered_MNIST['train'].select(range(n)), batch_size=configs.oracle_batch_size, shuffle=True)
            oracle_val_dl = DataLoader(ordered_MNIST['validation'].select(range(int(n*configs.val_ratio))), batch_size=len(ordered_MNIST['validation']), shuffle=True)

            trainer_kwargs = {
                'accelerator': device,
                'max_epochs': configs.oracle_epochs,
                'log_every_n_steps': 2,
                'enable_progress_bar': False,
                'devices': 1
            }

            trainer = lightning.Trainer(**trainer_kwargs)

            oracle = ClassifierFeatureMap(configs,configs.classes,configs.oracle_lr,trainer,seed=configs.rng_seed)

            oracle.fit(train_dataloaders=oracle_train_dl, val_dataloaders=oracle_val_dl)
            print(oracle.lightning_module.metrics.val_acc[-1])

            new_train_dataset = Noisy_ordered_MNIST['train'].select(list(range(n)))
            new_val_dataset = Noisy_ordered_MNIST['validation'].select(range(int(n*configs.val_ratio)))

            train_data = traj_to_contexts(new_train_dataset['image'], backend='numpy')
            val_data = traj_to_contexts(new_val_dataset['image'], backend='numpy')
            test_data = traj_to_contexts(Noisy_ordered_MNIST['test']['image'], backend='numpy')
            test_labels = np.take(Noisy_ordered_MNIST['test']['label'], np.squeeze(test_data.idx_map.lookback(1))).detach().cpu().numpy()

            try :
                model = Nonlinear(oracle, reduced_rank= configs.reduced_rank, rank=configs.classes, tikhonov_reg = iterate['tikhonov_reg']).fit(train_data)
                _err[j][i] = model.risk(val_data)
            except:
                _err[j][i] = np.inf

    error[iter_idx, 0, :] = np.mean(_err, axis = -1)
    error[iter_idx, 1, :] = np.std(_err, axis = -1)

best_idx = [np.argmin(error[:,0,j]) for j in range(len(Ns))] 
best_params = [params[best_idx[j]] for j in range(len(Ns))]
print(f"The best tikhonov regs are {[best_params[j]['tikhonov_reg']for j in range(len(Ns))]}")
Errors = [error[best_idx[j],0,j] for j in range(len(Ns))]
print(f"Errors: {Errors}")

print("Testing...")
test_errors = np.empty((len(Ns), configs.n_repits))
for i in range(configs.n_repits):
    # Load the dataset
    data_pipeline.main(configs, data_path, noisy_data_path) # Run data download and preprocessing
    ordered_MNIST = load_from_disk(data_path) # Load dataset (torch)
    Noisy_ordered_MNIST = load_from_disk(noisy_data_path) # Load dataset (torch)
   
    for j in range(len(Ns)):
        n = Ns[j]

        for tau in range(1,n):
            if delta >= 2*(n/(2*tau) - 1)*np.exp(-(np.exp(1) -  1)/np.exp(1)*tau) and (n / tau) % 2 == 0 :
                min_tau = tau
                break
        tau = min_tau # tau = 25

        oracle_train_dl = DataLoader(ordered_MNIST['train'].select(range(n)), batch_size=configs.oracle_batch_size, shuffle=True)
        oracle_val_dl = DataLoader(ordered_MNIST['validation'].select(range(int(n*configs.val_ratio))), batch_size=len(ordered_MNIST['validation']), shuffle=True)

        trainer_kwargs = {
            'accelerator': device,
            'max_epochs': configs.oracle_epochs,
            'log_every_n_steps': 2,
            'enable_progress_bar': False,
            'devices': 1
        }

        trainer = lightning.Trainer(**trainer_kwargs)

        oracle = ClassifierFeatureMap(configs,configs.classes,configs.oracle_lr,trainer,seed=configs.rng_seed)

        oracle.fit(train_dataloaders=oracle_train_dl, val_dataloaders=oracle_val_dl)
        print(oracle.lightning_module.metrics.val_acc[-1])

        new_train_dataset = Noisy_ordered_MNIST['train'].select(list(range(n)))
        new_val_dataset = Noisy_ordered_MNIST['validation'].select(range(int(n*configs.val_ratio)))

        train_data = traj_to_contexts(new_train_dataset['image'], backend='numpy')
        val_data = traj_to_contexts(new_val_dataset['image'], backend='numpy')
        test_data = traj_to_contexts(Noisy_ordered_MNIST['test']['image'], backend='numpy')
        test_labels = np.take(Noisy_ordered_MNIST['test']['label'], np.squeeze(test_data.idx_map.lookback(1))).detach().cpu().numpy()

        model = Nonlinear(oracle, reduced_rank= configs.reduced_rank, rank=configs.classes, tikhonov_reg = best_params[j]['tikhonov_reg']).fit(train_data)
        test_errors[j][i] = model.risk(test_data)

print(f"Test errors: {[np.mean(test_errors, axis = -1)[j] for j in range(len(Ns))]}")
CNN_RRR_tikhonov_regs = np.array([best_params[j]['tikhonov_reg'] for j in range(len(Ns))])
# Save the hyperparameters
Path(str(main_path)).mkdir(parents=True, exist_ok=True)

np.save(str(main_path) +'/results/Gaussian_RRR_length_scales.npy', Gaussian_RRR_length_scales)
np.save(str(main_path) +'/results/Gaussian_RRR_tikhonov_regs.npy', Gaussian_RRR_tikhonov_regs)
np.save(str(main_path) +'/results/CNN_RRR_tikhonov_regs.npy', CNN_RRR_tikhonov_regs)