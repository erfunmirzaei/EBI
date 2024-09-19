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
from tqdm import tqdm
from transfer_op import fit_transfer_operator_models
from oracle_net import ClassifierFeatureMap
from normalized_corr_est_cov_est import biased_covariance_estimator, unbiased_covariance_estimator


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
Ns = np.arange(configs.n_train_first, int((1-configs.val_ratio) * configs.train_samples), configs.n_train_step) # Ns = [500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000]
n_0 = len(Ns)
delta = configs.delta

biased_cov_ests = {
                   'Gaussian_RRR':np.empty((n_0, configs.n_repits)),
                   'Classifier_Baseline':np.empty((n_0, configs.n_repits)),
                   'DPNets':np.empty((n_0, configs.n_repits))}

unbiased_cov_ests = {
                   'Gaussian_RRR':np.empty((n_0, configs.n_repits)),
                   'Classifier_Baseline':np.empty((n_0, configs.n_repits)),
                   'DPNets':np.empty((n_0, configs.n_repits))}
ordered_acc = {
                    'Gaussian_RRR':np.empty((n_0, configs.n_repits, configs.eval_up_to_t)),
                    'Classifier_Baseline':np.empty((n_0, configs.n_repits, configs.eval_up_to_t)),
                    'DPNets':np.empty((n_0, configs.n_repits, configs.eval_up_to_t))}

pred_labels = {
                    'Gaussian_RRR':np.empty((n_0, configs.n_repits, configs.eval_up_to_t)),
                    'Classifier_Baseline':np.empty((n_0, configs.n_repits, configs.eval_up_to_t)),
                    'DPNets':np.empty((n_0, configs.n_repits, configs.eval_up_to_t))}

pred_images = {
                    'Gaussian_RRR':np.empty((n_0, configs.n_repits, configs.eval_up_to_t, configs.oracle_input_size1, configs.oracle_input_size2)),
                    'Classifier_Baseline':np.empty((n_0, configs.n_repits, configs.eval_up_to_t, configs.oracle_input_size1, configs.oracle_input_size2)),
                    'DPNets':np.empty((n_0, configs.n_repits, configs.eval_up_to_t, configs.oracle_input_size1, configs.oracle_input_size2))}

true_labels = np.empty((configs.n_repits, configs.test_samples - 1))
true_images = np.empty((configs.n_repits, configs.test_samples, configs.conv1_in_channels, configs.oracle_input_size1, configs.oracle_input_size2))

fn_i = {
                    'Gaussian_RRR':np.empty((n_0, configs.n_repits, configs.test_samples - 1)),
                    'Classifier_Baseline':np.empty((n_0, configs.n_repits, configs.test_samples - 1)),
                    'DPNets':np.empty((n_0, configs.n_repits, configs.test_samples - 1))}
fn_j = {
                    'Gaussian_RRR':np.empty((n_0, configs.n_repits, configs.test_samples - 1)),
                    'Classifier_Baseline':np.empty((n_0, configs.n_repits, configs.test_samples - 1)),
                    'DPNets':np.empty((n_0, configs.n_repits, configs.test_samples - 1))}

# lower_bound = np.empty((n_0, configs.n_repits))

for i in tqdm(range(configs.n_repits)):
    # Load the dataset
    data_pipeline.main(configs, data_path, noisy_data_path) # Run data download and preprocessing
    ordered_MNIST = load_from_disk(data_path) # Load dataset (torch)
    Noisy_ordered_MNIST = load_from_disk(noisy_data_path) # Load dataset (torch)
    # #TODO: Remove the following line
    # # Check if the dataset is different from the previous one
    # if i > 0:
    #     for key in ordered_MNIST:
    #         if torch.equal(ordered_MNIST[key]['image'], ordered_MNIST_prev[key]['image']) and torch.equal(ordered_MNIST[key]['label'], ordered_MNIST_prev[key]['label']):
    #             print(f"Error: The dataset is the same as the previous one. Repitition {i} is the same as repitition {i-1}")
    #             break

    # ordered_MNIST_prev = ordered_MNIST
    
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

        val_data = traj_to_contexts(new_val_dataset['image'], backend='numpy')
        test_data = traj_to_contexts(Noisy_ordered_MNIST['test']['image'], backend='numpy')
        test_labels = np.take(Noisy_ordered_MNIST['test']['label'], np.squeeze(test_data.idx_map.lookback(1))).detach().cpu().numpy()
        transfer_operator_models, report, C_H, B_H, kernel_matrices = fit_transfer_operator_models(new_train_dataset, oracle, val_data, test_data, test_labels, configs, device)

        true_images[i] = Noisy_ordered_MNIST['test']['image']
        true_labels[i] = test_labels
        for model_name in transfer_operator_models:
            biased_cov_ests[model_name][j][i] = biased_covariance_estimator(kernel_matrices[model_name], tau= tau, c_h=C_H[model_name], b_h = B_H[model_name])
            unbiased_cov_ests[model_name][j][i] = unbiased_covariance_estimator(kernel_matrices[model_name], tau= tau, c_h=C_H[model_name], b_h = B_H[model_name])
            ordered_acc[model_name][j][i] = report[model_name]['accuracy_ordered']
            pred_labels[model_name][j][i] = report[model_name]['label']
            pred_images[model_name][j][i] = report[model_name]['image']
            fn_i[model_name][j][i] = report[model_name]['fn_i']
            fn_j[model_name][j][i] = report[model_name]['fn_j']        

        # lower_bound[j][i] = 1 / tau

# Path("/results").mkdir(parents=True, exist_ok=True)
np.save(str(main_path) + f'/results/true_labels_eta_{configs.eta}.npy', true_labels)
np.save(str(main_path) + f'/results/true_images_eta_{configs.eta}.npy', true_images)
for i, model_name in enumerate(transfer_operator_models):
    model_name = model_name.replace(" ", "")
    np.save(str(main_path) + f'/results/biased_cov_ests_{model_name}_eta_{configs.eta}.npy', biased_cov_ests[model_name])
    np.save(str(main_path) + f'/results/unbiased_cov_ests_{model_name}_eta_{configs.eta}.npy', unbiased_cov_ests[model_name])
    np.save(str(main_path) + f'/results/ordered_acc_{model_name}_eta_{configs.eta}.npy', ordered_acc[model_name])  
    np.save(str(main_path) + f'/results/pred_labels_{model_name}_eta_{configs.eta}.npy', pred_labels[model_name])
    np.save(str(main_path) + f'/results/pred_images_{model_name}_eta_{configs.eta}.npy', pred_images[model_name])
    np.save(str(main_path) + f'/results/fn_i_{model_name}_eta_{configs.eta}.npy', fn_i[model_name])
    np.save(str(main_path) + f'/results/fn_j_{model_name}_eta_{configs.eta}.npy', fn_j[model_name])
