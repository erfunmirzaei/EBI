import numpy as np
import torch
import torch.nn as nn
import lightning
from torch.utils.data import DataLoader
from sklearn.gaussian_process.kernels import RBF
from kooplearn.abc import TrainableFeatureMap, BaseModel
from oracle_net import ClassifierFeatureMap, CNNEncoder, Metrics
from kooplearn.models import Linear, Nonlinear, Kernel
from kooplearn.models.feature_maps.nn import NNFeatureMap
from kooplearn.nn.data import collate_context_dataset
from kooplearn.nn import DPLoss, VAMPLoss
from kooplearn.data import traj_to_contexts
from sklearn.manifold import TSNE
from sklearn.model_selection import  ParameterGrid
from tqdm import tqdm

def evaluate_model(model: BaseModel, test_data, configs, oracle: ClassifierFeatureMap, test_labels: np.ndarray):
    assert model.is_fitted
    report = {
        'accuracy': [],
        'accuracy_ordered':[],
        'label': [],
        'image': [],
        'times': []
    }
    for t in range(1, configs.eval_up_to_t + 1):
        pred = (model.predict(test_data, t=t)).reshape(-1, 28,28) # Shape of the lookforward window
        pred_labels = oracle(pred).argmax(axis=1)
        new_test_labels = np.array(list(test_labels[t:]) + list((test_labels[-t:] + t)% configs.classes))
        accuracy = (pred_labels == new_test_labels).mean()
        accuracy_ordered =  (pred_labels == (test_labels + t)% configs.classes).mean()
        report['accuracy'].append(accuracy)
        report['accuracy_ordered'].append(accuracy_ordered)
        report['image'].append(pred[configs.test_seed_idx])
        report['label'].append(pred_labels[configs.test_seed_idx])
        report['times'].append(t)

        vals, lfuncs, rfuncs = model.eig(eval_right_on=test_data, eval_left_on=test_data)
        # returns the unique values and the index of the first occurrence of a value
        unique_vals, idx_start = np.unique(np.abs(vals), return_index=True)
        vals, lfuncs, rfuncs = vals[idx_start], lfuncs[:, idx_start], rfuncs[:, idx_start]

        fns = lfuncs
        fns = np.column_stack([lfuncs, rfuncs])
        reduced_fns = TSNE(n_components=2, random_state=configs.rng_seed).fit_transform(fns.real)

        report['fn_i'] = reduced_fns[:, 0]
        report['fn_j'] = reduced_fns[:, 1]

    return report


def fit_transfer_operator_models(train_dataset , oracle: ClassifierFeatureMap, val_data:np.ndarray ,test_data: np.ndarray, test_labels: np.ndarray, configs, device: torch.device):
    """
    Fit the transfer operator models

    """
    train_data = traj_to_contexts(train_dataset['image'], backend='numpy')
    n = train_dataset.shape[0]  
    # We will now fit the transfer operator models. We will use the following models:
    transfer_operator_models = {}

    # - Gaussian rank-reduced regression (Gaussian RRR) model

    Gaussian_RRR_length_scale = 784
    Gausian_RRR_tikhonov_reg = 1e-7

    kernel_model = Kernel(RBF(length_scale=Gaussian_RRR_length_scale), reduced_rank = configs.reduced_rank, rank = configs.classes, tikhonov_reg = Gausian_RRR_tikhonov_reg).fit(train_data)
    transfer_operator_models['Gaussian_RRR'] = kernel_model

    # - Nonlinear(CNN encoder) reduced-rank regression model

    CNN_RRR_tikhonov_reg = 1e-7 #TODO: Change it back to without regularization
    
    classifier_model = Nonlinear(oracle, reduced_rank= configs.reduced_rank, rank=configs.classes, tikhonov_reg= CNN_RRR_tikhonov_reg).fit(train_data)
    transfer_operator_models['Classifier_Baseline'] = classifier_model

    # - Nonlinear(DPNets encoder) reduced-rank regression model

    train_dl = DataLoader(train_data, batch_size = configs.dpnet_batch_size, shuffle=True, collate_fn=collate_context_dataset)
    trainer_kwargs = {
        'accelerator': device,
        'devices': 1,
        'max_epochs': configs.dpnet_max_epochs,
        'log_every_n_steps': 3,
        'enable_model_summary': False
    }

    feature_maps = {
        # 'DPNets_Relaxed': {
        #     'loss_fn': DPLoss,
        #     'loss_kwargs': {'relaxed': True, 'metric_deformation': 1, 'center_covariances': False}
        # },
        'DPNets': {
            'loss_fn': DPLoss,
            'loss_kwargs': {'relaxed': configs.dpnet_relaxed, 'metric_deformation': configs.dpnet_metric_deformation, 'center_covariances': configs.dpnet_center_covariances}
        },
        # 'VAMPNets': {
        #     'loss_fn': VAMPLoss,
        #     'loss_kwargs': {'schatten_norm': 2, 'center_covariances': False}
        # },
    }

    for fname, fdict in feature_maps.items():
        print(f"Fitting {fname.replace('_', ' ')}")
        trainer = lightning.Trainer(**trainer_kwargs)
        #Defining the model
        feature_map = NNFeatureMap(
            CNNEncoder,
            fdict['loss_fn'],
            torch.optim.Adam,
            trainer,
            encoder_kwargs={'num_classes': configs.classes, 'configs':configs},
            loss_kwargs=fdict['loss_kwargs'],
            optimizer_kwargs={'lr': configs.dpnet_lr},
            seed=configs.rng_seed
        )
        feature_map.fit(train_dl)

        nn_model = Nonlinear(feature_map, reduced_rank = configs.reduced_rank, rank=configs.classes).fit(train_data)
        transfer_operator_models[fname] = nn_model

    # Evaluate the transfer operator models
    report = {}
    for model_name, model in transfer_operator_models.items():
            print(f"Evaluating {model_name.replace('_', ' ')}")
            report[model_name] = evaluate_model(model, test_data, configs, oracle, test_labels)
    
    C_H = {'Gaussian_RRR':0.0,
        #    'Linear':0.0,
           'Classifier_Baseline':0.0,
           'DPNets':0.0}

    B_H = {'Gaussian_RRR':0.0,
        #    'Linear':0.0,
           'Classifier_Baseline':0.0,
           'DPNets':0.0}
    # Kernel matrices
    kernel_matrices = {}
    fm_linear = train_dataset['image'].numpy().reshape(n, -1)
    # print(fm_linear.shape)
    kernel_matrices['Gaussian_RRR'] = kernel_model._kernel(fm_linear, fm_linear)
    C_H['Gaussian_RRR'] = np.max(kernel_matrices['Gaussian_RRR'])
    # if  max > C_H['Gaussian_RRR']:
    #     C_H['Gaussian_RRR'] = max

    B_H['Gaussian_RRR'] = np.min(kernel_matrices['Gaussian_RRR'])
    # if  min < B_H['Gaussian_RRR']:
    #     B_H['Gaussian_RRR'] = min

    # print(kernel_matrices['Gaussian_RRR'].shape)
    fm_dpnet = feature_map(train_dataset['image'].numpy())
    # print(fm_dpnet.shape)  

    # print(fm_classifier.shape)
    kernel_matrices['DPNets'] = fm_dpnet @ fm_dpnet.T
    C_H['DPNets'] = np.max(kernel_matrices['DPNets'])
    # if  max > C_H['DPNets']:
    #     C_H['DPNets'] = max

    B_H['DPNets'] = np.min(kernel_matrices['DPNets'])
    # if  min < B_H['DPNets']:
    #     B_H['DPNets'] = min

    fm_classifier = oracle(train_dataset['image'].numpy())
    kernel_matrices['Classifier_Baseline'] = fm_classifier @ fm_classifier.T
    C_H['Classifier_Baseline'] = np.max(kernel_matrices['Classifier_Baseline'])
    # if  max > C_H['Classifier_Baseline']:
    #     C_H['Classifier_Baseline'] = max

    B_H['Classifier_Baseline'] = np.min(kernel_matrices['Classifier_Baseline'])
    # if  min < B_H['Classifier_Baseline']:
    #     B_H['Classifier_Baseline'] = min

    # kernel_matrices['Linear'] = fm_linear @ fm_linear.T
    # max = np.max(kernel_matrices['Linear'])
    # if  max > C_H['Linear']:
    #     C_H['Linear'] = max

    return transfer_operator_models, report, C_H, B_H, kernel_matrices