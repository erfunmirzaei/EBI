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

import numpy as np
import os
import torch
import random
import torch.nn as nn
import lightning
from typing import Optional, NamedTuple
from kooplearn.abc import TrainableFeatureMap
import logging
from utils import Metrics


#  Define the oracle network
# Setting up the architecture
class CNNEncoder(nn.Module):
    def __init__(self, num_classes, configs):
        super(CNNEncoder, self).__init__()
        # Set the seed
        random.seed(configs.rng_seed)
        np.random.seed(configs.rng_seed)
        torch.manual_seed(configs.rng_seed)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=configs.conv1_in_channels,
                out_channels=configs.conv1_out_channels,  
                kernel_size=configs.conv1_kernel_size,  
                stride=configs.conv1_stride,  
                padding=configs.conv1_padding,  
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=configs.maxpool1_kernel_size),  
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(configs.conv1_out_channels, configs.conv2_out_channels, configs.conv2_kernel_size, configs.conv2_stride, configs.conv2_padding),  
            nn.ReLU(),
            nn.MaxPool2d(configs.maxpool2_kernel_size),  
        )
        # Fully connected layer, output num_classes classes
        self.out = nn.Sequential(
            nn.Linear(configs.fc_input_size, num_classes)  
        )
        torch.nn.init.orthogonal_(self.out[0].weight)

    def forward(self, X):
        if X.dim() == 3:
            X = X.unsqueeze(1)  # Add a channel dimension if needed
        X = self.conv1(X)
        X = self.conv2(X)
        # Flatten the output of conv2
        X = X.view(X.size(0), -1)
        output = self.out(X)
        return output
    
    def encode(self, X):
        if X.dim() == 3:
            X = X.unsqueeze(1)  # Add a channel dimension if needed
        X = self.conv1(X)
        X = self.conv2(X)
        # Flatten the output of conv2
        X = X.view(X.size(0), -1)
        output = self.out(X)
        
        return output

# Following kooplearn implementations, we define a Pytorch Lightning module and then wrap it in a TrainableFeatureMap
class ClassifierModule(lightning.LightningModule):
    def __init__(self, num_classes: int, learning_rate: float, configs):
        super().__init__()
        self.num_classes = num_classes
        self.configs = configs
        self.encoder = CNNEncoder(num_classes=num_classes, configs= self.configs)

        self.learning_rate = learning_rate
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def on_fit_start(self):
        self.metrics = Metrics([], [], [], [])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        output = self.encoder(images)
        # TODO: Check if the following line is better than the above line
        # output = self.encoder.encode(images)

        loss = self.loss_fn(output, labels)
        with torch.no_grad():
            pred_labels = output.argmax(dim=1)
            accuracy = (pred_labels == labels).float().mean()

        #Log metrics
        self.metrics.train_acc.append(accuracy.item())
        self.metrics.train_steps.append(self.global_step)

        return {'loss': loss, 'train/accuracy': accuracy}

    def validation_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        output = self.encoder(images)
        pred_labels = output.argmax(dim=1)
        accuracy = (pred_labels == labels).float().mean() # Scalar

        self.metrics.val_acc.append(accuracy.item())
        self.metrics.val_steps.append(self.global_step)

        return {'val/accuracy': accuracy}

class ClassifierFeatureMap(TrainableFeatureMap):
    def __init__(
                self,
                configs,
                num_classes: int,
                learning_rate: float,
                trainer: lightning.Trainer,
                seed: Optional[int] = None
                ):
        
        self.configs = configs
        #Set rng seed
        lightning.seed_everything(seed)
        self.seed = seed
        self.lightning_module = ClassifierModule(num_classes, learning_rate, configs)

        #Init trainer
        self.lightning_trainer = trainer
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def lookback_len(self) -> int:
        return 1 #Hardcoding it here, as we are not using lookback windows

    #Not tested
    def save(self, path: os.PathLike):
        raise NotImplementedError

    #Not tested
    @classmethod
    def load(cls, path: os.PathLike):
       raise NotImplementedError

    def fit(self, **trainer_fit_kwargs: dict):
        if 'model' in trainer_fit_kwargs:
            logging.warn(f"The 'model' keyword should not be specified in trainer_fit_kwargs. The provided model '{trainer_fit_kwargs['model']}' is ignored.")
            trainer_fit_kwargs = trainer_fit_kwargs.copy()
            del trainer_fit_kwargs['model']
        self.lightning_trainer.fit(model=self.lightning_module, **trainer_fit_kwargs)
        self._is_fitted = True

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = torch.from_numpy(X.copy(order="C")).float()
        self.lightning_module.eval()
        with torch.no_grad():
            embedded_X = self.lightning_module.encoder(
                X.to(self.lightning_module.device)
            )
            embedded_X = embedded_X.detach().cpu().numpy()
        return embedded_X
