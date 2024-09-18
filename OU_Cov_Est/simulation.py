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
import matplotlib.pyplot as plt
import torch
import json
from typing import NamedTuple, Tuple
import numpy as np
from datetime import datetime
from tqdm import tqdm
import pickle
from  src import OU_process
from tqdm import tqdm
from kooplearn.models import Linear, Nonlinear, Kernel
from kooplearn.models.feature_maps.nn import NNFeatureMap
from kooplearn.data import traj_to_contexts
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import  ParameterGrid
import time
CV = True

main_path = Path(__file__).parent
data_path = main_path / "__data__"
configs = ml_confs.from_file(main_path / "configs.yaml")