from pathlib import Path
import random
import ml_confs
from datasets import DatasetDict, interleave_datasets, load_dataset, Dataset
from  src import OU_process

main_path = Path(__file__).parent
data_path = main_path / "__data__"
configs = ml_confs.from_file(main_path / "configs.yaml")

def make_dataset():
    data_points = OU_process.sample(configs.n_train_points + 1, num_trajectories = configs.n_trajectories)
    return data_points
