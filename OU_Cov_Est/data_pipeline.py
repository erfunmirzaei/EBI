from src import OU_process

def make_dataset(configs):
    data_points = data_points = OU_process.sample(configs.n_train_points + configs.n_val_points + configs.n_test_points, num_trajectories= configs.n_repits)
    return data_points
