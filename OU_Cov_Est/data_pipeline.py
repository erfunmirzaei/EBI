from  src import OU_process

def make_dataset(configs):
    data_points = OU_process.sample(configs.n_train_points + 1, num_trajectories = configs.n_trajectories)
    return data_points
