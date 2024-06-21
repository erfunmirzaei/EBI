#Misc
import sys
sys.path.append("../../")
#Torch
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import random
import torch.nn as nn
#Numpy-Matplotlib-tqdm
import numpy as np

_train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
)
_test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)

# print(len(_train_data.targets))
def build_sequential_data(targets=_train_data.targets, num_classes=10):
    assert num_classes <= 10
    keys = np.arange(num_classes)
    vals = [[] for _ in range(num_classes)]
    digit_indices_dict = dict(zip(keys, vals))
    _inserted_indices = 0
    for _idx in range(len(targets)):
        idx_label = targets[_idx].item()
        if idx_label < num_classes:  
            digit_indices_dict[idx_label].append(_idx)
            _inserted_indices += 1
    data = []
    _catch_exception = False
    # print(_idx)
    while not _catch_exception:
        try:
            _target = _idx % num_classes   
            data.append(digit_indices_dict[_target][-1])
            digit_indices_dict[_target].pop()
            _idx += 1          
        except Exception as e:
            break
    return np.array(data)

data = _train_data.data.float().numpy()

# num_classes = 10
# d = 28
# perm_data = build_sequential_data(num_classes=num_classes)
# dataset = data[perm_data]


# # Plot the image
# plt.imshow(dataset[3], cmap='gray')
# # plt.title(f'MNIST Image Label: {y_train[image_index]}')
# plt.axis('off')
# plt.show()