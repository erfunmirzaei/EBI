from typing import TypeVar, NamedTuple
from numpy.typing import NDArray

T = TypeVar('T') #Placeholder type

class LinalgDecomposition(NamedTuple):
    values: NDArray
    vectors: NDArray

class Dataset(NamedTuple):
    X: NDArray
    Y: NDArray 

class Simulation(NamedTuple):
    dataset: Dataset
    eig: LinalgDecomposition

class Statistic(NamedTuple):
    mean: NDArray
    std: NDArray
