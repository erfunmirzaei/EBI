from typing import TypeVar, NamedTuple
from numpy.typing import NDArray

T = TypeVar('T') #Placeholder type

class LinalgDecomposition(NamedTuple):
    values: NDArray
    vectors: NDArray

class Dataset(NamedTuple):
    X: NDArray
    Y: NDArray 

class MDSimulation(NamedTuple):
    dataset: Dataset
    MD_spectrum: LinalgDecomposition

class Statistic(NamedTuple):
    mean: NDArray
    std: NDArray

class SpectralData(NamedTuple):
    configs: dict
    training_sizes: NDArray
    x: NDArray #Domain discretization
    FEM_spectrum: LinalgDecomposition
    MD_spectrum: LinalgDecomposition
    evalues_err: Statistic
    efuncs_err: Statistic
    metric_distortion: Statistic
    bias: Statistic
    total_bias: Statistic
