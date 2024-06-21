from kooplearn.kernels import ScalarProduct
import numpy as np
from numpy.typing import NDArray
from scipy.special import eval_hermitenorm, factorial

class HermitePoly(ScalarProduct):
    def __init__(self, rank:int, kind: str = 'good', oversamples: int = 10):
        self.rank = rank
        self.oversamples = oversamples
        assert kind in ['good', 'bad', 'ugly'], "The Hermite Polynomial kernel comes in two flavours: 'good', 'bad' and 'ugly'. The three kinds corresponds to: PCR and RRR working correctly, RRR only working correctly, neither ot the two algorithms working correctly  "
        self.kind = kind
    
    def __feature_map__(self, X:NDArray) -> NDArray:
        assert X.shape[1] == 1, "The NastyHermite kernel is defined only on scalar inputs"
        _X = X[:, 0] 
        truncation = self.rank + self.oversamples
        
        eigfuns = np.zeros((truncation, X.shape[0]), dtype=np.float64)
        for order in range(truncation):
            eigfuns[order] = self.OU_eigfun(_X, order)
        return (self.sqrt_eigvals)*(eigfuns.T)
    
    @property
    def sqrt_eigvals(self):
        truncation = self.rank + self.oversamples
        if self.kind == 'bad':
            alpha = self.rank**-2
            sqrt_eigvals = np.exp(-alpha*np.arange(truncation))
            nasty_sqrt_eigevals = sqrt_eigvals.copy()
            init_idxs = [i for i in range(2*self.rank)]
            fin_idxs =  [i for i in range(self.rank, 2*self.rank)]
            fin_idxs.reverse()
            fin_idxs += [i for i in range(self.rank)]
            nasty_sqrt_eigevals[init_idxs] = nasty_sqrt_eigevals[fin_idxs]
            return nasty_sqrt_eigevals
        elif self.kind == 'ugly':
            alpha = self.rank**2
            sqrt_eigvals = np.exp(-alpha*np.arange(truncation))
            nasty_sqrt_eigevals = sqrt_eigvals.copy()
            init_idxs = [i for i in range(2*self.rank)]
            fin_idxs =  [i for i in range(self.rank, 2*self.rank)]
            fin_idxs.reverse()
            # fin_idxs_tmp = [i for i in range(self.rank)]
            # fin_idxs_tmp.reverse()
            # fin_idxs += fin_idxs_tmp
            fin_idxs += [i for i in range(self.rank)]
            nasty_sqrt_eigevals[init_idxs] = nasty_sqrt_eigevals[fin_idxs]
            return nasty_sqrt_eigevals
        else:
            return np.exp(-0.5*np.arange(truncation))
    
    def OU_eigfun(self, x:NDArray, order:int) -> NDArray:
        return factorial(order)**(-0.5) * eval_hermitenorm(order, x)