import numpy as np
from tqdm import tqdm
from corr_est_cov_est import biased_covariance_estimator, unbiased_covariance_estimator, sum_diagonals, sum_off_diagonals
from utils import get_divisors
from sklearn.gaussian_process.kernels import RBF
from src import OU_process 
from kooplearn.models import Linear, Nonlinear, Kernel


def risk_bound_N(data_points, Ns, lamda, length_scale, configs):
    """
    Compute the risk bounds for different values of N

    """
    n_0 = len(Ns)
    emp_risk = np.empty((n_0, configs.n_repits))
    risk_bound = np.empty((n_0, configs.n_repits))
    for i in tqdm(range(configs.n_repits)):    
        X = data_points[0:Ns[-1]][:,i]
        X = X.reshape(X.shape[0], -1)
        for j in range(len(Ns)):
            n = Ns[j]
            X_n = X[0:n]

            for tau in range(1,n):
                if configs.delta >= 2*(n/(2*tau) - 1)*np.exp(-(np.exp(1) -  1)/np.exp(1)*tau) and (n / tau) % 2 == 0 :
                    min_tau = tau
                    break
            
            tau = min_tau 
            beta_coeff = np.exp((1/np.exp(1) - 1) *tau)
            m = n / (2*tau)

            # Compute the empirical risk
            
            kernel_model = Kernel(RBF(length_scale= length_scale), reduced_rank = configs.reduced_rank, rank = configs.rank, tikhonov_reg = lamda).fit(train_data)
            emp_risk[j,i] = kernel_model.score(X_n)

    return emp_risk, risk_bound