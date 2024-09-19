import numpy as np
from sklearn.gaussian_process.kernels import RBF
from tqdm import tqdm
from corr_est_cov_est import biased_covariance_estimator, unbiased_covariance_estimator
from utils import get_divisors

def Covariance_Estimation(data_points, n, delta, length_scale, configs):
    for tau in range(1,n):
        if delta >= 2*(n/(2*tau) - 1)*np.exp(-(np.exp(1) -  1)/np.exp(1)*tau) and (n / tau) % 2 == 0 :
            min_tau = tau
            break
    
    gauss_kernel = RBF(length_scale=length_scale)
    c_h = 1
    L = 2 * c_h
    sigma = c_h
    divisors = np.array(list(get_divisors(n)))
    divisors = list(np.sort(divisors[divisors > min_tau]))
    taus = [divisor for divisor in divisors if (n / divisor) % 2 == 0 ]
    n_0 = len(taus)
    data_bound_biased_cov_est = np.empty((n_0, configs.n_repits))
    data_bound_unbiased_cov_est = np.empty((n_0, configs.n_repits))
    pess_bound = np.empty((n_0, configs.n_repits))

    for i in tqdm(range(configs.n_repits)):    
        X = data_points[0:n][:,i]
        X = X.reshape(X.shape[0], -1)
        kernel_matrix = gauss_kernel(X, X)
        # print(kernel_matrix) 
        for j in range(n_0):
            tau = taus[j]
            beta_coeff = np.exp((1/np.exp(1) - 1) *tau)
            m = n / (2*tau)
            # print(delta - 2*(m-1)*beta_coeff)

            l_tau = np.log(4/(delta - 2*(m-1)*beta_coeff))
            
            pess_bound[j][i] = (((2 * L ) / m)  + (2 * sigma)/np.sqrt(m))* l_tau
            
            cov_biased = biased_covariance_estimator(kernel_matrix, tau= tau)
            data_bound_biased_cov_est[j][i] = ((16*c_h)/(3*m))*l_tau + np.sqrt(((2*l_tau + 1)*cov_biased)/m)
            
            cov_unbiased = unbiased_covariance_estimator(kernel_matrix, tau= tau)
            # print(cov_biased, cov_unbiased)
            data_bound_unbiased_cov_est[j][i] = ((13*c_h)/(m))*l_tau + np.sqrt(((2*l_tau + 1)*cov_unbiased)/m)
            # biased_bounds.append(biased_covariance_estimator(kernel_matrix, tau= tau))
            # unbiased_bounds.append(unbiased_covariance_estimator(kernel_matrix, tau= tau))
    
    return pess_bound, data_bound_biased_cov_est, data_bound_unbiased_cov_est, taus