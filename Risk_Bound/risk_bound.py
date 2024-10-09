import numpy as np
from tqdm import tqdm
from corr_est_cov_est import biased_covariance_estimator, unbiased_covariance_estimator
from sklearn.gaussian_process.kernels import RBF
from kooplearn.models import Linear, Nonlinear, Kernel
from kooplearn.data import traj_to_contexts

def risk_bound_N_OU(train_dataset, val_dataset, test_dataset, Ns, lamdas, length_scale, configs):
    
    """
    Compute the risk bounds for different values of N

    """
    n_0 = len(Ns)
    emp_risk = np.empty((n_0, configs.n_repits))
    risk_bound = np.empty((n_0, configs.n_repits))
    test_emp_risk = np.empty((n_0, configs.n_repits))
    for i in tqdm(range(configs.n_repits)):    
        for j in range(len(Ns)):
            n = Ns[j]
            X_tr = train_dataset[0:n+1][:,i]
            X_tr = X_tr.reshape(X_tr.shape[0], -1)
            train_data = traj_to_contexts(X_tr, backend='numpy')
            
            X_val = val_dataset[:,i]
            X_val = X_val.reshape(X_val.shape[0], -1)
            val_data = traj_to_contexts(X_val, backend='numpy')
            
            X_ts = test_dataset[:,i]
            X_ts = X_ts.reshape(X_ts.shape[0], -1)
            test_data = traj_to_contexts(X_ts, backend='numpy')


            # Compute the empirical risk
            lamda = lamdas[j]['tikhonov_reg']
            r = configs.rank
            c_h = 1
            # RRR Estimator
            kernel_model = Kernel(RBF(length_scale= length_scale), reduced_rank = configs.reduced_rank, rank = r, tikhonov_reg = lamda).fit(train_data)
            # S_hat_Star = np.conj(kernel_model.kernel_X).T
            # Z_hat = kernel_model.kernel_Y
            K = kernel_model.kernel_X / n
            L = kernel_model.kernel_Y / n
            U = kernel_model.U
            V = kernel_model.V
            norm_est = np.trace(U.T @ K @ U @ V.T @ L @ V)
            for tau in range(1,n):
                # TODO: This is for OU process only. You can change this to any other process
                if (n / tau) % 2 == 0: 
                    if configs.delta/(2*norm_est) >= 2*(n/(2*tau) - 1)*np.exp(-(np.exp(1) -  1)/np.exp(1)*tau):
                        min_tau = tau
                        break
            
            tau = min_tau 
            beta_coeff = np.exp((1/np.exp(1) - 1) *tau)
            beta_coeff_prime = np.exp((1/np.exp(1) - 1) *(tau -1))
            m = n / (2*tau)
            l_tau = np.log(12/((configs.delta/(2*norm_est))- 2*(m-1)*beta_coeff))
            L_tau = np.log(12/((configs.delta/(2*norm_est)) - 2*(m-1)*beta_coeff_prime))
            # Compute the empirical risk for train data and test data
            emp_risk[j,i] = kernel_model.risk(train_data)
            test_emp_risk[j,i] = kernel_model.risk(test_data)

            # Compute the kernel matrix and the biased and unbiased covariance estimator
            kernel_matrix = kernel_model.kernel_X
            biased_cov_est = biased_covariance_estimator(kernel_matrix, tau)
            unbiased_cov_est = unbiased_covariance_estimator(kernel_matrix, tau)

            # TODO: How do you choose lamda?? Excess risk bounds
            T_hat = kernel_model.kernel_YX
            biased_cross_cov_est = biased_covariance_estimator(T_hat, tau)
            unbiased_cross_cov_est = unbiased_covariance_estimator(T_hat, tau)
            # Compute the Variance trace(D- D_hat) 
            diag_elss = np.diagonal(kernel_matrix)
            diag_elss = diag_elss.reshape(int(2*m), tau)
            Ys = diag_elss.mean(axis = 1)
            V_D = 2*np.var(Ys, ddof=1)

            First_term = ((128*(norm_est**2)*c_h*tau)/(3*n))*l_tau 
            Second_term = ((128*np.sqrt(r)*norm_est*c_h*tau)/(3*n))*L_tau 
            Third_term = ((7*c_h)/(3*(m-1)))*l_tau
            Fourth_term = np.sqrt(((2*l_tau + 1)*32*(norm_est**4)*tau*biased_cov_est)/n)
            Fifth_term = np.sqrt(((2*L_tau + 1)*8*r*(norm_est**2)*tau*biased_cross_cov_est)/n)
            Sixth_term = np.sqrt((2*V_D*tau*l_tau)/n)
            risk_bound[j,i] = First_term + Second_term + Third_term + Fourth_term + Fifth_term + Sixth_term 
            
    return emp_risk, risk_bound, test_emp_risk