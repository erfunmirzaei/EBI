import numpy as np
from tqdm import tqdm
from corr_est_cov_est import biased_covariance_estimator, unbiased_covariance_estimator, sum_diagonals, sum_off_diagonals
from utils import get_divisors
from sklearn.gaussian_process.kernels import RBF
from src import OU_process 

def Covariance_Estimation_tau(data_points, n, delta, length_scale, configs):
    for tau in range(1,n):
        if delta >= 2*(n/(2*tau) - 1)*np.exp(-(np.exp(1) -  1)/np.exp(1)*tau): #and (n / tau) % 2 == 0 :
            min_tau = tau
            break
    
    gauss_kernel = RBF(length_scale=length_scale)
    divisors = np.array(list(get_divisors(n)))
    divisors = list(np.sort(divisors[divisors > min_tau]))
    taus = [divisor for divisor in divisors if (n / divisor) % 2 == 0 ]
    n_0 = len(taus)
    M_bound = np.empty((n_0, configs.n_repits))
    M_emp_bound_biased_cov_est = np.empty((n_0, configs.n_repits))
    M_emp_bound_unbiased_cov_est = np.empty((n_0, configs.n_repits))
    Pinelis_bound = np.empty((n_0, configs.n_repits))
    Pinelis_emp_bound_biased_cov_est = np.empty((n_0, configs.n_repits))
    Pinelis_emp_bound_unbiased_cov_est = np.empty((n_0, configs.n_repits))
    c_h = 1
    L = 2 * c_h
    sigma = c_h 

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
            ltau = np.log(8/(delta - 2*(m-1)*beta_coeff))
            l_tau = np.log(4/(delta - 2*(m-1)*beta_coeff))
            L_tau = np.log(2/(delta - 2*(m-1)*beta_coeff))

            cov_biased = biased_covariance_estimator(kernel_matrix, tau= tau)
            cov_unbiased = unbiased_covariance_estimator(kernel_matrix, tau= tau)
            # print(cov_biased, cov_unbiased)
            M_bound[j][i] = (4*c_h)/(3*m)*L_tau + np.sqrt(((2*L_tau + 1)*2*(sigma**2))/m) # Apply lemma 2 to Theorem 4
            M_emp_bound_biased_cov_est[j][i] = ((16*c_h)/(3*m))*l_tau + np.sqrt(((2*l_tau + 1)*cov_biased)/m) # Theorem 2
            M_emp_bound_unbiased_cov_est[j][i] = ((11*c_h)/(m))*l_tau + np.sqrt(((2*l_tau + 1)*cov_unbiased)/m) # Theorem 3

            Pinelis_bound[j][i] = (((4 * c_h ) / m)  + (2 * sigma)/np.sqrt(m))* l_tau # Apply lemma 2 to Pinelis
            Pinelis_emp_bound_biased_cov_est[j][i] = ((2*c_h)/m)*ltau*(2+np.sqrt(2*l_tau)) + 2*ltau*np.sqrt(cov_biased/m) # Apply cov biased estimation to pinelis and then lemma 2
            Pinelis_emp_bound_unbiased_cov_est[j][i] = ((4*c_h)/m)*ltau*(1+np.sqrt(4*l_tau)) + 2*ltau*np.sqrt(cov_unbiased/m)

    return Pinelis_bound, Pinelis_emp_bound_biased_cov_est, Pinelis_emp_bound_unbiased_cov_est, M_bound, M_emp_bound_biased_cov_est, M_emp_bound_unbiased_cov_est, taus

def Cov_Est_N(data_points, Ns, delta, length_scale, configs):
    gauss_kernel = RBF(length_scale=length_scale)
    n_0 = len(Ns)
    M_bound = np.empty((n_0, configs.n_repits))
    M_emp_bound_biased_cov_est = np.empty((n_0, configs.n_repits))
    M_emp_bound_unbiased_cov_est = np.empty((n_0, configs.n_repits))
    Pinelis_bound = np.empty((n_0, configs.n_repits))
    Pinelis_emp_bound_biased_cov_est = np.empty((n_0, configs.n_repits))
    Pinelis_emp_bound_unbiased_cov_est = np.empty((n_0, configs.n_repits))
    True_value = np.empty((n_0, configs.n_repits))
    # Cov_biased = np.empty((n_0, configs.n_repits))
    # Cov_unbiased = np.empty((n_0, configs.n_repits))
    c_h = 1
    L = 2 * c_h
    sigma = c_h 
    

    for i in tqdm(range(configs.n_repits)):    
        X = data_points[0:Ns[-1]][:,i]
        X = X.reshape(X.shape[0], -1)
        # print(X.shape)
        kernel_Matrix = gauss_kernel(X, X)
        # print("SALAVAT")   
        trace_C2 = np.sqrt(1/(1+4/(length_scale*length_scale)))
        n_sample_est_tr = configs.n_sample_est_tr 
        n_repits_est_tr = configs.n_repits_est_tr
        data_points2 = OU_process.sample(n_sample_est_tr, num_trajectories= n_repits_est_tr)
            
        for j in range(len(Ns)):
            n = Ns[j]
            kernel_matrix = kernel_Matrix[0:n,0:n]
            
            trace_C_hat2 = (np.linalg.norm(kernel_matrix)**2)/(n*n)
            trace_C_C_hat = 0
            for t in range(n_repits_est_tr):
                X_prime = data_points2[:,t]
                X_prime = X_prime.reshape(X_prime.shape[0], -1) 
                c = (np.linalg.norm(gauss_kernel(X_prime,X[0:n]))**2)/(n_sample_est_tr*n*n_repits_est_tr)
                trace_C_C_hat += c
            
            # print( trace_C2, - (2*trace_C_C_hat),trace_C_hat2)
            for tau in range(1,n):
                if delta >= 2*(n/(2*tau) - 1)*np.exp(-(np.exp(1) -  1)/np.exp(1)*tau) and (n / tau) % 2 == 0 :
                    min_tau = tau
                    break
            
            tau = min_tau 
            beta_coeff = np.exp((1/np.exp(1) - 1) *tau)
            m = n / (2*tau)
            # print(delta - 2*(m-1)*beta_coeff)
            ltau = np.log(8/(delta - 2*(m-1)*beta_coeff))
            l_tau = np.log(4/(delta - 2*(m-1)*beta_coeff))
            L_tau = np.log(2/(delta - 2*(m-1)*beta_coeff))

            cov_biased = biased_covariance_estimator(kernel_matrix, tau= tau)
            cov_unbiased = unbiased_covariance_estimator(kernel_matrix, tau= tau)
            # print(cov_biased, cov_unbiased)

            # pess_bound[j][i] = (((2 * L ) / m)  + (2 * sigma)/np.sqrt(m))* l_tau            
            # data_bound_biased_cov_est[j][i] = ((16*c_h)/(3*m))*l_tau + np.sqrt(((2*l_tau + 1)*cov_biased)/m)
            # data_bound_unbiased_cov_est[j][i] = ((13*c_h)/(m))*l_tau + np.sqrt(((2*l_tau + 1)*cov_unbiased)/m)
            # Cov_biased[j][i] = cov_biased
            # Cov_unbiased[j][i] = cov_unbiased
            M_bound[j][i] = (4*c_h)/(3*m)*L_tau + np.sqrt(((2*L_tau + 1)*2*(sigma**2))/m) # Apply lemma 2 to Theorem 4
            M_emp_bound_biased_cov_est[j][i] = ((16*c_h)/(3*m))*l_tau + np.sqrt(((2*l_tau + 1)*cov_biased)/m) # Theorem 2
            M_emp_bound_unbiased_cov_est[j][i] = ((11*c_h)/(m))*l_tau + np.sqrt(((2*l_tau + 1)*cov_unbiased)/m) # Theorem 3

            Pinelis_bound[j][i] = (((4 * c_h ) / m)  + (2 * sigma)/np.sqrt(m))* l_tau # Apply lemma 2 to Pinelis
            Pinelis_emp_bound_biased_cov_est[j][i] = ((2*c_h)/m)*ltau*(2+np.sqrt(2*l_tau)) + 2*ltau*np.sqrt(cov_biased/m) # Apply cov biased estimation to pinelis and then lemma 2
            Pinelis_emp_bound_unbiased_cov_est[j][i] = ((4*c_h)/m)*ltau*(1+np.sqrt(4*l_tau)) + 2*ltau*np.sqrt(cov_unbiased/m)

            True_value[j][i] = np.sqrt(abs(trace_C2 - (2*trace_C_C_hat) + trace_C_hat2))
        
        del kernel_Matrix
        del X
        del X_prime
        del cov_biased
        del cov_unbiased

    return Pinelis_bound, Pinelis_emp_bound_biased_cov_est, Pinelis_emp_bound_unbiased_cov_est, M_bound, M_emp_bound_biased_cov_est, M_emp_bound_unbiased_cov_est, True_value

def Cov_Est_N2(data_points, Ns, delta, length_scale, configs):
    gauss_kernel = RBF(length_scale=length_scale)
    n_0 = len(Ns)
    M_bound = np.empty((n_0, configs.n_repits))
    M_emp_bound_biased_cov_est = np.empty((n_0, configs.n_repits))
    M_emp_bound_unbiased_cov_est = np.empty((n_0, configs.n_repits))
    Pinelis_bound = np.empty((n_0, configs.n_repits))
    Pinelis_emp_bound_biased_cov_est = np.empty((n_0, configs.n_repits))
    Pinelis_emp_bound_unbiased_cov_est = np.empty((n_0, configs.n_repits))
    True_value = np.empty((n_0, configs.n_repits))
    # Cov_biased = np.empty((n_0, configs.n_repits))
    # Cov_unbiased = np.empty((n_0, configs.n_repits))

    c_h = 1
    L = 2 * c_h
    sigma = c_h 
    

    for i in tqdm(range(configs.n_repits)):    

        trace_C2 = np.sqrt(1/(1+4/(length_scale*length_scale)))
        n_sample_est_tr = configs.n_sample_est_tr 
        n_repits_est_tr = configs.n_repits_est_tr
        data_points2 = OU_process.sample(n_sample_est_tr, num_trajectories= n_repits_est_tr)
            
        for j in range(len(Ns)):
            n = Ns[j]
            for tau in range(1,n):
                if delta >= 2*(n/(2*tau) - 1)*np.exp(-(np.exp(1) -  1)/np.exp(1)*tau) and (n / tau) % 2 == 0: #and (n_sample_est_tr / tau) % 2 == 0:
                    min_tau = tau
                    break
            
            tau = min_tau 
            m = n / (2*tau)
            if configs.n_sample_thresh >= n:
                X = data_points[0:Ns[j]][:,i]
                X = X.reshape(X.shape[0], -1)
                kernel_matrix = gauss_kernel(X, X)
                cov_biased = biased_covariance_estimator(kernel_matrix, tau= tau)
                cov_unbiased = unbiased_covariance_estimator(kernel_matrix, tau= tau)
                trace_C_hat2 = (np.linalg.norm(kernel_matrix)**2)/(n*n)
                trace_C_C_hat = 0
                for t in range(n_repits_est_tr):
                    X_prime = data_points2[:,t]
                    X_prime = X_prime.reshape(X_prime.shape[0], -1) 
                    c = (np.linalg.norm(gauss_kernel(X_prime,X[0:n]))**2)/(n_sample_est_tr*n*n_repits_est_tr)
                    trace_C_C_hat += c
        

            else:
                assert (n_sample_est_tr / tau) % 2 == 0, "n_sample_est_tr diveded by tau must be even"
                X = data_points[0:Ns[j]][:,i]
                X = X.reshape(X.shape[0], -1)
                n_parts = int(n / configs.n_sample_thresh)
                assert n_parts % 2 == 0, "n_parts must be even"
                cov_biased = 0
                cov_unbiased = 0
                trace_C_hat2 = 0
                trace_C_C_hat = 0
                for k in range(n_parts):
                    for l in range(n_parts):
                        kernel_matrix = gauss_kernel(X[k*configs.n_sample_thresh:(k+1)*configs.n_sample_thresh], X[l*configs.n_sample_thresh:(l+1)*configs.n_sample_thresh])
                        trace_C_hat2 += (np.linalg.norm(kernel_matrix)**2)/(n*n)
                        if k == l:
                            cov_biased += biased_covariance_estimator(kernel_matrix, tau= tau)
                            cov_unbiased += sum_off_diagonals(kernel_matrix, tau=tau)/(2*m*(m - 1))
                        else:
                            cov_unbiased += sum_diagonals(kernel_matrix, tau= tau)/(2*m*(m - 1))

                    
                    for t in range(n_repits_est_tr):
                        X_prime = data_points2[:,t]
                        X_prime = X_prime.reshape(X_prime.shape[0], -1) 
                        c = (np.linalg.norm(gauss_kernel(X_prime,X[k*configs.n_sample_thresh:(k+1)*configs.n_sample_thresh]))**2)
                        trace_C_C_hat += c

                trace_C_C_hat = trace_C_C_hat/(n_sample_est_tr*n*n_repits_est_tr)
                cov_biased = cov_biased / n_parts            
                cov_unbiased = max(cov_biased - cov_unbiased, 0)

            
            # print( trace_C2, - (2*trace_C_C_hat),trace_C_hat2)
            beta_coeff = np.exp((1/np.exp(1) - 1) *tau)
            # print(delta - 2*(m-1)*beta_coeff)
            ltau = np.log(8/(delta - 2*(m-1)*beta_coeff))
            l_tau = np.log(4/(delta - 2*(m-1)*beta_coeff))
            L_tau = np.log(2/(delta - 2*(m-1)*beta_coeff))
            # print(cov_biased, cov_unbiased)
            # Cov_biased[j][i] = cov_biased
            # Cov_unbiased[j][i] = cov_unbiased

            M_bound[j][i] = (4*c_h)/(3*m)*L_tau + np.sqrt(((2*L_tau + 1)*2*(sigma**2))/m) # Apply lemma 2 to Theorem 4
            M_emp_bound_biased_cov_est[j][i] = ((16*c_h)/(3*m))*l_tau + np.sqrt(((2*l_tau + 1)*cov_biased)/m) # Theorem 2
            M_emp_bound_unbiased_cov_est[j][i] = ((11*c_h)/(m))*l_tau + np.sqrt(((2*l_tau + 1)*cov_unbiased)/m) # Theorem 3

            Pinelis_bound[j][i] = (((4 * c_h ) / m)  + (2 * sigma)/np.sqrt(m))* l_tau # Apply lemma 2 to Pinelis
            Pinelis_emp_bound_biased_cov_est[j][i] = ((2*c_h)/m)*ltau*(2+np.sqrt(2*l_tau)) + 2*ltau*np.sqrt(cov_biased/m) # Apply cov biased estimation to pinelis and then lemma 2
            Pinelis_emp_bound_unbiased_cov_est[j][i] = ((4*c_h)/m)*ltau*(1+np.sqrt(4*l_tau)) + 2*ltau*np.sqrt(cov_unbiased/m)

            True_value[j][i] = np.sqrt(abs(trace_C2 - (2*trace_C_C_hat) + trace_C_hat2))
        
        del kernel_matrix
        del X
        del X_prime
        del cov_biased
        del cov_unbiased

    return Pinelis_bound, Pinelis_emp_bound_biased_cov_est, Pinelis_emp_bound_unbiased_cov_est, M_bound, M_emp_bound_biased_cov_est, M_emp_bound_unbiased_cov_est, True_value