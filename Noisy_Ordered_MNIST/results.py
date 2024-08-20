biased_cov_ests = {}
unbiased_cov_ests = {}
# Ns = [500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000]
Ns = np.arange(500, configs.train_samples, 500)
delta = configs.delta

for i, model_name in enumerate(['Gaussian_RRR',"DPNets", "Classifier_Baseline"]):
    model_name = model_name.replace(" ", "")
    unbiased_cov_ests[model_name] = np.load(f'/content/drive/MyDrive/Noisy_ordered_MNIST/unbiased_cov_ests_{model_name}_eta_{configs.eta}.npy')
    biased_cov_ests[model_name] = np.load(f'/content/drive/MyDrive/Noisy_ordered_MNISTbiased_cov_ests_{model_name}_eta_{configs.eta}.npy')

# Plot the results
# Assuming the required data structures are available
create_figure(transfer_operator_models, biased_cov_ests, unbiased_cov_ests, Ns, delta, report, configs)

# Plot the image forecast for the first 16 examples in the test set
plot_image_forecast(Noisy_ordered_MNIST, report, configs, test_seed_idx=0)

# Plot the t-SNE of the feature functions for all the transfer operator models in the report dictionary
plot_TNSE(report, configs, test_data, test_labels, transfer_operator_models)