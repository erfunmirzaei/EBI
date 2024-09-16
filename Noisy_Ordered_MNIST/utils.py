import numpy as np
import matplotlib.pyplot as plt
from typing import NamedTuple

from pathlib import Path

main_path = Path(__file__).parent

class Metrics(NamedTuple):
    train_acc: list[float]
    train_steps: list[float]
    val_acc: list[float]
    val_steps: list[float]

def plot_noisy_ordered_MNIST(Noisy_ordered_MNIST: dict, n: int = 16):
    """
    Plot the noisy ordered MNIST dataset
    """
    fig, ax = plt.subplots(3, n, figsize=(n, 3))
    for j, split in enumerate(['train', 'test', 'validation']):
        print(f'{split} ({len(Noisy_ordered_MNIST[split])}) example: {Noisy_ordered_MNIST[split]["label"][:n]}')
        for i in range(n):
            data = Noisy_ordered_MNIST[split][i]
            ax[j, i].imshow(np.squeeze(data['image']), cmap='gray')
            ax[j, i].set_title(data['label'].item())
            ax[j, i].axis('off')
    fig.tight_layout()
    plt.show()

# Plot the training and validation accuracy of the oracle
def plot_oracle_metrics(metrics: Metrics):
    fig, ax = plt.subplots(1, 1)
    ax.plot(metrics.train_steps, metrics.train_acc, label='Train')
    ax.plot(metrics.val_steps, metrics.val_acc, label='Validation')
    ax.set_xlabel('Global step')
    ax.set_ylabel('Accuracy')
    ax.legend(frameon=False, loc='lower right')
    ax.margins(x=0)
    plt.show()

def plot_image_forecast(true_labels, true_images, models_name: list, pred_labels: dict, pred_images: dict, configs):
    num_models = len(models_name)
    num_cols = configs.eval_up_to_t + 1
    fig, axes = plt.subplots(num_models, num_cols, figsize=(9.75, 3), sharex=True, sharey=True)

    # Remove margins between columns
    plt.subplots_adjust(wspace=0)

    for model_idx, model_name in enumerate(models_name):
        ax = axes[model_idx, 0]
        ax.imshow(np.squeeze(true_images[model_name][configs.n_train_acc_plot][configs.n_rep_plot]), cmap='gray')
        # Remove axes and ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

        for t_idx in range(num_cols - 1):
            pred_label = pred_labels[model_name][configs.n_train_acc_plot][configs.n_rep_plot][t_idx]
            true_label = (true_labels[model_name][configs.n_train_acc_plot][configs.n_rep_plot][configs.test_seed_idx] + t_idx + 1)%configs.classes
            # true_label = test_labels[test_seed_idx + t_idx + 1]
            img = np.squeeze(pred_images[model_name][configs.n_train_acc_plot][configs.n_rep_plot][t_idx])

            # Set subplot for the current class
            ax = axes[model_idx, t_idx + 1]

            # Plot the MNIST image
            ax.imshow(img, cmap='gray')

            # Remove axes and ticks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')

            # Add a white background for the subplot
            ax.set_facecolor('white')

            # # Add an inset for the predicted label in the upper right corner
            # if pred_label == true_label:
            #     color = 'green'
            # else:
            #     color = 'red'
            # inset_ax = ax.inset_axes([0.75, 0.75, 0.25, 0.25])
            # inset_ax.set_xlim(0, 1)
            # inset_ax.set_ylim(0, 1)
            # inset_ax.text(0.5, 0.4, f"{pred_label}" , color=color, fontsize=9, ha='center', va='center')
            # inset_ax.set_xticks([])
            # inset_ax.set_yticks([])
            # inset_ax.set_facecolor('white')

    # Display the model names on the left of each row
    for model_idx, model_name in enumerate(models_name):
        axes[model_idx, 0].text(-0.1, 0.5, model_name.replace('_', ' '), fontsize=14, ha='right', va='center', transform=axes[model_idx, 0].transAxes)

    for class_idx in range(num_cols):
        title = (true_labels[model_name][configs.n_train_acc_plot][configs.n_rep_plot][configs.test_seed_idx]  + class_idx)%configs.classes
        if class_idx == 0:
            axes[0, class_idx].set_title(f"Seed: {title}", fontsize=14)
        else:
            axes[0, class_idx].set_title(f"{title}", fontsize=14)

    plt.savefig(str(main_path) + f'/results/second_figure_eta_{configs.eta}.pdf', dpi = 600, bbox_inches='tight', pad_inches=0.1)
    plt.show()

def plot_TNSE(models_name:list, fn_i, fn_j, configs,  true_labels): 
    """Plot the t-SNE of the feature functions for all the transfer operator models in the report dictionary"""
    # Define the dimensionality reduction method
    n_models = len(models_name)
    num_rows = 1
    num_cols = n_models
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3.25*3, 3.25))
    axes = axes.flatten()

    for model_idx, (ax, model_name) in enumerate(zip(axes, models_name)):
        ax.set_title(model_name.replace('_', ' '), fontsize=14)  # Adjust title font size
        fni = fn_i[model_name][configs.n_train_acc_plot][configs.n_rep_plot]
        fnj = fn_j[model_name][configs.n_train_acc_plot][configs.n_rep_plot]

        scatter = ax.scatter(fni, fnj, c=true_labels[model_name][configs.n_train_acc_plot][configs.n_rep_plot], cmap='tab10', vmax=10, alpha=0.7, linewidths=0)

    # Add space for legend
    plt.subplots_adjust(right=0.95)

    # Add a legend for the last axis
    legend = fig.legend(*scatter.legend_elements(num=4), loc='center left', bbox_to_anchor=(0.97, 0.55),
                        title="Digits", frameon=False, fontsize=12)  # Adjust legend font size

    plt.tight_layout()
    plt.savefig(str(main_path) + f"/results/third_figure_eta_{configs.eta}.pdf", format="pdf", dpi=600, bbox_inches='tight')  # Adjust DPI for better quality
    plt.show()

def plot_normalized_biased_corr_est(transfer_operator_models, biased_cov_ests, Ns, configs):
    # Plot
    plt.figure(figsize=(12, 8))  # Adjust figure size as needed

    # Define marker styles
    markers = ['o', 's', '^', 'P', 'D', "*"]

    for i, model_name in enumerate(transfer_operator_models):
        biased_est_mean = np.mean(biased_cov_ests[model_name], axis=-1)
        biased_est_std = np.std(biased_cov_ests[model_name], axis=-1)
        plt.plot(Ns, biased_est_mean, marker=markers[i], label=f"{model_name} (biased cov. est.)", linewidth=2)
        plt.fill_between(Ns, biased_est_mean - biased_est_std, biased_est_mean + biased_est_std, alpha=0.2)

    # plt.plot(Ns, np.ones((n_0,1)), label = 'Upper Bound')
    # plt.plot(Ns, lower_bound[:,0], label = 'Lower Bound')

    plt.xlabel("Number of training samples", fontsize=14)
    plt.ylabel("Normalized correlation", fontsize=14)
    # plt.title(f"Noisy ordered MNIST for different models (delta = {delta})", fontsize=16)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.savefig(str(main_path) + f"/results/Normalized_corr_delta_{configs.delta}_biased_p_{configs.eta}.pdf", format="pdf", dpi=300)
    plt.show()

def plot_normalized_unbiased_corr_est(transfer_operator_models, unbiased_cov_ests,  Ns, configs):
    # Plot
    plt.figure(figsize=(12, 8))  # Adjust figure size as needed

    # Define marker styles
    markers = ['o', 's', '^', 'P', 'D', "*"]

    for i, model_name in enumerate(transfer_operator_models):
        unbiased_est_mean = np.mean(unbiased_cov_ests[model_name], axis=-1)
        unbiased_est_std = np.std(unbiased_cov_ests[model_name], axis=-1)
        plt.plot(Ns, unbiased_est_mean,marker=markers[i], label=f"{model_name} (unbiased cov. est.)", linewidth=2)
        plt.fill_between(Ns, unbiased_est_mean - unbiased_est_std, unbiased_est_mean + unbiased_est_std, alpha=0.2)

    # plt.plot(Ns, np.ones((n_0,1)), label = 'Upper Bound')
    # plt.plot(Ns, lower_bound[:,0], label = 'Lower Bound')

    plt.xlabel("Number of training samples", fontsize=14)
    plt.ylabel("Normalized correlation", fontsize=14)
    # plt.title(f"Noisy ordered MNIST for different models (delta = {delta})", fontsize=16)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.savefig(str(main_path) + f"/results/Normalized_corr_delta_{configs.delta}_unbiased_p_{configs.eta}.pdf", format="pdf", dpi=300)
    plt.show()


# First figure in the first row
def plot_biased_estimates(ax, models_name, biased_cov_ests, Ns):
    markers = ['o', 's', '^', 'P', 'D', "*"]
    for i, model_name in enumerate(models_name):
        biased_est_mean = np.mean(biased_cov_ests[model_name], axis=-1)
        biased_est_std = np.std(biased_cov_ests[model_name], axis=-1)
        ax.plot(Ns, biased_est_mean, marker=markers[i], label=model_name, linewidth=2)
        ax.fill_between(Ns, biased_est_mean - biased_est_std, biased_est_mean + biased_est_std, alpha=0.2)
    ax.set_xlabel("Number of training samples", fontsize=12)
    ax.set_ylabel("Normalized correlation", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_title("Biased Covariance Estimates", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True)

# Second figure in the first row
def plot_unbiased_estimates(ax, models_name, unbiased_cov_ests, Ns):
    markers = ['o', 's', '^', 'P', 'D', "*"]
    for i, model_name in enumerate(models_name):
        unbiased_est_mean = np.mean(unbiased_cov_ests[model_name], axis=-1)
        unbiased_est_std = np.std(unbiased_cov_ests[model_name], axis=-1)
        ax.plot(Ns, unbiased_est_mean, marker=markers[i], label=model_name, linewidth=2)
        ax.fill_between(Ns, unbiased_est_mean - unbiased_est_std, unbiased_est_mean + unbiased_est_std, alpha=0.2)
    ax.set_xlabel("Number of training samples", fontsize=12)
    # ax.set_ylabel("Normalized correlation", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_title("Unbiased Covariance Estimates", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True)

# Third figure in the first row
def plot_accuracy_ordered_vs_time(ax, models_name, ordered_acc, configs):
    for model_name in models_name:
        acc_ordered_mean = np.mean(ordered_acc[model_name][configs.n_train_acc_plot], axis=0)
        acc_ordered_std = np.std(ordered_acc[model_name][configs.n_train_acc_plot], axis=0)
        ax.plot(np.arange(configs.eval_up_to_t), acc_ordered_mean, label=model_name.replace('_', ' '))
        ax.fill_between(np.arange(configs.eval_up_to_t), acc_ordered_mean - acc_ordered_std, acc_ordered_mean + acc_ordered_std, alpha=0.2)

    ax.axhline(1/configs.classes, color='black', linestyle='--', label='Random')
    ax.margins(x=0)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel('Time steps', fontsize=12)
    ax.set_ylabel('Average accuracy', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_title("Forecasting MNIST", fontsize=12)
    ax.grid(True)
    ax.legend(fontsize=10)

# Function to create the full figure with the required subplots
def Plot_first_figure(models_name, biased_cov_ests, unbiased_cov_ests, ordered_acc, Ns, configs):
    fig = plt.figure(figsize=(9.75, 4.5))  # Width: 14 inches, Height: 5 inches

    # First row of subplots
    gs1 = fig.add_gridspec(nrows=1, ncols=3, top=0.95, bottom=0.1, left=0.05, right=0.95, wspace=0.3)

    ax1 = fig.add_subplot(gs1[0, 0])
    plot_biased_estimates(ax1, models_name, biased_cov_ests, Ns)

    ax2 = fig.add_subplot(gs1[0, 1])
    plot_unbiased_estimates(ax2, models_name, unbiased_cov_ests, Ns)

    ax3 = fig.add_subplot(gs1[0, 2])
    plot_accuracy_ordered_vs_time(ax3, models_name, ordered_acc, configs)

    plt.savefig(str(main_path) + f'/results/first_figure_eta_{configs.eta}.pdf', dpi = 600, bbox_inches='tight', pad_inches=0.1)
    plt.show()