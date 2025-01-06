import time

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from copy import deepcopy
from typing import Dict, List, Tuple

# Load DCIts
from src.dcits import DCITS

def dataset(len_of_timeseries, ground_truth_alpha, bias, noise_frequency=0, mu=0, sigma=1, burn_in=1000, seed=None, nonlinear_function=None):
    """
    Generates a multivariate time series based on the specified equations with reproducible noise.
    Args:
        len_of_timeseries (int): Length of the time series to generate
        ground_truth_alpha (torch.Tensor): Coefficients tensor of shape (no_of_timeseries, no_of_timeseries, window_length)
        bias (torch.Tensor): Bias term for each time series component, shape (no_of_timeseries,)
        noise_frequency (float): Probability of adding noise at each step (0 to 1)
        mu (float): Mean of the Gaussian noise
        sigma (float): Standard deviation of the Gaussian noise
        burn_in (int): Number of initial samples to discard
        seed (int, optional): Seed for random number generation for reproducibility
        nonlinear_function (callable, optional): Nonlinear function to apply after linear combination of lagged values.
                                                 Common choice can be torch.tanh for bounded dynamics.
    Returns:
        torch.Tensor: Generated time series of shape (no_of_timeseries, len_of_timeseries)
    """
    if seed is not None:
        np.random.seed(seed)
        
    # set dimensions    
    no_of_timeseries, window_length = ground_truth_alpha.shape[-2:]
    total_length = len_of_timeseries + burn_in
    
    # Initialize all values with Gaussian variates N(0,1) - sets initial values, and array shape
    time_series = torch.tensor(
        np.random.normal(0, 1, (no_of_timeseries, total_length)), 
        dtype=torch.float32
    )
    
    # Generate noise mask only for values after initial window
    noise_mask = (np.random.rand(no_of_timeseries, total_length - window_length) < noise_frequency)
    # Add noise as Gaussian variates N(mu,sigma) with mask 
    noise_after_window = np.random.normal(mu, sigma, (no_of_timeseries, total_length - window_length)) * noise_mask
    
    # Only apply masked noise after the initial window
    time_series[:, window_length:] = torch.tensor(noise_after_window, dtype=torch.float32)
    
    # Populate the time series based on lagged values
    for t in range(window_length, total_length):
        # Get all relevant lags at once: shape (no_of_timeseries, window_length)
        #  flip(1) to properly align the lags (most recent first)
        lagged_values = time_series[:, t-window_length:t].flip(1)

        # einsum 'ijk,ki->i': i=target series, j=source series, k=time lags
        lagged_sum = torch.einsum('ijk,jk->i', ground_truth_alpha, lagged_values)
        
        # Update time series with computed values and bias
        transition_value = lagged_sum + bias
        
        if nonlinear_function:
            transition_value = nonlinear_function(transition_value)
        
        time_series[:, t] = transition_value + time_series[:, t] # add noise at step t
    
    # Drop the burn-in samples
    return time_series[:, burn_in:]

def plot_ts(dataset_sample, dataset_name="Dataset", alpha=0.7):
    plt.figure(figsize=(10, 6))

    for i in range(dataset_sample.shape[0]):
        plt.plot(dataset_sample[i, :], label=f'Time series {i+1}',alpha=alpha)
    
    # Add labels and title
    plt.xlabel('Time steps')
    plt.ylabel('Value')
    plt.title(dataset_name)
    plt.grid(True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=dataset_sample.shape[0]);
    plt.show()

def create_windowed_dataset(time_series, window_size):
    window_size_p1 = window_size + 1  # We add 1 to include the target in the window
    num_series, series_length = time_series.shape
    windows = []
    for start in range(series_length - window_size_p1 + 1):
        window = time_series[:, start:start + window_size_p1]
        windows.append(window.unsqueeze(0))
    return torch.cat(windows, dim=0)

def split_time_series(time_series, train_ratio=0.6, val_ratio=0.2, window_size=5):
    """
    Splits time_series into train, validation, and test sets with no overlapping time steps.
    Ensures that each split has at least one window's worth of data.
    """
    num_series, series_length = time_series.shape
    min_series_length = window_size + 1  # Minimum required length to create at least one window

    if series_length < 3 * min_series_length:
        raise ValueError("Not enough data to create train, validation, and test splits with the given window_size.")

    # Calculate sizes ensuring minimum length for each split
    total_length = series_length
    initial_train_size = int(train_ratio * total_length)
    initial_val_size = int(val_ratio * total_length)
    initial_test_size = total_length - initial_train_size - initial_val_size

    # Adjust sizes if any are less than the minimum
    sizes = {
        'train': max(initial_train_size, min_series_length),
        'val': max(initial_val_size, min_series_length),
        'test': max(initial_test_size, min_series_length)
    }

    # Recalculate total size and adjust if necessary
    total_assigned = sum(sizes.values())
    if total_assigned > total_length:
        # Reduce the largest split(s) to compensate
        while total_assigned > total_length:
            for key in ['train', 'val', 'test']:
                if sizes[key] > min_series_length:
                    sizes[key] -= 1
                    total_assigned -= 1
                    if total_assigned == total_length:
                        break

    # Ensure sizes sum to total_length
    sizes['train'] = min(sizes['train'], total_length - sizes['val'] - sizes['test'])
    sizes['val'] = min(sizes['val'], total_length - sizes['train'] - sizes['test'])
    sizes['test'] = total_length - sizes['train'] - sizes['val']

    # Check final sizes
    if any(size < min_series_length for size in sizes.values()):
        raise ValueError("Unable to split time series into train, val, and test sets with sufficient data.")

    # Split the series
    train_end = sizes['train']
    val_end = train_end + sizes['val']

    train_series = time_series[:, :train_end]
    val_series = time_series[:, train_end:val_end]
    test_series = time_series[:, val_end:]

    return train_series, val_series, test_series


def train_and_evaluate(
    time_series,
    window_size,
    temperature,
    order=[1,1],
    config=None
):
    """
    Enhanced training and evaluation function using configuration dictionary.
    
    Args:
        time_series (torch.Tensor): Input time series data
        window_size (int): Size of the sliding window
        temperature (float): Temperature parameter for the model
        order (list, optional): List indicating which orders to include.
                                    Example: [1,1,0,1] includes bias (order 0), linear (order 1),
                                    excludes quadratic (order 2), and includes cubic (order 3).0
                                    Default: [1, 1]
        config (dict): Configuration dictionary with training parameters
            
    Returns:
        tuple: (test_loss, train_losses, val_losses, f_test, c_test, debug_info)
    """
    # Default configuration
    default_config = {
        'seed': 1000,
        'batch_size': 64,
        'learning_rate': 1e-3,
        'train_ratio': 0.6,
        'val_ratio': 0.2,
        'epochs': 100,
        'device': None,
        'scheduler_patience': 5,
        'verbose': False,
        'debug_mode': False,
        'memory_callback': None,
        'criterion': None,
        'early_stopping_modifier': 2,
        'min_epochs': 10,
        'min_learning_rate': 1e-6
    }

    # Update default config with provided config
    if config is not None:
        default_config.update(config)
    config = default_config

    # Set device if not provided
    if config['device'] is None:
        config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set default loss function
    if config['criterion'] is None:
        config['criterion'] = nn.MSELoss()
        
    # Debug information dictionary
    debug_info = {
        'epoch_details': [],
        'early_stopping_trigger': None,
        'lr_changes': [],
        'batch_losses': [] if config['debug_mode'] else None,
        'gradient_norms': [] if config['debug_mode'] else None
    }

    # Prepare data
    train_series, val_series, test_series = split_time_series(
        time_series, 
        train_ratio=config['train_ratio'], 
        val_ratio=config['val_ratio'],
        window_size=window_size
    )

    # Create windowed datasets
    train_windowed_dataset = create_windowed_dataset(train_series, window_size)
    val_windowed_dataset = create_windowed_dataset(val_series, window_size)
    test_windowed_dataset = create_windowed_dataset(test_series, window_size)

    # Extract inputs and targets
    train_inputs = train_windowed_dataset[:, :, :-1].float()
    train_targets = train_windowed_dataset[:, :, -1].float()

    val_inputs = val_windowed_dataset[:, :, :-1].float()
    val_targets = val_windowed_dataset[:, :, -1].float()

    test_inputs = test_windowed_dataset[:, :, :-1].float()
    test_targets = test_windowed_dataset[:, :, -1].float()

    # Create data loaders with fixed seeds
    train_gen = torch.Generator().manual_seed(config['seed'])
    val_gen = torch.Generator().manual_seed(config['seed'])
    test_gen = torch.Generator().manual_seed(config['seed'])

    train_dataset = TensorDataset(train_inputs, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                            shuffle=True, generator=train_gen, num_workers=0)

    val_dataset = TensorDataset(val_inputs, val_targets)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                          shuffle=False, generator=val_gen, num_workers=0)

    test_dataset = TensorDataset(test_inputs, test_targets)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                           shuffle=False, generator=test_gen, num_workers=0)

    # Check that datasets are not empty
    if len(train_loader.dataset) == 0:
        raise ValueError("Training dataset is empty. Cannot proceed with training.")
    if len(val_loader.dataset) == 0:
        raise ValueError("Validation dataset is empty. Cannot proceed with training.")
    if len(test_loader.dataset) == 0:
        raise ValueError("Test dataset is empty. Cannot proceed with evaluation.")

    # Initialize model and training components
    model = DCITS(no_of_timeseries=time_series.shape[0], 
                 window_length=window_size, 
                 temperature=temperature,
                 order=order)
    model = model.to(config['device'])

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=config['scheduler_patience'],
        min_lr=config['min_learning_rate'])

    # Early stopping setup
    early_stopping_patience = int(config['scheduler_patience'] * config['early_stopping_modifier'])
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    train_losses = []
    val_losses = []

    def compute_gradient_norm():
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0.0
        batch_losses_epoch = [] if config['debug_mode'] else None
        gradient_norms_epoch = [] if config['debug_mode'] else None

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if config['memory_callback'] is not None:
                config['memory_callback']()
                
            inputs = inputs.to(config['device']).unsqueeze(1)
            targets = targets.to(config['device'])
            
            optimizer.zero_grad()
            outputs, _, _ = model(inputs)
            loss = config['criterion'](outputs, targets)
            if torch.isnan(loss):
                raise ValueError(f"Loss is NaN at epoch {epoch+1}, batch {batch_idx+1}.")
            loss.backward()
            
            if config['debug_mode']:
                grad_norm = compute_gradient_norm()
                gradient_norms_epoch.append(grad_norm)
                debug_info['gradient_norms'].append(grad_norm)
                batch_losses_epoch.append(loss.item())
            
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)

        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(config['device']).unsqueeze(1)
                targets = targets.to(config['device'])
                outputs, _, _ = model(inputs)
                loss = config['criterion'](outputs, targets)
                if torch.isnan(loss):
                    raise ValueError(f"Validation loss is NaN at epoch {epoch+1}.")
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if old_lr != new_lr:
            debug_info['lr_changes'].append((epoch + 1, old_lr, new_lr))

        # Store epoch details
        epoch_info = {
            'epoch': epoch + 1,
            'train_loss': epoch_loss,
            'val_loss': val_loss,
            'learning_rate': new_lr
        }
        if config['debug_mode']:
            epoch_info.update({
                'batch_losses': batch_losses_epoch,
                'gradient_norms': gradient_norms_epoch
            })
        debug_info['epoch_details'].append(epoch_info)

        if config['verbose']:
            print(f"Epoch {epoch+1}/{config['epochs']}, "
                  f"Train Loss: {epoch_loss:.6e}, "
                  f"Val Loss: {val_loss:.6e}, "
                  f"LR: {new_lr:.2e}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if (epoch + 1) >= config['min_epochs'] and epochs_no_improve >= early_stopping_patience:
                debug_info['early_stopping_trigger'] = epoch + 1
                if config['verbose']:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load best model and evaluate
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Test the model
    model.eval()
    
    # Initialize dictionaries for interpretability data (focus weights and coefficients)
    test_focus = {i: [] for i in range(len(model.order))}
    test_coefficients = {i: [] for i in range(len(model.order))}
    
    # Evaluate the model on the test set
    test_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(config['device']).unsqueeze(1)
            targets = targets.to(config['device'])
            outputs, f, c = model(inputs)
            loss = config['criterion'](outputs, targets)
            if torch.isnan(loss):
                raise ValueError("Loss became NaN during testing.")
            test_loss += loss.item() * inputs.size(0)

            # Collect interpretability data: focus weights and coefficients
            if model.bias:
                test_focus[0].append(f[0].cpu().numpy())
                test_coefficients[0].append(c[0].cpu().numpy())
            
            for i, order in enumerate(model.order[1:], start=1):
                if order == 1:
                    test_focus[i].append(f[i].cpu().numpy())
                    test_coefficients[i].append(c[i].cpu().numpy())

    # Convert lists to numpy arrays for further analysis or visualization
    f_test = {i: np.concatenate(test_focus[i], axis=0) for i in test_focus if test_focus[i]}
    c_test = {i: np.concatenate(test_coefficients[i], axis=0) for i in test_coefficients if test_coefficients[i]}
    
    test_loss /= len(test_loader.dataset)

    if config['verbose']:
        print(f"Final Test Loss: {test_loss:.6f}")

    return test_loss, train_losses, val_losses, f_test, c_test, debug_info, test_inputs, test_targets, model


def format_values_with_std(value_tmp, value_tmp_std, round=2):
    """
    Combines values and their standard deviations into a formatted string with specified rounding.
    
    Parameters:
    value_tmp (np.array): Array of values.
    value_tmp_std (np.array): Array of standard deviations.
    round (int): Number of decimal places for rounding. Default is 2.
    
    Returns:
    str: Combined array as a string with each element in the format 'value ± std' with specified precision.
    """
    
    # Create the format string dynamically based on the 'round' parameter, ensuring fixed width
    format_str = f'{{:>{round+4}.{round}f}} ± {{:>{round+2}.{round}f}}'  # Adjust width with round

    # Combine the two arrays element by element
    combined_array = np.array2string(
        np.array([[format_str.format(value, std) for value, std in zip(row, row_std)] for row, row_std in zip(value_tmp, value_tmp_std)]),
        formatter={'str_kind': lambda x: x},
        max_line_width=200  # Set a large line width to prevent wrapping
    )
    
    # Return the combined formatted string
    return combined_array

def collect_multiple_runs(
    n_runs: int,
    time_series: torch.Tensor,
    window_size: int,
    temperature: float,
    order: List[int] = None,
    config=None,
    seed: int = 1000,
    verbose: bool = True
) -> Dict:
    """
    Perform multiple runs of the model and collect statistics.
    
    Args:
        n_runs (int): Number of runs to perform
        time_series (torch.Tensor): Input time series data
        window_size (int): Size of the sliding window
        temperature (float): Temperature parameter for the model
        order (List[int], optional): Same as in train_eval (default: [1, 1])
        config (dict): Configuration dictionary with training parameters
        seed (int): Base seed for reproducibility
        verbose (bool): Whether to print progress
        
    Returns:
        Dict: Dictionary containing test losses and alpha statistics for each run
    """
    if order is None:
        order = [1, 1]

    if config is None:
        train_config = {
        'verbose': True,
        'device' : device,
        'learning_rate': 1e-3,
        'scheduler_patience': 5,
        'early_stopping_modifier': 2,
        'criterion': nn.MSELoss()
        }
    else:
        train_config = config

    # Print the training configuration in a formatted way
    print("Training Configuration:")
    for key, value in train_config.items():
        print(f"  {key}: {value}")
    
    results = {}
    
    for run in range(n_runs):
        if verbose:
            print(f"Starting Run {run + 1}/{n_runs}")
            
        # Record the start time
        start_time = time.time()
        
        # Set seed for this run
        current_seed = seed + run
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        
        # Run the model
        test_loss, train_losses, val_losses, f_test, c_test, debug_info,_,_,_ = train_and_evaluate(
            time_series,
            window_size=window_size,
            temperature=temperature,
            order=order,
            config=train_config
        )
     
        alpha = {}
        alpha_std = {}
        f_means = {}
        c_means = {}
        for i in f_test.keys():
            if i == 0:
                alpha_bias = (f_test[0] * c_test[0]).mean(0)  # Mean of bias alpha coefficients
                alpha_bias_std = (f_test[0] * c_test[0]).std(0)  # Std of bias alpha coefficients
            else:
                alpha[i] = (f_test[i] * c_test[i]).mean(0)  # Mean of regular alpha coefficients
                alpha_std[i] = (f_test[i] * c_test[i]).std(0)  # Std of regular alpha coefficients
                f_means[i] = f_test[i].mean(0)  # Mean of f_test
                c_means[i] = c_test[i].mean(0)  # Mean of c_test
        
        # Store results for this run
        results[f'run_{run + 1}'] = {
            'test_loss': test_loss,
            'alpha': alpha,
            'alpha_std': alpha_std,
            'alpha_bias': alpha_bias,
            'alpha_bias_std': alpha_bias_std,
            'f_means': f_means,
            'c_means': c_means
        }
        
        if verbose:
            print(f"Run {run + 1} completed. Test Loss: {test_loss:.6e}")
    # Record the end time
    end_time = time.time()
    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Time taken for Run {i+1}: {elapsed_time:.2f} seconds")    
    
    # Add summary statistics across runs
    test_losses = [results[f'run_{i+1}']['test_loss'] for i in range(n_runs)]
    results['summary'] = {
        'mean_test_loss': np.mean(test_losses),
        'std_test_loss': np.std(test_losses),
        'min_test_loss': np.min(test_losses),
        'max_test_loss': np.max(test_losses),
        'best_run': f"run_{np.argmin(test_losses) + 1}"
    }
    
    return results

def print_bias(alpha_bias, alpha_bias_std, ground_truth_bias):
    """
    Print bias terms with their mean, std, and ground truth values.
    
    Args:
        alpha_bias (np.ndarray): Bias terms (mean values) of shape (N, N, L).
        alpha_bias_std (np.ndarray): Bias standard deviations of shape (N, N, L).
        ground_truth_bias (torch.Tensor): Ground truth bias terms as a 1D tensor.
    """
    # Convert ground_truth_bias to numpy and make alpha_bias 1D
    ground_truth_bias = ground_truth_bias.numpy()
    bias_mean_1d = np.diag(alpha_bias[:, :, 0])  # Extract diagonal (source bias terms)
    bias_std_1d = np.diag(alpha_bias_std[:, :, 0])  # Extract diagonal std
    
    # Iterate over each bias term
    for source, (bias_value, bias_std, gt_bias_value) in enumerate(zip(bias_mean_1d, bias_std_1d, ground_truth_bias), start=1):
        # Determine significant decimal places based on std
        significant_decimal_places = max(0, int(-np.floor(np.log10(bias_std))))
        
        # Format values without scientific notation
        bias_value_formatted = f"{bias_value:.{significant_decimal_places}f}"
        bias_std_formatted = f"{bias_std:.{significant_decimal_places}f}"
        gt_bias_value_formatted = f"{gt_bias_value:.{significant_decimal_places}f}"
        
        # Print bias information
        print(f"bias_{source} = ({bias_value_formatted} ± {bias_std_formatted}), gt_bias_{source} = {gt_bias_value_formatted}")

def print_significant_alpha(alpha, alpha_std, ground_truth_alpha, threshold=0.01):
    """
    Print significant alpha values and compare with ground truth.

    Args:
        alpha (np.ndarray): Alpha values (source, target, lag).
        alpha_std (np.ndarray): Standard deviation of alpha values.
        ground_truth_alpha (torch.Tensor): Ground truth alpha values as a tensor.
        threshold (float): Minimum threshold for significance.
    """
    # Convert ground truth to NumPy array if it's a tensor
    if isinstance(ground_truth_alpha, torch.Tensor):
        ground_truth_alpha = ground_truth_alpha.numpy()

    # Adjust ground truth shape to match alpha's shape
    if ground_truth_alpha.shape[2] < alpha.shape[2]:
        ground_truth_alpha = np.pad(
            ground_truth_alpha,
            ((0, 0), (0, 0), (0, alpha.shape[2] - ground_truth_alpha.shape[2])),
            mode='constant',
        )

    for i in range(alpha.shape[0]):  # Iterate over source (rows in the matrix)
        # Flip axis=1 to match lag notation in figures for alpha
        value_tmp = np.flip(alpha[i], axis=1)
        value_tmp_std = np.flip(alpha_std[i], axis=1)

        # Calculate significance mask (|mean| > 1.95 * std) and ignore abs(value) < threshold
        significance_mask = (np.abs(value_tmp) > 1.95 * value_tmp_std) & (np.abs(value_tmp) >= threshold)

        # Compare with alpha_mask
        alpha_mask = (ground_truth_alpha[i] != 0)
        ground_truth_mask = alpha_mask

        # Process all positions
        for j in range(value_tmp.shape[0]):  # targets
            for k in range(value_tmp.shape[1]):  # lags
                target = j + 1
                lag = k + 1
                source = i + 1

                is_significant = significance_mask[j, k]
                has_ground_truth = ground_truth_mask[j, k]

                if is_significant:
                    value = value_tmp[j, k]
                    std = value_tmp_std[j, k]
                    significant_decimal_places = max(0, int(-np.floor(np.log10(std))))
                    value_formatted = f"{value:.{significant_decimal_places}f}"
                    std_formatted = f"{std:.{significant_decimal_places}f}"

                    if has_ground_truth:
                        # Print as matching significant value
                        ground_truth_value = ground_truth_alpha[i, j, k]
                        ground_truth_formatted = f"{ground_truth_value:.{significant_decimal_places}f}"
                        print(f"alpha_{source}{target}{lag} = ({value_formatted} ± {std_formatted}), gt_alpha_{source}{target}{lag} = {ground_truth_formatted}")
                    else:
                        # Print as not in ground truth
                        print(f"alpha_{source}{target}{lag} = ({value_formatted} ± {std_formatted}) Not in ground truth")
                elif has_ground_truth:
                    # Print as missing from result
                    ground_truth_value = ground_truth_alpha[i, j, k]
                    print(f"gt_alpha_{source}{target}{lag} = {ground_truth_value:.2f} Not in result")

def plot_bias(alpha_bias):
    # Put alpha_bias in diagonal form
    alpha_diag = np.squeeze(alpha_bias, axis=-1)
    
    # Find the absolute maximum value
    abs_max_value = np.max(np.abs(alpha_diag))
    # Plot the image with the seismic colormap
    # Also PiYG as cmap
    plt.imshow(alpha_diag, cmap='seismic', vmin=-abs_max_value, vmax=abs_max_value)
    ax = plt.gca()
    
    # Round values to 2 decimal places
    alpha_rounded = np.round(alpha_diag, 2)
    
    # Add text annotations
    for ii in range(alpha_rounded.shape[0]):
        for jj in range(alpha_rounded.shape[1]):
            value = alpha_rounded[ii, jj]
            if value != 0:  # Only show non-zero values
                # Choose text color based on value
                text_color = 'black' if abs(value) < 0.2 else 'white'
                ax.text(jj, ii, f'{value:.2f}', 
                       ha='center', va='center',
                       color=text_color)
    
    # Set the title and ticks
    plt.title(r'$\alpha_\mathrm{bias}$')
    ax.set_xticks(range(alpha_diag.shape[-1]))
    ax.set_xticklabels(range(1, alpha_diag.shape[-1]+1))
    ax.set_yticks(range(alpha_bias.shape[0]))
    ax.set_yticklabels(range(1, alpha_diag.shape[0]+1))
    
    # Show colorbar and the plot
    plt.colorbar();


def plot_alphas(alpha, ground_truth_alpha, cmap='seismic', figsize=(6, 5), font_size=14, title=r'\alpha',space=0.1,cbar_font=11, force_12=False):
    """
    Plot alpha matrices and ground truth alpha matrices with full-height colorbars.

    Args:
        alpha (np.ndarray): Input array of shape (n_matrices, height, width).
        ground_truth_alpha (np.ndarray): Ground truth array of the same shape as alpha.
        cmap (str): Name of the colormap to use ('seismic' or 'PiYG').
        figsize (tuple): Base figure size for each plot.
        font_size (int): Font size for all text in the plots.
        title (str): Title for the plots, defaults to '\alpha'.
        space (float): Space between subplots.
        force_12 (bool): If True, forces a 1x2 layout regardless of data shape.
    """
    ground_truth_alpha = ground_truth_alpha.numpy()
    n_ts, _, w_l = ground_truth_alpha.shape
    for i in range(alpha.shape[0]):
        if force_12 or w_l <= n_ts:
            fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]), gridspec_kw={'wspace': space})
        else:
            fig, axes = plt.subplots(2, 1, figsize=(figsize[0], 1.5 * figsize[1]), gridspec_kw={'hspace': space})

        for idx, (data, title_prefix, needs_flip) in enumerate(zip(
            [alpha[i], ground_truth_alpha[i]],
            [rf'${title}_{{{i+1},j,l}}$', rf'Ground truth ${title}_{{{i+1},j,l}}$'],
            [True, False]  # Flip only `alpha`, not `ground_truth_alpha`
        )):
            ax = axes[idx]

            # Flip the data if needed
            if needs_flip:
                if data.ndim < 2:
                    raise ValueError(f"Expected 2D data, got {data.ndim}D data with shape {data.shape}")
                data = np.flip(data, axis=1)

            # Find the absolute maximum value
            abs_max_value = np.max(np.abs(data))

            # Create the main image
            im = ax.imshow(data,
                           cmap=cmap,
                           vmin=-abs_max_value,
                           vmax=abs_max_value)

            # Round values to 2 decimal places
            data_rounded = np.round(data, 2)

            # Add text annotations
            for ii in range(data_rounded.shape[0]):
                for jj in range(data_rounded.shape[1]):
                    value = data_rounded[ii, jj]
                    if value != 0:  # Only show non-zero values
                        # Choose text color based on value
                        text_color = 'black' if abs(value) < 0.2 * abs_max_value else 'white'
                        ax.text(jj, ii, f'{value:.2f}',
                                ha='center', va='center',
                                color=text_color, fontsize=font_size)

            # Set the title and ticks
            ax.set_title(title_prefix, fontsize=font_size + 2)
            ax.set_xlabel('$l$', fontsize=font_size)
            ax.set_ylabel('$j$', rotation=0, fontsize=font_size)
            ax.set_xticks(range(data.shape[-1]))
            ax.set_xticklabels(range(1, data.shape[-1] + 1), fontsize=font_size - 2)
            ax.set_yticks(range(data.shape[0]))
            ax.set_yticklabels(range(1, data.shape[0] + 1), fontsize=font_size - 2)

            # Add colorbar with the same height as plot
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)

            # Set colorbar ticks to 1 decimal place
            cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
            cbar.ax.tick_params(labelsize=cbar_font)
        
        # Adjust layout and display
        # plt.tight_layout()
        plt.show()
        plt.close()

def plot_beta(beta,ground_truth_beta):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'wspace': 0.2})
    
    # Plot beta
    im1 = axes[0].imshow(beta, cmap=plt.cm.binary)
    axes[0].set_title(r'$\beta$', fontsize=14)
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    beta_rounded = np.round(beta, 3)
    for ii in range(beta_rounded.shape[0]):
        for jj in range(beta_rounded.shape[1]):
            value = beta_rounded[ii, jj]
            if value != 0:
                text_color = 'black' if abs(value) < 0.3 else 'white'
                axes[0].text(jj, ii, f'{value:.2f}', ha='center', va='center', color=text_color, fontsize=14)
    
    axes[0].set_xticks(range(beta.shape[0]))
    axes[0].set_xticklabels(range(1, beta.shape[0] + 1), fontsize=14)
    axes[0].set_yticks(range(beta.shape[0]))
    axes[0].set_yticklabels(range(1, beta.shape[0] + 1), fontsize=14)
    
    # Plot ground_truth_beta
    im2 = axes[1].imshow(ground_truth_beta, cmap=plt.cm.binary)
    axes[1].set_title(r'Ground truth $\beta$', fontsize=14)
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    ground_truth_beta_rounded = np.round(ground_truth_beta, 3)
    for ii in range(ground_truth_beta_rounded.shape[0]):
        for jj in range(ground_truth_beta_rounded.shape[1]):
            value = ground_truth_beta_rounded[ii, jj]
            if value != 0:
                text_color = 'black' if abs(value) < 0.3 else 'white'
                axes[1].text(jj, ii, f'{value:.2f}', ha='center', va='center', color=text_color, fontsize=14)
    
    axes[1].set_xticks(range(ground_truth_beta.shape[0]))
    axes[1].set_xticklabels(range(1, ground_truth_beta.shape[0] + 1), fontsize=14)
    axes[1].set_yticks(range(ground_truth_beta.shape[0]))
    axes[1].set_yticklabels(range(1, ground_truth_beta.shape[0] + 1), fontsize=14);


def evaluate_window_sizes(time_series, window_sizes, order, config, temperature=1., verbose=True):
    """
    Evaluate test losses for multiple window sizes.

    Args:
        time_series (torch.Tensor): Input time series data.
        window_sizes (list): List of window sizes to evaluate.
        order (list): Model order for evaluation.
        config (dict): Configuration parameters for the model.
        temperature (float, optional): Temperature parameter. Default is 1.0.
        verbose (bool, optional): Whether to print progress and results. Default is True.

    Returns:
        dict: Dictionary of test losses for each window size.
    """
    test_losses = {}
    for window_size in window_sizes:
        if verbose:
            print(f"Evaluating window size: {window_size}")
        test_loss, train_losses, val_losses, _, _, _, _, _, _ = train_and_evaluate(time_series, 
                                                                          window_size=window_size,
                                                                          temperature=temperature,
                                                                          order=order,
                                                                          config=config)
        test_losses[window_size] = test_loss
        if verbose:
            print(f"Window Size: {window_size}, Test Loss: {test_loss:.6f}")
    return test_losses

def calculate_multiple_run_statistics(results: Dict) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Calculate mean and standard deviation of alpha, alpha_bias, beta, f_means, and c_means across all runs.

    Args:
        results (Dict): Results dictionary from collect_multiple_runs.

    Returns:
        Dict[str, Dict[str, np.ndarray]]: Dictionary containing the mean and standard deviation of
                                          alpha, alpha_bias, beta, f_means, and c_means for each key.
    """
    # Get number of runs
    run_keys = [k for k in results.keys() if k.startswith('run_')]
    n_runs = len(run_keys)

    # Prepare output dictionary
    stats = {
        'alpha': {},
        'alpha_bias': {'mean': None, 'std': None},
        'beta': {},
        'f': {},
        'c': {}
    }

    # Collect alpha keys
    alpha_keys = results['run_1']['alpha'].keys()

    # Process each alpha key
    for key in alpha_keys:
        # Get shape from the first run for the current key
        alpha_shape = results['run_1']['alpha'][key].shape

        # Stack all alphas for the current key
        alphas = np.stack([results[f'run_{i+1}']['alpha'][key] for i in range(n_runs)], axis=0)

        # Calculate beta for each run
        betas = np.empty((n_runs,) + alpha_shape[:-1])  # Shape: (n_runs, height, width)
        for i in range(n_runs):
            beta_tilde = np.abs(alphas[i]).sum(axis=-1)  # Sum over the last axis
            betas[i] = beta_tilde / beta_tilde.sum(axis=1, keepdims=True)  # Normalize

        # Calculate mean and std across runs (axis 0)
        stats['alpha'][key] = {
            'mean': np.mean(alphas, axis=0),
            'std': np.std(alphas, axis=0)
        }
        stats['beta'][key] = {
            'mean': np.mean(betas, axis=0),
            'std': np.std(betas, axis=0)
        }

    # Process alpha_bias
    alpha_biases = np.stack([results[f'run_{i+1}']['alpha_bias'] for i in range(n_runs)], axis=0)
    stats['alpha_bias']['mean'] = np.mean(alpha_biases, axis=0)
    stats['alpha_bias']['std'] = np.std(alpha_biases, axis=0)

    # Process f_means and c_means
    for key in alpha_keys:  # Same keys as alpha
        f_means = np.stack([results[f'run_{i+1}']['f_means'][key] for i in range(n_runs)], axis=0)
        c_means = np.stack([results[f'run_{i+1}']['c_means'][key] for i in range(n_runs)], axis=0)

        stats['f'][key] = {
            'mean': np.mean(f_means, axis=0),
            'std': np.std(f_means, axis=0)
        }
        stats['c'][key] = {
            'mean': np.mean(c_means, axis=0),
            'std': np.std(c_means, axis=0)
        }

    return stats
