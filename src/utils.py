import pickle
import torch
import datetime as dt
import numpy as np
import pandas as pd

from torch.utils.data import Dataset,DataLoader,random_split, Subset
from torch.utils.tensorboard import SummaryWriter
from src.dcits import DCITS
from tqdm.autonotebook import tqdm
import os
    
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        """
        Custom dataset for time series data.

        Args:
            X (torch.Tensor): Input features.
            y (torch.Tensor): Target values.
        """
        self.X = X
        self.y = y
        
    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return self.X.shape[0]

    def __getitem__(self, idx):
        """
        Retrieves the sample at the given index.

        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            tuple: (X[idx], y[idx])
        """
        return self.X[idx], self.y[idx]

def model_data_from_time_series(time_series, device, remove_first=0, window_len=10, mask_lag=0):
    """
    Prepares model data from time series.

    Args:
        time_series (np.ndarray): The time series data.
        device (torch.device): The device to store the tensors.
        remove_first (int): Number of initial timesteps to remove.
        window_len (int): Length of the window to use for creating samples.
        mask_lag (int): Lag to mask in the input window.
    
    Returns:
        tuple: (X, y) where X is the input tensor and y is the target tensor.
    
    Raises:
        ValueError: If mask_lag is not smaller than window_len or equal to zero.
    """
    
    if type(time_series)==np.ndarray:
        time_series=torch.tensor(time_series)    
        
    time_series = time_series[:, remove_first:]
    num_of_timeseries, len_of_timeseries = time_series.shape
    num_of_examples = len_of_timeseries - window_len
    
    X = torch.empty((num_of_examples, 1, num_of_timeseries, window_len)).normal_(mean=0, std=1)
    y = torch.zeros(num_of_examples, num_of_timeseries)
    
    if mask_lag < 0 or mask_lag >= window_len:
        raise ValueError('Mask lag should be nonnegative and smaller than window length.')
    
    for i in range(num_of_examples):
        X[i, 0, :, :] = time_series[:, i:(i + window_len)]
        if mask_lag > 0:
            X[i, 0, :, :mask_lag] = 0
        y[i, :] = time_series[:, i + window_len]
    
    X = X.to(device)
    y = y.to(device)
    
    return X, y


def train_model(model, train_one_epoch, valid_one_epoch, epochs=10, save_folder='./outputs/runs/', enable_writer=True, save_model=True,verbose=True):
    """
    Trains the model and optionally saves the best performing model.

    Args:
        model (torch.nn.Module): The model to train.
        train_one_epoch (function): Function to train the model for one epoch.
        valid_one_epoch (function): Function to validate the model for one epoch.
        epochs (int): Number of epochs to train the model.
        save_folder (str): Folder to save the trained model and logs.
        enable_writer (bool): Flag to enable/disable tensorboard writer.
        save_model (bool): Flag to enable/disable model saving.

    Returns:
        torch.nn.Module: The trained model.
    """
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if enable_writer:
        writer = SummaryWriter(os.path.join(save_folder, f'DCITS_{timestamp}'))
    else:
        writer = None
    
    epoch_number = 0
    best_vloss = float('inf')

    for epoch in range(epochs):
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}")

        # Training phase
        model.train()
        model, avg_loss = train_one_epoch(model, epoch_number, writer)
        
        if verbose:
            print(f"  Training Loss: {avg_loss:.8f}")

        # Validation phase
        model.eval()
        with torch.no_grad():
            running_vloss, v_batches = valid_one_epoch(model)
        avg_vloss = running_vloss / (v_batches + 1)
        if verbose:
            print(f"  Validation Loss: {avg_vloss:.8f}")
        # Logging
        if enable_writer:
            writer.add_scalars('Training vs. Validation Loss',
                               {'Training': avg_loss, 'Validation': avg_vloss},
                               epoch_number + 1)
            writer.flush()

        # Save the best model
        if save_model and avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = os.path.join(save_folder, f'model_{timestamp}_{epoch_number}.pt')
            torch.save(model.state_dict(), model_path)
            if verbose:
                print(f"  Saved best model to {model_path}")

        epoch_number += 1
    
    if enable_writer:
        writer.close()
        
    return model


def get_data_loaders_and_eval_datasets(X, y, optimizer, loss_fn, batch_size=128,shuffle=True,return_train_valid_dataset=False):
    """
    Prepares data loaders and evaluation datasets.

    Args:
        X (torch.Tensor): Input features.
        y (torch.Tensor): Target values.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        loss_fn (callable): Loss function for training and validation.
        batch_size (int): Batch size for the data loaders.
        shuffle (bool): If datasets will be shuffled.
        return_train_valid_dataset (bool): If train_dataset and valid_dataset should be also returned.

    Returns:
        tuple: (train_one_epoch, valid_one_epoch, test_dataset)
    """
    # Create dataset
    dataset = TimeSeriesDataset(X, y)

    # Split dataset into training, validation, and test sets
    train_size = int(0.75 * len(dataset))
    valid_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    
    if shuffle:
        train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])
    
    else:
        train_dataset = torch.utils.data.Subset(dataset, range(train_size))
        valid_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + valid_size))
        test_dataset = torch.utils.data.Subset(dataset, range(train_size + valid_size, train_size + valid_size + test_size))

    
    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    validation_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    def train_one_epoch(model, epoch_index, tb_writer=None):
        """
        Trains the model for one epoch.

        Args:
            model (torch.nn.Module): The model to train.
            epoch_index (int): Current epoch index.
            tb_writer (SummaryWriter): TensorBoard writer for logging.

        Returns:
            float: Average loss for the epoch.
        """
        model.train()
        running_loss = 0.0

        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs, batch_f, batch_c = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 10 == 0:
                last_loss = running_loss / (i + 1)
                if tb_writer:
                    tb_x = epoch_index * len(train_dataloader) + i + 1
                    tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.0
        return model, last_loss
    
    
    def valid_one_epoch(model):
        """
        Validates the model for one epoch.

        Args:
            model (torch.nn.Module): The model to validate.

        Returns:
            tuple: (running_vloss, number_of_batches)
        """
        model.eval()
        running_vloss = 0.0

        with torch.no_grad():
            for i, vdata in enumerate(validation_dataloader):
                vinputs, vlabels = vdata
                voutputs, vbatch_f, vbatch_c = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss.item()

        return running_vloss, i
    
    if return_train_valid_dataset:
        return train_one_epoch, valid_one_epoch, train_dataset,valid_dataset, test_dataset
    else:
        return train_one_epoch, valid_one_epoch, test_dataset


def evaluate_model(model, train_dataset, valid_dataset, loss_function, run_num=0, mask_lag=0,temperature=1, save_coefficients=False, folder_name='./outputs/coeffs/'):
    """
    Evaluates the model on training and validation datasets.

    Args:
        model (torch.nn.Module): The model to evaluate.
        train_dataset (torch.utils.data.Dataset): The training dataset.
        valid_dataset (torch.utils.data.Dataset): The validation dataset.
        loss_function (callable): The loss function to use for evaluation.
        run_num (int): The run number for identification in saved files.
        mask_lag (int): The lag to mask in the input window.
        save_coefficients (bool): Flag to save the coefficients a and c.
        folder_name (str): Folder to save the coefficients if required.

    Returns:
        tuple: (loss_train, loss_valid) containing the training and validation losses.
    """
    model.eval()

    train_inputs = train_dataset.dataset.X[train_dataset.indices]
    train_labels = train_dataset.dataset.y[train_dataset.indices]
    train_pred, _, _ = model(train_inputs)
    loss_train = loss_function(train_pred, train_labels)

    # Evaluate on validation dataset
    valid_inputs = valid_dataset.dataset.X[valid_dataset.indices]
    valid_labels = valid_dataset.dataset.y[valid_dataset.indices]
    valid_pred, f, c = model(valid_inputs)
    loss_valid = loss_function(valid_pred, valid_labels)

    if save_coefficients:
        # Create a unique filename for saving the coefficients
        timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        save_string = f"dcits_{timestamp}_run{run_num}_mask{mask_lag}_temperature{temperature}_a_c.p"

        # Ensure the folder exists
        os.makedirs(folder_name, exist_ok=True)

        # Save the coefficients
        with open(os.path.join(folder_name, save_string), 'wb') as file:
            pickle.dump((f, c), file)
    
    return loss_train, loss_valid


def train_and_eval(path_string, output_folder='./outputs/coeffs/', epochs=10, window_len=10,
                    run_num=0, device=None, temperature=1,in_ch=16,  mask_lag=0, remove_first=100,
                   learning_rate=0.001, weight_decay=0.0001,batch_size=128,shuffle=True, save_coefficients=False,
                  save_coeffs_on_test_set=False):
    """
    Trains and evaluates the DCITS model on time series data from a specified file.

    Args:
        path_string (str): Path to the pickle file containing the time series data.
        output_folder (str, optional): Folder to save the output coefficients. Default is './outputs/coeffs/'.
        epochs (int, optional): Number of epochs to train the model. Default is 10.
        window_len (int, optional): The length of the window to use for creating samples. Default is 10.
        run_num (int, optional): Run number for saving output files. Default is 0.
        device (torch.device, optional): The device to use for computation (CPU or CUDA). Default is GPU if available, otherwise CPU.
        temperature (float, optional): Temperature parameter for the softmax operation. Default is 1.
        in_ch (int, optional): Number of input channels for the convolutional layers. Default is 16.
        mask_lag (int, optional): Lag to mask in the input window. Default is 0.
        remove_first (int, optional): Number of initial timesteps to remove from the time series. Default is 100.
        learning_rate (float, optional): Learning rate for the optimizer. Default is 0.001.
        weight_decay (float, optional): Weight decay (L2 regularization) for the optimizer. Default is 0.0001.
        batch_size (int, optional): Batch size for the DataLoader. Default is 128.
        shuffle (bool, optional): Whether to shuffle the datasets in the DataLoader. Default is True.
        save_coefficients (bool): Flag if focuser and modeler tensors calcualted on valid dataset should be saved
        save_coeffs_on_test_set (bool): Flag if focuser and modeler tensors calcualted on test set should be saved

    Returns:
        tuple: (train_score, valid_score) containing the training and validation scores.
    """
    
 # Set the device if not provided
    if device is None:
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    # Load time series data
    with open(path_string, 'rb') as f:
        time_series = pickle.load(f)
    num_of_time_series = time_series.shape[0]
    
    # Initialize the model
    model = DCITS(num_of_time_series, window_len, in_ch=in_ch, temp=temperature)
    model = model.to(device)

    # Prepare the data
    X, y = model_data_from_time_series(time_series, device, remove_first=remove_first, window_len=window_len, mask_lag=mask_lag)

    # Set up the loss function and optimizer
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
 
    # Get data loaders and evaluation datasets
    train_one_epoch, valid_one_epoch,  train_dataset,valid_dataset, test_dataset=get_data_loaders_and_eval_datasets(X,y,optimizer,loss_fn,batch_size=batch_size,shuffle=shuffle,return_train_valid_dataset=True)
    
    model=train_model(model,train_one_epoch,valid_one_epoch,epochs=epochs,enable_writer=False, save_model=False,verbose=False)
    
    train_score,valid_score=evaluate_model(model,train_dataset, valid_dataset,loss_fn, run_num=run_num, mask_lag=mask_lag, temperature=temperature, save_coefficients=save_coefficients,folder_name=output_folder)
    if save_coeffs_on_test_set:
        save_model_coefficients(model,test_dataset,run_num=run_num ,mask_lag=mask_lag, temperature=temperature, folder_name=output_folder)
    
    return train_score,valid_score

def save_model_coefficients(model, test_dataset, run_num, mask_lag, temperature, folder_name):
    """
    Extracts predictions and coefficients from the model using the test dataset and saves them to a file.

    Args:
        model (torch.nn.Module): The trained model used for prediction.
        test_dataset (torch.utils.data.Dataset): The test dataset containing inputs and labels.
        run_num (int): The run number for identification in saved files.
        mask_lag (int): The lag to mask in the input window.
        temperature (float): Temperature parameter used in the experiment.
        folder_name (str): The directory where the coefficients will be saved.
    """
    # Extract inputs and labels from the test dataset
    test_inputs = test_dataset.dataset.X[test_dataset.indices]
    test_labels = test_dataset.dataset.y[test_dataset.indices]
    
    # Perform predictions with the model
    test_pred, f, c = model(test_inputs)

    # Create a unique filename for saving the coefficients
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_string = f"dcits_testset_{timestamp}_run{run_num}_mask{mask_lag}_temperature{temperature}_a_c.p"

    # Ensure the folder exists
    os.makedirs(folder_name, exist_ok=True)

    # Save the coefficients
    with open(os.path.join(folder_name, save_string), 'wb') as file:
        pickle.dump((f, c), file)

def run_single_experiment(load_string, output_folder,window_len, mask_lag, repeats, temperature,epochs=10,save_coefficients=True,save_coeffs_on_test_set=False):
    """
    Runs the training and evaluation for a single time series and mask lag over multiple repeats.

    Args:
        load_string (str): Path to the time series data file.
        output_folder (str): Folder to save the output coefficients.
        mask_lag (int): The mask lag value to use in the experiment.
        repeats (int): Number of times to repeat the experiment.
        temperature (float): Temperature parameter used in the experiment.
        save_coefficients (bool): Flag if focuser and modeler tensors calcualted on valid dataset should be saved
        save_coeffs_on_test_set (bool): Flag if focuser and modeler tensors calcualted on test set should be saved

    Returns:
        dict: A dictionary containing the time series identifier, mask lag, temperature, and
              the mean and standard deviation of training and test losses.
    """
    experiment_train_loss = []
    experiment_test_loss = []
    
    # Repeat the experiment multiple times
    for i in range(repeats):
        # Train and evaluate the model
        train_loss, test_loss = train_and_eval(load_string, output_folder, run_num=i, window_len=window_len, mask_lag=mask_lag, temperature=temperature, save_coefficients=save_coefficients, save_coeffs_on_test_set=save_coeffs_on_test_set)
        
        # Convert losses to numpy arrays and store them
        experiment_train_loss.append(train_loss.cpu().detach().numpy())
        experiment_test_loss.append(test_loss.cpu().detach().numpy())

    # Return the results as a dictionary
    return {
        'time_series': load_string,
        'mask_lag': mask_lag,
        'temperature': temperature,
        'train_loss_mean': np.mean(experiment_train_loss),
        'train_loss_std': np.std(experiment_train_loss),
        'test_loss_mean': np.mean(experiment_test_loss),
        'test_loss_std': np.std(experiment_test_loss),
    }