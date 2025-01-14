# DCIts - Deep Convolutional Interpreter for Time Series

Welcome to the official repository for **DCIts (Deep Convolutional Interpreter for Time Series)**, an advanced deep learning model developed for interpretable multivariate time series forecasting. This repository accompanies the research paper: *"DCIts - Deep Convolutional Interpreter for Time Series"*.

**DCIts** is specifically tailored for multivariate time series forecasting, emphasizing both predictive performance and interpretability. Its architecture is designed to enhance understanding of complex physical phenomena, as well as other dynamical systems that can be effectively modeled using multivariate time series. **DCIts** excels in prediction accuracy while providing high interpretability, enabling the identification of key time series and their corresponding lags that contribute to future value forecasts, and offering clear, intuitive explanations of complex systems.

Experimental results indicate that **DCIts** not only matches but often surpasses existing models in interpretability, without compromising prediction accuracy. Through extensive experimentation, **DCIts** demonstrates its ability to identify the most relevant time series and time lags of the underlying generative process, thereby offering intuitive explanations for its predictions. To minimize the need for manual supervision, the model is designed so one can robustly determine the optimal window size that captures all necessary interactions within the smallest possible time frame. Additionally, it effectively identifies the optimal model order, balancing complexity when incorporating higher-order terms.

## Repository Structure

The repository is organized as follows:

```
- src/
  - dcits.py         # Model architecture implementation
  - utils.py         # Utility functions
- examples/          # Notebook demonstrating how to use DCIts on various time series.
```

## Installation

To use **DCIts**, clone the repository and install the required dependencies:

```bash
git clone https://github.com/hc-xai/DCIts.git
cd DCIts
```

Ensure that you have **PyTorch** installed, as it is a core dependency for this model. This repository uses **Python 3.9.7** and **PyTorch 2.5**. Final code was tested in Jupyter lab enviroment created as:

```bash
conda create --name DCIts python=3.9 -y
conda activate DCIts
onda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install matplotlib
conda install pandas -c conda-forge
conda install tensorboard -c conda-forge
conda install tqdm  -c conda-forge
```
If you are using Jupyter Lab, install `ipykernel` in the DCIts env and add kernel to Jupyter:

```bash
conda install ipykernel
python -m ipykernel install --user --name=DCIts --display-name "Python (DCIts)"
```

We also provide requirements.txt file for pip installation:

```bash
pip install -r requirements.txt
```

## Key Components

### DCITS Model (`dcits.py`)

#### Classes

1. **`FlattenLinearReshape`**:
   - A custom PyTorch module that flattens input tensors, applies a linear transformation, and reshapes the output.
   - Designed for use as part of activation layers.

2. **`Backbone`**:
   - Implements a neural network backbone with convolutional and fully connected layers.
   - Supports multiple kernel sizes to capture various interaction patterns.
   - Can output activations with either linear or sigmoid transformations, depending on the use case.

3. **`DCITS`**:
   - The main model class that orchestrates multiple `Backbone` modules to compute dependencies for different orders of interactions (e.g., bias, linear, quadratic).
   - Flexible order specification to include or exclude specific interaction terms.
   - Implements forward propagation to compute interaction coefficients and predictions.

### Utility Functions (`utils.py`)

#### Data Preparation

1. **`dataset`**:
   - Generates synthetic multivariate time series data with customizable interaction coefficients, biases, noise, and non-linearities.

2. **`split_time_series`**:
   - Splits a multivariate time series into training, validation, and testing sets without overlap.

3. **`create_windowed_dataset`**:
   - Transforms time series data into sliding windows for supervised learning.

#### Model Training and Evaluation

4. **`train_and_evaluate`**:
   - Trains the DCITS model using specified parameters and evaluates performance on validation and test sets.
   - Supports early stopping, learning rate scheduling, and debug mode for detailed logging.

5. **`collect_multiple_runs`**:
   - Runs multiple training experiments to evaluate robustness and variability.
   - Aggregates and summarizes performance metrics across runs.

6. **`evaluate_window_sizes`**:
   - Tests the effect of different window sizes on model performance.

#### Results Interpretation

7. **`plot_ts`**:
   - Visualizes time series data.

8. **`plot_alphas`** and **`plot_beta`**:
   - Visualizes learned interaction coefficients and their ground truth counterparts.

9. **`print_significant_alpha`** and **`print_bias`**:
   - Formats and compares learned coefficients and biases with ground truth values, highlighting significant interactions.

## Getting Started

1. **Usage**:
   - Define your multivariate time series or generate synthetic data using `dataset()`.
   - Train the DCITS model using `train_and_evaluate()` or `collect_multiple_runs()`.
   - Visualize and interpret the results using the plotting and summary utilities.

2. **Customization**:
   - Adjust model hyperparameters to fit your data.
   - Add custom nonlinear functions or alternative loss metrics if needed.

## Example Workflow

```python
from dcits import DCITS
from utils import dataset, train_and_evaluate, plot_ts

# Generate synthetic data
ground_truth_alpha = torch.rand(5, 5, 3)  # 5 time series, window length 3
ground_truth_alpha[ground_truth_alpha < 0.7] = 0  # Make ground truth sparse
bias = torch.rand(5)  # Bias for each series
time_series = dataset(1000, ground_truth_alpha, bias)

# Train the model
config = {
    'learning_rate': 1e-3,
    'epochs': 50,
    'batch_size': 32
}
test_loss, train_losses, val_losses, f_test, c_test, debug_info, *_ = train_and_evaluate(
    time_series, window_size=3, temperature=1.0, order=[1, 1], config=config
)
```

### Using `order`

The `order=[1, 1]` configuration specifies that the model should include both bias terms and linear interaction terms:

- **Bias terms** (order 0): These represent constant effects that are independent of input values. They are particularly useful for modeling static shifts or baseline levels in the time series.
- **Linear interaction terms** (order 1): These capture pairwise linear dependencies between time series components. This is crucial for understanding direct relationships and their dynamics.

By using `order=[1, 1]`, the model leverages both the baseline levels and the linear relationships to predict the time series dynamics effectively. For higher-order interactions, you can extend this list (e.g., `order=[1, 1, 1]` to include quadratic terms), or `order=[1, 0, 1]` to include only bias and quadratic term.

### Example notebooks

Example notebooks in the `examples/` directory provide guidance on using **DCIts**:

1. `DCIts-DS*.ipynb`: Notebooks demonstratee how to apply **DCIts** to multivariate time series.
2. `DCIts-cubic-*.ipynb`: Explains the generalization of **DCIts** to higher-order interactions, including the analysis of a cubic map-inspired process.

Last suffix in notebooke names determines loss function used, L1 - MAE, L2 - MSE.

## Contributing

We welcome contributions! If you have suggestions for improvements, encounter any bugs, or have ideas for new features, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the **MIT License** - see the LICENSE file for details.

## Citation

If you use **DCIts** in your research, please consider citing our paper:

```bibtex
@article{dcits2024,
  title={DCIts - Deep Convolutional Interpreter for Time Series},
  author={Domjan Barić, Davor Horvatić},
  journal={arXiv preprint arXiv:2501.04339},
  year={2024},
  note={\url{https://arxiv.org/abs/2501.04339}}
}
```

## Contact

For any questions or further information, feel free to contact **davorh@phy.hr** and **dbaric.phy@pmf.hr**.