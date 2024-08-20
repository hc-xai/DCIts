# DCITS - Deep Convolutional Interpreter for Time Series

Welcome to the official repository for **DCITS (Deep Convolutional Interpreter for Time Series)**, a deep learning model designed for interpretable multivariate time series forecasting. This repository accompanies the scientific paper: *"DCIts - Deep Convolutional Interpreter for Time Series"*.

A deep learning model for interpretable multivariate time series forecasting, focusing on prediction accuracy and interpretability. DCITS identifies key time series and lags, providing intuitive explanations for complex systems.

## Abstract

DCITS is an interpretable deep learning model tailored for multivariate time series forecasting. The model architecture is focused on both prediction performance and interpretability, which is crucial for understanding complex physical phenomena. The results demonstrate that DCITS not only matches but also surpasses existing models in terms of interpretability without compromising prediction accuracy. Through comprehensive experiments, DCITS showcases its ability to identify the most relevant time series and lags, offering intuitive explanations for its predictions. The model reduces the need for manual supervision through two innovations: a heuristic for determining the optimal window size in the time domain and an algorithm for ascertaining the optimal model order. These advancements have significant implications for modeling and understanding dynamic physical systems, providing a powerful tool for applied and computational physicists.

## Repository Structure

The repository is organized as follows:
- data/
  - dataset7.pickle  # Example dataset used in the models
  - dataset8.pickle  # Example dataset used in the models
- src/
  - dcits.py         # Model architecture
  - utils.py         # Utility functions
- examples/
  - DCIts_example.ipynb           # Notebook demonstrating how to apply the DCITS model to different timeseries.
  - Higher_order_example.ipynb    # Generalization of DCITS to higher-order, analyzing a cubic map.
  - Optimisation_example.ipynb    # Example of hyperparameter optimisation.

## Installation

To use DCITS, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/DCITS.git
cd DCITS
pip install -r requirements.txt
```

Make sure you have PyTorch installed, as the model relies heavily on it.
This repo uses Python 3.9.7.

## Usage

### Model Overview

DCITS is built around a convolutional neural network (CNN) backbone that processes time series data and produces interpretable results. The model can be extended to higher-order interactions using `DCITSOrder2` and `DCITSOrder3` classes, which apply quadratic and cubic transformations respectively.

### Example Usage

You can find example notebooks in the `examples/` directory that demonstrate how to use the DCITS model:

1. **DCIts_example.ipynb**: Demonstrates how to apply the DCITS model to different timeseries.
2. **Higher_order_example.ipynb**: Shows the generalization of DCITS to higher-order interactions, specifically analyzing a cubic map.
3. **Optimisation_example.ipynb**: Illustrates how to perform hyperparameter optimisation for the DCITS model.

## Contributing

Contributions are welcome! If you have ideas to improve the model, find bugs, or have suggestions for new features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use DCITS in your research, please cite our paper:
```bibtex
@article{your_paper_citation,
  title={DCIts - Deep Convolutional Interpreter for Time Series},
  author={Your Name and Co-Authors},
  journal={Journal Name},
  year={2024},
  volume={X},
  pages={Y-Z},
}
```

## Contact

For any questions or further information, please contact domjanbaric@gmail.com.