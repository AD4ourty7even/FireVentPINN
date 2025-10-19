# PINN Temperature Sensor Analysis

##  Project Overview

Physics-Informed Neural Network (PINN) for analyzing temperature sensor data from 9 sensors positioned in a 3x3 grid. This project implements a neural network with physics constraints to predict temperature distributions and solve the heat equation.

## Features

- **Physics-Informed Neural Network**: Incorporates heat equation constraints
- **Adaptive Weighting**: Dynamic sensor weighting for balanced training
- **Time-dependent Parameters**: Variable thermal diffusivity based on time
- **Comprehensive Visualization**: Multiple plots for analysis and validation
- **Modular Architecture**: Clean, organized code structure

## 📁 Project Structure

```
├── main.py                     # Main execution script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/                       # Data files
│   ├── devc.csv
│   ├── devpos.json
│   ├── dfdevc.csv
│   └── dfdevc_norm.csv
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── pinn_model.py          # PINN model implementation
│   ├── data_processing.py     # Data loading and preprocessing
│   ├── training.py            # Training functions
│   └── visualization.py       # Plotting and visualization
├── notebooks/                  # Jupyter notebooks
│   ├── analysis.ipynb
│   ├── tuning1.ipynb
│   ├── tuning2.ipynb
│   ├── ad.ipynb
│   └── gpt.ipynb
└── results/                    # Output files
    ├── plots/                  # Generated visualizations
    ├── models/                 # Saved model weights
    └── *.csv, *.json          # Results and logs
```






## 🔧 Final Tuned Hyperparameters

The model uses the following optimized hyperparameters from extensive tuning:

```python
config = {
    'lr': 0.001,
    'layers': [3, 128, 128, 128, 1],
    'activation': "tanh",
    'epochs': 500,
    'w_data': 0.9,      # Data loss weight
    'w_pde': 1e-3,      # PDE loss weight
    'pde_weight_min': 1e-6,
    'pde_weight_max': 0.05,
    'pde_ramp_power': 2.0
}
```

##  Results

The model achieves:
- **RMSE**: 11.7


##  Methodology

### Physics-Informed Neural Network
- **Input**: Spatial coordinates (x, z) and time (t)
- **Output**: Temperature prediction
- **Physics Constraint**: Heat equation with time-dependent diffusivity

### Heat Equation
```
∂T/∂t = α(t) * (∂²T/∂x² + ∂²T/∂z²)
```

Where:
- `α(t) = 2e-5` for t ≤ 15s
- `α(t) = 1.9e-5` for t > 15s

### Training Strategy
1. **Data Loss**: MSE between predicted and actual temperatures
2. **Physics Loss**: PDE residual minimization
3. **Adaptive Weighting**: Dynamic sensor weighting based on prediction errors
4. **Gradient Clipping**: Prevents exploding gradients



##  Research Context

This project demonstrates the application of Physics-Informed Neural Networks to solve partial differential equations in engineering applications. The methodology combines:

- **Data-driven learning** from sensor measurements
- **Physics constraints** from the heat equation
- **Adaptive optimization** for improved convergence





---

**Note**: This project was developed for academic research and demonstrates advanced machine learning techniques for solving physics problems.
