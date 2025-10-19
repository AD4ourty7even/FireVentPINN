"""
Visualization Functions
Generate plots and save results
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def plot_results(df, sensor_positions, log, rmse, output_dir):
    """
    Generate and save visualization plots
    
    Args:
        df (pd.DataFrame): Processed dataframe with predictions
        sensor_positions (dict): Sensor position mapping
        log (dict): Training log
        rmse (float): RMSE value
        output_dir (str): Output directory
    """
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    
    # 1. Sensor predictions plot
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    sensors_to_plot = df["sensor"].unique()[:9]
    
    for i, sensor in enumerate(sensors_to_plot):
        x_s, z_s = sensor_positions[sensor]
        s_data = df[(df["x"] == x_s) & (df["z"] == z_s)]
        
        axes[i].plot(s_data["t"], s_data["T"], 'o-', label="Actual", markersize=4)
        axes[i].plot(s_data["t"], s_data["T_pred"], 'x--', label="Predicted", markersize=4)
        axes[i].set_title(f"{sensor} (x={x_s}, z={z_s})")
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Temperature")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle(f"PINN Temperature Predictions | RMSE: {rmse:.4f}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{output_dir}/plots/sensor_predictions.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Training loss plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss curves
    ax1.plot(log["epoch"], log["data_loss"], label="Data Loss", alpha=0.8)
    ax1.plot(log["epoch"], log["pde_loss"], label="PDE Loss", alpha=0.8)
    ax1.plot(log["epoch"], log["total_loss"], label="Total Loss", alpha=0.8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # PDE weight schedule
    ax2.plot(log["epoch"], log["pde_weight"], label="PDE Weight", color='red')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("PDE Weight")
    ax2.set_title("PDE Weight Schedule")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/training_curves.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Error distribution
    errors = df["T"] - df["T_pred"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Error histogram
    ax1.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    ax1.set_xlabel("Prediction Error")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Error Distribution")
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot: Actual vs Predicted
    ax2.scatter(df["T"], df["T_pred"], alpha=0.6, s=20)
    ax2.plot([df["T"].min(), df["T"].max()], [df["T"].min(), df["T"].max()], 'r--', alpha=0.8)
    ax2.set_xlabel("Actual Temperature")
    ax2.set_ylabel("Predicted Temperature")
    ax2.set_title("Actual vs Predicted")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/error_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

def save_results(model, log, df, sensor_positions, config, rmse, mae, output_dir):
    """
    Save results to files
    
    Args:
        model: Trained PINN model
        log (dict): Training log
        df (pd.DataFrame): Processed dataframe
        sensor_positions (dict): Sensor positions
        config (dict): Configuration
        rmse (float): RMSE value
        mae (float): MAE value
        output_dir (str): Output directory
    """
    # Save training log
    log_df = pd.DataFrame(log)
    log_df.to_csv(f"{output_dir}/training_log.csv", index=False)
    
    # Save predictions
    df.to_csv(f"{output_dir}/predictions.csv", index=False)
    
    # Save configuration and results
    results_summary = {
        'rmse': rmse,
        'mae': mae,
        'config': config,
        'sensor_positions': sensor_positions
    }
    
    import json
    with open(f"{output_dir}/results_summary.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"Results saved to {output_dir}/")
    print(f"  - training_log.csv: Training metrics")
    print(f"  - predictions.csv: Model predictions")
    print(f"  - results_summary.json: Configuration and metrics")
