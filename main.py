#!/usr/bin/env python3
"""
PINN Temperature Sensor Analysis - Main Execution Script
Final tuned hyperparameters from tuning2.ipynb
"""

import torch
import numpy as np
import argparse
import os
from pathlib import Path

# Import custom modules
from src.pinn_model import PINN
from src.data_processing import load_data, prepare_data
from src.training import train_model
from src.visualization import plot_results, save_results

def main():
    """Main execution function with final tuned hyperparameters"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='PINN Temperature Analysis - Final Tuned Model')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--save_model', action='store_true', help='Save trained model')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/plots", exist_ok=True)
    os.makedirs(f"{args.output_dir}/models", exist_ok=True)
    
    # Final tuned hyperparameters (from tuning2.ipynb)
    config = {
        'lr': args.lr,
        'layers': [3, 128, 128, 128, 1],
        'activation': "tanh",
        'epochs': args.epochs,
        'seed': 42,
        'use_log': True,
        'w_data': 0.9,  # Best weight from tuning
        'w_pde': 1e-3,  # Best weight from tuning
        'pde_weight_min': 1e-6,
        'pde_weight_max': 0.05,
        'pde_ramp_power': 2.0
    }
    
    print("🚀 PINN Temperature Analysis - Final Tuned Model")
    print("=" * 50)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 50)
    
    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Load and prepare data
    print("\n📊 Loading and preparing data...")
    sensor_positions, df_full = load_data()
    X, T, df_processed = prepare_data(df_full, use_log=config['use_log'])
    
    print(f"Data shape: {X.shape}")
    print(f"Temperature range: {df_processed['T'].min():.2f} - {df_processed['T'].max():.2f}")
    
    # Initialize model
    print(f"\n🧠 Initializing PINN model...")
    model = PINN(config['layers'], config['activation'])
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print(f"\n🏋️ Training model for {config['epochs']} epochs...")
    model, log = train_model(
        model, X, T, df_processed, 
        config['epochs'], 
        config['w_data'], 
        config['w_pde'],
        config
    )
    
    # Evaluate model
    print(f"\n📈 Evaluating model...")
    model.eval()
    with torch.no_grad():
        T_pred_scaled = model(X).numpy()
        T_pred = np.expm1(T_pred_scaled) if config['use_log'] else T_pred_scaled
        df_processed["T_pred"] = T_pred
    
    # Calculate metrics
    errors = df_processed["T"] - df_processed["T_pred"]
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))
    
    print(f"✅ Final Results:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    
    # Save results
    print(f"\n💾 Saving results to {args.output_dir}/...")
    save_results(
        model, log, df_processed, sensor_positions, 
        config, rmse, mae, args.output_dir
    )
    
    # Generate plots
    print(f"\n📊 Generating visualizations...")
    plot_results(df_processed, sensor_positions, log, rmse, args.output_dir)
    
    if args.save_model:
        model_path = f"{args.output_dir}/models/pinn_final.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")
    
    print(f"\n🎉 Analysis complete! Results saved to {args.output_dir}/")
    print(f"Check {args.output_dir}/plots/ for visualizations")

if __name__ == "__main__":
    main()
