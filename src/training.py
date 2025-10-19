"""
Training Functions
PINN training with physics constraints and adaptive weighting
"""

import torch
import numpy as np
from src.pinn_model import pde_residual

def pde_weight_schedule(epoch, epochs, wmin, wmax, power):
    """
    Calculate PDE weight schedule
    
    Args:
        epoch (int): Current epoch
        epochs (int): Total epochs
        wmin (float): Minimum weight
        wmax (float): Maximum weight
        power (float): Ramp power
        
    Returns:
        float: Current PDE weight
    """
    frac = (epoch / epochs) ** power
    return wmin + (wmax - wmin) * frac

def train_model(model, X, T, df, epochs, w_data, w_pde, config):
    """
    Train PINN model with physics constraints
    
    Args:
        model: PINN model
        X (torch.Tensor): Input coordinates
        T (torch.Tensor): Target temperatures
        df (pd.DataFrame): Processed dataframe
        epochs (int): Number of training epochs
        w_data (float): Data loss weight
        w_pde (float): PDE loss weight
        config (dict): Configuration parameters
        
    Returns:
        tuple: (trained_model, training_log)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    # Initialize tracking variables
    running_data, running_pde = 0.0, 0.0
    ema_decay = 0.99
    
    # Get initial sensor weights
    from src.data_processing import get_sensor_weights
    sensor_weights, weights_tensor = get_sensor_weights(df)
    
    # Training log
    log = {
        "epoch": [], "data_loss": [], "pde_loss": [], 
        "total_loss": [], "mean_residual": [], "pde_weight": []
    }
    
    t_max = df["t"].max()
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        T_pred = model(X)
        
        # Update sensor weights every 100 epochs
        if epoch % 100 == 0 and epoch > 0:
            with torch.no_grad():
                df["T_pred_tmp"] = T_pred.numpy()
                for s in sensor_weights.keys():
                    s_data = df[df["sensor"] == s]
                    mse_s = ((s_data["T_scaled"] - s_data["T_pred_tmp"]) ** 2).mean()
                    sensor_weights[s] = 0.5 + mse_s / mse_s.sum()
                
                # Normalize weights
                total = sum(sensor_weights.values())
                sensor_weights = {k: v/total for k, v in sensor_weights.items()}
                weights_tensor = torch.tensor(
                    df["sensor"].map(sensor_weights).values, 
                    dtype=torch.float32
                ).unsqueeze(1)
        
        # Calculate data loss
        data_loss = torch.mean(weights_tensor * (T_pred - T) ** 2)
        
        # Calculate PDE residual
        residual = pde_residual(model, X, t_max)
        pde_weight = pde_weight_schedule(
            epoch, epochs, 
            config['pde_weight_min'], 
            config['pde_weight_max'], 
            config['pde_ramp_power']
        )
        pde_loss = torch.mean(residual ** 2) * pde_weight
        
        # Update running averages
        running_data = ema_decay * running_data + (1 - ema_decay) * data_loss.item()
        running_pde = ema_decay * running_pde + (1 - ema_decay) * pde_loss.item()
        
        # Calculate total loss with normalization
        total_loss = (
            w_data * (data_loss / (running_data + 1e-12)) + 
            w_pde * (pde_loss / (running_pde + 1e-12))
        )
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        # Log metrics
        log["epoch"].append(epoch)
        log["data_loss"].append(data_loss.item())
        log["pde_loss"].append(pde_loss.item())
        log["total_loss"].append(total_loss.item())
        log["mean_residual"].append(residual.abs().mean().item())
        log["pde_weight"].append(pde_weight)
        
        # Print progress
        if epoch % 50 == 0:
            print(f"Epoch {epoch:4d} | Data: {data_loss.item():.6f} | "
                  f"PDE: {pde_loss.item():.6f} | Total: {total_loss.item():.6f}")
    
    return model, log
