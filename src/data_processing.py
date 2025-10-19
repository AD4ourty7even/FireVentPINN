"""
Data Processing Functions
Load, preprocess, and prepare data for PINN training
"""

import pandas as pd
import numpy as np
import json
import torch
from pathlib import Path

def load_data(data_dir="data"):
    """
    Load sensor positions and temperature data
    
    Args:
        data_dir (str): Directory containing data files
        
    Returns:
        tuple: (sensor_positions, df_full)
    """
    data_path = Path(data_dir)
    
    # Load sensor positions
    with open(data_path / "devpos.json", "r") as f:
        sensor_positions = json.load(f)
    
    # Load temperature data
    data = pd.read_csv(data_path / "dfdevc.csv")
    
    # Create full dataset with coordinates
    records = []
    for sensor, pos in sensor_positions.items():
        x, z = pos
        for _, row in data.iterrows():
            t = row["Time"]
            T = row[sensor]
            records.append([sensor, x, z, t, T])
    
    df_full = pd.DataFrame(records, columns=["sensor", "x", "z", "t", "T"])
    
    return sensor_positions, df_full

def prepare_data(df, use_log=True):
    """
    Prepare data for training
    
    Args:
        df (pd.DataFrame): Full dataset
        use_log (bool): Whether to use log scaling for temperature
        
    Returns:
        tuple: (X, T, df_processed)
    """
    df = df.copy()
    
    # Scale temperature
    if use_log:
        df["T_scaled"] = np.log1p(df["T"])
    else:
        df["T_scaled"] = df["T"]
    
    # Normalize coordinates
    df["x_norm"] = (df["x"] - df["x"].min()) / (df["x"].max() - df["x"].min())
    df["z_norm"] = (df["z"] - df["z"].min()) / (df["z"].max() - df["z"].min())
    df["t_norm"] = df["t"] / df["t"].max()
    
    # Convert to tensors
    X = torch.tensor(df[["x_norm", "z_norm", "t_norm"]].values, dtype=torch.float32)
    T = torch.tensor(df[["T_scaled"]].values, dtype=torch.float32)
    
    return X, T, df

def get_sensor_weights(df):
    """
    Calculate sensor weights for balanced training
    
    Args:
        df (pd.DataFrame): Processed dataframe
        
    Returns:
        tuple: (sensor_weights_dict, weights_tensor)
    """
    sensor_counts = df.groupby("sensor").size()
    weights = {s: 1.0/len(sensor_counts) for s in sensor_counts.keys()}
    weights_tensor = torch.tensor(
        df["sensor"].map(weights).values, 
        dtype=torch.float32
    ).unsqueeze(1)
    
    return weights, weights_tensor
