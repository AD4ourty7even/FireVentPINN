"""
PINN Model Implementation
Physics-Informed Neural Network for temperature prediction
"""

import torch
import torch.nn as nn
import numpy as np

class PINN(nn.Module):
    """
    Physics-Informed Neural Network for temperature analysis
    
    Args:
        layers (list): List of layer sizes [input, hidden1, hidden2, ..., output]
        activation (str): Activation function ('tanh' or 'silu')
    """
    
    def __init__(self, layers, activation="tanh"):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Create linear layers
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
        # Set activation function
        if activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input tensor [x, z, t]
            
        Returns:
            torch.Tensor: Predicted temperature
        """
        out = x
        for layer in self.layers[:-1]:
            out = self.act(layer(out))
        return self.layers[-1](out)

def pde_residual(model, xyz, t_max):
    """
    Calculate PDE residual for heat equation
    
    Args:
        model: PINN model
        xyz (torch.Tensor): Input coordinates [x, z, t]
        t_max (float): Maximum time for normalization
        
    Returns:
        torch.Tensor: PDE residual
    """
    xyz.requires_grad_(True)
    T_pred = model(xyz)
    
    # First derivatives
    grads = torch.autograd.grad(
        T_pred, xyz, 
        grad_outputs=torch.ones_like(T_pred),
        retain_graph=True, create_graph=True
    )[0]
    
    T_x = grads[:, 0:1]
    T_z = grads[:, 1:2] 
    T_t = grads[:, 2:3]
    
    # Second derivatives
    T_xx = torch.autograd.grad(
        T_x, xyz, 
        grad_outputs=torch.ones_like(T_x),
        retain_graph=True, create_graph=True
    )[0][:, 0:1]
    
    T_zz = torch.autograd.grad(
        T_z, xyz, 
        grad_outputs=torch.ones_like(T_z),
        retain_graph=True, create_graph=True
    )[0][:, 1:2]
    
    # Dynamic alpha based on real time
    t_real = xyz[:, 2] * t_max
    alpha = torch.where(
        t_real <= 15.0, 
        torch.tensor(2e-5), 
        torch.tensor(1.9e-5)
    )
    alpha = alpha.unsqueeze(1)
    
    # Heat equation: ∂T/∂t = α(∂²T/∂x² + ∂²T/∂z²)
    return T_t - alpha * (T_xx + T_zz)
