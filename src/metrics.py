import numpy as np
import torch
import torch.nn as nn

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target):
        return torch.sqrt(self.mse(pred, target))

class MAPELoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        return torch.mean(torch.abs((target - pred) / target)) * 100

def calculate_metrics(y_true, y_pred):
    """计算多个评估指标"""
    # 转换为numpy数组
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
        
    # MSE
    mse = np.mean((y_true - y_pred) ** 2)
    
    # RMSE
    rmse = np.sqrt(mse)
    
    # MAE
    mae = np.mean(np.abs(y_true - y_pred))
    
    # MAPE
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # R2 Score
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    } 