import torch
import torch.nn as nn
import numpy as np

class TimeSeriesEnsemble(nn.Module):
    """时间序列集成模型"""
    
    def __init__(self, models, weights=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights if weights is not None else torch.ones(len(models)) / len(models)
        
    def forward(self, x):
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
            
        # 加权平均
        ensemble_pred = torch.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            ensemble_pred += weight * pred
            
        return ensemble_pred

class StackingEnsemble(nn.Module):
    """Stacking集成模型"""
    
    def __init__(self, base_models, meta_model):
        super().__init__()
        self.base_models = nn.ModuleList(base_models)
        self.meta_model = meta_model
        
    def forward(self, x):
        # 获取基础模型预测
        base_predictions = []
        for model in self.base_models:
            pred = model(x)
            base_predictions.append(pred)
            
        # 合并基础模型预测
        stacked_predictions = torch.cat(base_predictions, dim=-1)
        
        # 元模型预测
        final_prediction = self.meta_model(stacked_predictions)
        
        return final_prediction 