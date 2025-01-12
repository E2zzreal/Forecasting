import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from .metrics import calculate_metrics, RMSELoss, MAPELoss

class ModelTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # 多目标损失函数
        self.mse_criterion = nn.MSELoss()
        self.rmse_criterion = RMSELoss()
        self.mape_criterion = MAPELoss()
        
        # 优化器
        self.optimizer = torch.optim.Adam(model.parameters())
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5,
            patience=5,
            verbose=True
        )
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc='Training') as pbar:
            for batch_X, batch_y in pbar:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                # 计算多个损失
                mse_loss = self.mse_criterion(outputs, batch_y)
                rmse_loss = self.rmse_criterion(outputs, batch_y)
                mape_loss = self.mape_criterion(outputs, batch_y)
                
                # 综合损失
                loss = mse_loss + 0.5 * rmse_loss + 0.3 * mape_loss
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = self.mse_criterion(outputs, batch_y)
                
                total_loss += loss.item()
                predictions.append(outputs.cpu().numpy())
                actuals.append(batch_y.cpu().numpy())
                
        predictions = np.concatenate(predictions)
        actuals = np.concatenate(actuals)
        
        # 计算评估指标
        metrics = calculate_metrics(actuals, predictions)
        
        return total_loss / len(val_loader), metrics
    
    def train(self, train_loader, val_loader, epochs=100, patience=10):
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = []
        
        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss, metrics = self.validate(val_loader)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 记录历史
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                **metrics
            })
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Training Loss: {train_loss:.4f}')
            print(f'Validation Loss: {val_loss:.4f}')
            print('Metrics:', metrics)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'metrics': metrics
                }, 'best_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print('Early stopping triggered')
                break
                
        return training_history
    
    def predict(self, test_loader):
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch_X, _ in test_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                predictions.append(outputs.cpu().numpy())
                
        return np.concatenate(predictions, axis=0) 