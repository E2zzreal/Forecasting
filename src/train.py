import os
import time
import torch
import numpy as np
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from src.models.conv_bigru_attention import ConvBiGRUAttention, MAPELoss
from src.utils.data_utils import generate_synthetic_data, preprocess_data, create_sequences, create_dataloaders

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model()
        self.criterion = MAPELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.writer = SummaryWriter(log_dir=config['log_dir'])
        
    def _build_model(self):
        model = ConvBiGRUAttention(
            num_features=self.config['num_features'],
            hidden_dims=self.config['hidden_dims'],
            num_hiddens=self.config['num_hiddens'],
            num_layers=self.config['num_layers'],
            output_dim=self.config['output_dim'],
            seq_length=self.config['seq_length'],
            batch_size=self.config['batch_size']
        ).to(self.device)
        return model

    def train(self, train_loader, val_loader):
        best_val_loss = float('inf')
        for epoch in range(self.config['epochs']):
            self.model.train()
            train_loss = 0
            start_time = time.time()
            
            for batch_idx, (X, y) in enumerate(train_loader):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output, _ = self.model(X)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")
            
            avg_train_loss = train_loss / len(train_loader)
            val_loss = self.validate(val_loader)
            
            self.writer.add_scalar('Loss/train', avg_train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model()
            
            print(f"Epoch {epoch+1}/{self.config['epochs']} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Time: {time.time()-start_time:.2f}s")
            
    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)
                output, _ = self.model(X)
                val_loss += self.criterion(output, y).item()
        return val_loss / len(val_loader)

    def save_model(self):
        os.makedirs(self.config['model_dir'], exist_ok=True)
        torch.save(self.model.state_dict(), 
                  os.path.join(self.config['model_dir'], 'best_model.pth'))
        print(f"Model saved to {self.config['model_dir']}")

def main():
    config = {
        'num_features': 4,
        'hidden_dims': [32, 64],
        'num_hiddens': 128,
        'num_layers': 2,
        'output_dim': 1,
        'seq_length': 96,
        'target_length': 24,
        'batch_size': 128,
        'epochs': 100,
        'lr': 0.001,
        'log_dir': './logs',
        'model_dir': './models'
    }
    
    # 数据准备
    data = generate_synthetic_data()
    data, scaler = preprocess_data(data)
    X, y = create_sequences(data, config['seq_length'], config['target_length'])
    train_loader, val_loader = create_dataloaders(X, y, batch_size=config['batch_size'])
    
    # 训练
    trainer = Trainer(config)
    trainer.train(train_loader, val_loader)

if __name__ == '__main__':
    main()
