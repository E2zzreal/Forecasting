import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

class DataProcessor:
    def __init__(self, sequence_length=96, prediction_length=96):
        """
        初始化数据处理器
        
        Args:
            sequence_length: 输入序列长度
            prediction_length: 预测序列长度
        """
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.scaler = MinMaxScaler()
        
    def create_virtual_data(self, start_date='2024-01-01', end_date='2024-07-31'):
        """
        创建虚拟数据集
        """
        # 创建时间索引
        time_index = pd.date_range(start=start_date, end=end_date, freq='15min')
        
        # 创建DataFrame
        load_data = pd.DataFrame(index=time_index)
        
        # 添加日历数据
        load_data['Weekday'] = time_index.weekday
        load_data['IsWeekend'] = load_data['Weekday'].isin([5, 6]).astype(int)
        
        # 添加天气数据
        temperature = 20 + 10 * np.sin(2 * np.pi * (np.arange(len(time_index))-32) / 96) + np.random.rand(len(time_index)) * 5
        humidity = 50 + 20 * np.sin(2 * np.pi * np.arange(len(time_index)) / 96) + np.random.rand(len(time_index)) * 10
        wind_speed = 5 + 3 * np.random.rand(len(time_index))
        
        load_data['Temperature'] = temperature
        load_data['Humidity'] = humidity
        load_data['WindSpeed'] = wind_speed
        
        # 生成负荷数据
        base_loads = 200 + np.random.uniform(0.9,1.1) * 50 * np.abs(load_data['Temperature']-20)/15
        work_hours = np.zeros(len(time_index))
        
        for i in range(len(time_index)):
            hour = time_index[i].hour
            if 9 <= hour <= 17 and load_data['IsWeekend'][i] == 0:
                work_hours[i] = 1
                
        load_data['base_loads'] = base_loads
        load_data['work_hours'] = work_hours
        load_data['real_load'] = base_loads * (1 + 0.3 * work_hours)
        
        return load_data
    
    def prepare_data(self, data, train_ratio=0.7, val_ratio=0.2):
        """
        准备训练、验证和测试数据
        """
        # 标准化数据
        features = ['Weekday', 'IsWeekend', 'Temperature', 'Humidity', 'WindSpeed', 'work_hours']
        target = ['real_load']
        
        X = data[features].values
        y = data[target].values
        
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.scaler.fit_transform(y)
        
        # 创建序列数据
        X_sequences = []
        y_sequences = []
        
        for i in range(len(X_scaled) - self.sequence_length - self.prediction_length + 1):
            X_sequences.append(X_scaled[i:(i + self.sequence_length)])
            y_sequences.append(y_scaled[i + self.sequence_length:i + self.sequence_length + self.prediction_length])
            
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        # 划分数据集
        train_size = int(len(X_sequences) * train_ratio)
        val_size = int(len(X_sequences) * val_ratio)
        
        X_train = X_sequences[:train_size]
        y_train = y_sequences[:train_size]
        
        X_val = X_sequences[train_size:train_size+val_size]
        y_val = y_sequences[train_size:train_size+val_size]
        
        X_test = X_sequences[train_size+val_size:]
        y_test = y_sequences[train_size+val_size:]
        
        # 转换为PyTorch张量
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test)
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def create_data_loaders(self, X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
        """
        创建数据加载器
        """
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return train_loader, val_loader, test_loader 