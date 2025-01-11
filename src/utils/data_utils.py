import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

def generate_synthetic_data():
    """生成模拟数据"""
    np.random.seed(42)
    time_index = pd.date_range(start='2024-01-01', end='2024-07-31 23:45', freq='15min')
    
    load_data = pd.DataFrame(index=time_index)
    load_data['Weekday'] = time_index.weekday
    load_data['IsWeekend'] = load_data['Weekday'].isin([5, 6]).astype(int)
    
    temperature = 20 + 10 * np.sin(2 * np.pi * (np.arange(len(time_index))-32) / 96) + np.random.rand(len(time_index)) * 5
    humidity = 50 + 20 * np.sin(2 * np.pi * np.arange(len(time_index)) / 96) + np.random.rand(len(time_index)) * 10
    wind_speed = 5 + 3 * np.random.rand(len(time_index))
    
    load_data['Temperature'] = temperature
    load_data['Humidity'] = humidity
    load_data['WindSpeed'] = wind_speed
    
    base_loads = 200 + np.random.uniform(0.9,1.1) * 50 * np.abs(load_data['Temperature']-20)/15
    work_hours_variations = np.array([max(0,np.sin(np.pi * (i % 96-32) / 32))*200 for i in range(len(time_index))])*np.random.uniform(0.9,1.1,len(time_index)) *(1+np.abs((load_data['Humidity']-50)/240))
    noise = np.random.rand(len(time_index)) * 50
    load_data['base_loads'] = base_loads + noise
    load_data["work_hours"] = work_hours_variations
    load_data["real_load"] = load_data["work_hours"] * (1-load_data["IsWeekend"])+ load_data['base_loads']
    
    return load_data

def load_real_data(filepath):
    """加载真实数据"""
    data = pd.read_csv(filepath, index_col='time', parse_dates=True)
    data.drop(index=data[data['values']>250].index, inplace=True)
    return data

def preprocess_data(data, input_cols=['values']):
    """数据预处理"""
    scaler = MinMaxScaler()
    data[input_cols] = scaler.fit_transform(data[input_cols])
    data.fillna(method='ffill', inplace=True)
    return data, scaler

def create_sequences(data, seq_length, target_length):
    """创建输入输出序列"""
    xs, ys = [], []
    for i in range(len(data) - seq_length - target_length):
        x = data[i:(i + seq_length)].values
        y = data['values'][(i + seq_length):(i + seq_length + target_length)].values
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def create_dataloaders(X, y, batch_size=128, train_ratio=0.8):
    """创建DataLoader"""
    train_size = int(train_ratio * len(X))
    train_dataset = TensorDataset(torch.FloatTensor(X[:train_size]), torch.FloatTensor(y[:train_size]))
    val_dataset = TensorDataset(torch.FloatTensor(X[train_size:]), torch.FloatTensor(y[train_size:]))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader
