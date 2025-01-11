import os

class Config:
    def __init__(self):
        # 数据配置
        self.data_dir = './data'
        self.raw_data_path = os.path.join(self.data_dir, 'raw_data.csv')
        self.processed_data_path = os.path.join(self.data_dir, 'processed_data.pkl')
        
        # 模型配置
        self.num_features = 4
        self.hidden_dims = [32, 64]
        self.num_hiddens = 128
        self.num_layers = 2
        self.output_dim = 1
        self.seq_length = 96
        self.target_length = 24
        self.batch_size = 128
        self.dropout = 0.1
        
        # 训练配置
        self.epochs = 100
        self.lr = 0.001
        self.patience = 10
        self.min_delta = 0.001
        
        # 路径配置
        self.log_dir = './logs'
        self.model_dir = './models'
        self.result_dir = './results'
        
        # 创建必要目录
        self._create_dirs()
    
    def _create_dirs(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

config = Config()
