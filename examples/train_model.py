import torch
from src.data_processor import DataProcessor
from src.trainer import ModelTrainer
from models import BiGRUEncoderDecoder, TransformerEncoder, ConvBiGRU
from src.utils import plot_predictions

def main():
    # 初始化数据处理器
    data_processor = DataProcessor()
    
    # 创建虚拟数据
    data = data_processor.create_virtual_data()
    
    # 准备数据集
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_processor.prepare_data(data)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = data_processor.create_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    
    # 初始化模型(以BiGRU为例)
    model = BiGRUEncoderDecoder(
        input_size=6,
        hidden_size=64,
        num_layers=2,
        output_size=1
    )
    
    # 初始化训练器
    trainer = ModelTrainer(model)
    
    # 训练模型
    trainer.train(train_loader, val_loader, epochs=100)
    
    # 预测并可视化结果
    predictions = trainer.predict(test_loader)
    plot_predictions(y_test.numpy(), predictions)

if __name__ == '__main__':
    main() 