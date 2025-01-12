import torch
from src.data_processor import DataProcessor
from src.trainer import ModelTrainer
from models.predictor import CompletePredictor
from src.utils import plot_predictions
from src.data_augmentation import augment_batch

def main():
    # 初始化数据处理器
    data_processor = DataProcessor()
    
    # 创建/加载数据
    data = data_processor.create_virtual_data()
    
    # 准备数据集
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_processor.prepare_data(data)
    
    # 数据增强
    X_train_aug = augment_batch(X_train.numpy())
    X_train = torch.cat([X_train, torch.FloatTensor(X_train_aug)], dim=0)
    y_train = torch.cat([y_train, y_train], dim=0)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = data_processor.create_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    
    # 初始化完整预测模型
    model = CompletePredictor(
        input_size=6,
        hidden_size=64,
        num_layers=2
    )
    
    # 初始化训练器
    trainer = ModelTrainer(model)
    
    # 训练模型
    history = trainer.train(train_loader, val_loader, epochs=100)
    
    # 预测并可视化结果
    predictions = trainer.predict(test_loader)
    plot_predictions(y_test.numpy(), predictions)

if __name__ == '__main__':
    main() 