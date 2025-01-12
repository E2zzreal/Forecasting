# 电网负荷时间序列预测项目

## 项目简介

本项目旨在构建一个基于深度学习的电网负荷时间序列预测系统，采用卷积GRU网络结合注意力机制（ConVBiGRUAtteintion）作为核心模型，实现多步长负荷预测。

## 主要特性

- 支持多种时间序列预测模型：GRU、Transformer、扩散模型等
- 提供完整的数据预处理流程
- 包含多种特征工程方法
- 支持模型集成和对比实验
- 提供可视化工具和评估指标

## 安装说明

1. 克隆项目仓库：
   ```bash
   git clone https://github.com/your-repo/forecasting-new.git
   cd forecasting-new
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. 准备数据：
   - 将负荷数据文件放置在`data/`目录下
   - 支持CSV格式，需包含时间戳和负荷值两列

## 快速开始

```python
from src.data_processor import DataProcessor
from src.trainer import Trainer
from models.conv_gru import ConvGRUModel

# 数据预处理
processor = DataProcessor('data/load_data.csv')
train_data, test_data = processor.process()

# 模型训练
model = ConvGRUModel(input_dim=1, hidden_dim=64, output_dim=24)
trainer = Trainer(model)
trainer.train(train_data, epochs=100)

# 模型预测
predictions = trainer.predict(test_data)
```

## 项目结构

```
.
├── data/                # 数据目录
├── examples/            # 示例代码
├── models/              # 模型实现
│   ├── attention.py     # 注意力机制
│   ├── conv_gru.py      # 卷积GRU
│   ├── diffusion.py     # 扩散模型
│   ├── ensemble.py      # 集成学习
│   └── transformer.py   # Transformer
├── notebooks/           # Jupyter Notebook
├── src/                 # 核心代码
│   ├── data_processor.py # 数据预处理
│   ├── feature_selection.py # 特征选择
│   ├── metrics.py       # 评估指标
│   ├── time_features.py # 时间特征
│   ├── trainer.py       # 模型训练
│   └── utils.py         # 工具函数
├── requirements.txt     # 依赖库
└── README.md            # 项目说明
```

## 模块说明

### 数据处理模块
- `data_processor.py`: 数据加载、清洗、标准化
- `feature_selection.py`: 特征选择与工程
- `time_features.py`: 时间特征提取

### 模型模块
- `conv_gru.py`: 卷积GRU网络
- `attention.py`: 注意力机制
- `transformer.py`: Transformer模型
- `diffusion.py`: 扩散模型
- `ensemble.py`: 模型集成

### 训练评估
- `trainer.py`: 模型训练与验证
- `metrics.py`: 评估指标计算

## 贡献指南

欢迎提交issue和pull request，贡献代码请遵循以下规范：
1. 新功能开发请创建feature分支
2. Bug修复请创建fix分支
3. 提交代码前请运行单元测试
4. 更新README.md中的相关说明

## 许可证

本项目采用MIT许可证，详情请见LICENSE文件。
