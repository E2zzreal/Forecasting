# 微电网负荷时间序列预测

本项目使用ConvBiGRUAttention模型进行微电网负荷时间序列预测，包含数据预处理、模型训练、验证和预测等功能。

## 项目结构

```
.
├── README.md
├── requirements.txt
├── pjstructure.md
├── .gitignore
├── notebook/                # Jupyter notebook文件
│   └── ConVBiGRUAtteintion_multisteps.ipynb
├── src/                     # 源代码
│   ├── config.py            # 配置文件
│   ├── train.py             # 训练脚本
│   ├── models/              # 模型定义
│   │   └── conv_bigru_attention.py
│   └── utils/               # 工具函数
│       └── data_utils.py
├── data/                    # 数据文件
├── logs/                    # 训练日志
├── models/                  # 保存的模型
└── results/                 # 预测结果
```

## 快速开始

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 训练模型：
```bash
python src/train.py
```

3. 使用Jupyter Notebook进行实验：
```bash
jupyter notebook
```

## 主要功能

- 数据生成与预处理
- ConvBiGRUAttention模型训练
- 模型验证与保存
- 多步预测
- 注意力机制可视化

## 依赖

- Python 3.8+
- PyTorch 1.12+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Tensorboard
