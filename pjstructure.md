# 项目结构说明

## 1. 目录结构

```
.
├── README.md                # 项目说明文档
├── requirements.txt         # 项目依赖
├── pjstructure.md           # 项目结构说明
├── .gitignore               # Git忽略文件配置
├── notebook/                # Jupyter notebook文件
│   └── ConVBiGRUAtteintion_multisteps.ipynb
├── src/                     # 源代码目录
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

## 2. 主要模块说明

### 2.1 数据模块
- `data_utils.py`: 数据预处理工具
  - 数据生成
  - 数据标准化
  - 序列生成
  - 数据加载器

### 2.2 模型模块
- `conv_bigru_attention.py`: ConvBiGRUAttention模型
  - 卷积层
  - BiGRU层
  - 注意力机制
  - 全连接层

### 2.3 训练模块
- `train.py`: 训练脚本
  - 模型训练
  - 模型验证
  - 模型保存
  - 日志记录

### 2.4 配置模块
- `config.py`: 配置文件
  - 数据路径配置
  - 模型参数配置
  - 训练参数配置
  - 路径管理
