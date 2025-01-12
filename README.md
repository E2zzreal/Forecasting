## 模块调用关系

项目采用模块化设计，主要模块调用关系如下：

```
src/
├── data_augmentation.py  # 数据增强
├── data_processor.py     # 数据预处理
├── decomposition.py      # 时间序列分解
├── feature_selection.py  # 特征选择
├── metrics.py            # 评估指标
├── time_features.py      # 时间特征提取
├── trainer.py            # 模型训练
├── utils.py              # 工具函数
└── __init__.py           # 模块初始化

models/
├── attention.py          # 注意力机制
├── conv_gru.py           # 卷积GRU
├── diffusion.py          # 扩散模型
├── ensemble.py           # 集成学习
└── transformer.py        # Transformer模型
```

各模块通过src/__init__.py统一导入，使用时可直接调用：
```python
from src import data_processor, trainer
from src.models import conv_gru, attention
```
