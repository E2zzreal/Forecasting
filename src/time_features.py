import pandas as pd
import numpy as np

class TimeFeatureExtractor:
    """时间特征提取器"""
    
    @staticmethod
    def get_time_features(datetime_index):
        """从时间索引中提取时间特征"""
        features = pd.DataFrame(index=datetime_index)
        
        # 基本时间特征
        features['hour'] = datetime_index.hour
        features['day'] = datetime_index.day
        features['month'] = datetime_index.month
        features['weekday'] = datetime_index.weekday
        features['quarter'] = datetime_index.quarter
        
        # 周期性特征
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        features['weekday_sin'] = np.sin(2 * np.pi * features['weekday'] / 7)
        features['weekday_cos'] = np.cos(2 * np.pi * features['weekday'] / 7)
        
        # 工作时间特征
        features['is_workday'] = (~features['weekday'].isin([5, 6])).astype(int)
        features['is_workhour'] = ((features['hour'] >= 9) & 
                                 (features['hour'] <= 17) & 
                                 features['is_workday']).astype(int)
        
        return features 