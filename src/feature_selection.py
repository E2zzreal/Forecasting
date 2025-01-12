import numpy as np
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.ensemble import RandomForestRegressor

class FeatureSelector:
    """特征选择器"""
    
    @staticmethod
    def mutual_info_selection(X, y, k=10):
        """基于互信息的特征选择"""
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
        selector.fit(X, y)
        
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_
        })
        return feature_scores.sort_values('score', ascending=False)
    
    @staticmethod
    def importance_selection(X, y, n_estimators=100):
        """基于随机森林的特征重要性选择"""
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        rf.fit(X, y)
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        })
        return feature_importance.sort_values('importance', ascending=False) 