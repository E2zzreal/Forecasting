from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np

class TimeSeriesDecomposer:
    """时间序列分解器"""
    
    @staticmethod
    def decompose(data, period=96, model='additive'):
        """
        将时间序列分解为趋势、季节性和残差
        
        Args:
            data: 时间序列数据
            period: 季节性周期
            model: 分解模型类型 ('additive' 或 'multiplicative')
        """
        decomposition = seasonal_decompose(
            data,
            period=period,
            model=model,
            extrapolate_trend='freq'
        )
        
        return {
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid
        }
    
    @staticmethod
    def extract_multi_period(data, periods=[24, 96, 672]):
        """提取多个周期的季节性模式"""
        seasonal_patterns = {}
        
        for period in periods:
            try:
                decomp = seasonal_decompose(
                    data,
                    period=period,
                    model='additive',
                    extrapolate_trend='freq'
                )
                seasonal_patterns[f'seasonal_{period}'] = decomp.seasonal
            except:
                continue
                
        return seasonal_patterns 