import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def plot_predictions(actual_values, predictions, n_steps=384):
    """
    绘制预测结果对比图
    
    Args:
        actual_values: 实际值
        predictions: 预测值
        n_steps: 显示的时间步数
    """
    plt.figure(figsize=(12, 6))
    plt.plot(actual_values.flatten()[:n_steps], label='Actual')
    plt.plot(predictions.flatten()[:n_steps], label='Predicted')
    plt.xlabel('Time Step')
    plt.ylabel('Load')
    plt.legend()
    plt.title('Actual vs. Predicted Load')
    plt.show()

def analyze_periodicity(time_series, sampling_rate=400):
    """
    使用FFT分析时间序列的周期性
    
    Args:
        time_series: 时间序列数据
        sampling_rate: 采样率
    """
    frequencies, power_spectrum = signal.periodogram(time_series, fs=sampling_rate)
    
    plt.figure(figsize=(12, 6))
    plt.plot(power_spectrum[:1000])
    plt.xlabel('Frequency/Hz')
    plt.ylabel('Magnitude')
    plt.title('Frequency Spectrum Analysis')
    plt.show()
    
    return frequencies, power_spectrum 