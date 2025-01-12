import numpy as np
import torch

class TimeSeriesAugmentation:
    @staticmethod
    def add_gaussian_noise(x, noise_factor=0.05):
        """添加高斯噪声"""
        noise = np.random.normal(loc=0, scale=noise_factor, size=x.shape)
        return x + noise
    
    @staticmethod
    def time_warping(x, sigma=0.2):
        """时间扭曲"""
        timestamps = np.arange(x.shape[1])
        warped_timestamps = timestamps + np.random.normal(0, sigma, size=timestamps.shape)
        warped_timestamps = np.sort(warped_timestamps)
        return np.interp(timestamps, warped_timestamps, x)
    
    @staticmethod
    def scaling(x, sigma=0.1):
        """幅值缩放"""
        scaling_factor = np.random.normal(1.0, sigma, size=(x.shape[0], 1, x.shape[2]))
        return x * scaling_factor
    
    @staticmethod
    def rotation(x, max_rotation=0.1):
        """旋转变换"""
        rotation_angle = np.random.uniform(-max_rotation, max_rotation)
        rotation_matrix = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)]
        ])
        return np.dot(x, rotation_matrix)

def augment_batch(batch, augmentation_types=['noise', 'scaling']):
    """对批次数据进行增强"""
    augmenter = TimeSeriesAugmentation()
    augmented_batch = batch.copy()
    
    for aug_type in augmentation_types:
        if aug_type == 'noise':
            augmented_batch = augmenter.add_gaussian_noise(augmented_batch)
        elif aug_type == 'warping':
            augmented_batch = augmenter.time_warping(augmented_batch)
        elif aug_type == 'scaling':
            augmented_batch = augmenter.scaling(augmented_batch)
        elif aug_type == 'rotation':
            augmented_batch = augmenter.rotation(augmented_batch)
            
    return augmented_batch 