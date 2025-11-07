import numpy as np
from typing import Tuple, Union, Callable

def BrightnessMultiplicativeTransform_self(data_ori: np.ndarray, p_per_sample=0.5, multiplier_range=(0.7, 1.3)) -> np.ndarray:
    """_summary_

    Args:
        data (np.ndarray): (D, H, W), don't has channel and batch dim.
        multiplier_range (tuple, optional): _description_. Defaults to (0.7, 1.3).
        p_per_sample (float, optional): _description_. Defaults to 0.3.
    """
    data = data_ori.copy()
    d, h, w = data.shape
    for b in range(0, d):
        if np.random.uniform() < p_per_sample:
            for c in range(0, h):
                multiplier = np.random.uniform(multiplier_range[0], multiplier_range[1])
                data[b][c] *= multiplier
    return data

def ContrastAugmentationTransform_self(data_ori: np.ndarray, p_per_sample: float = 0.5, 
                                       contrast_range: Union[Tuple[float, float], Callable[[], float]] = (0.75, 1.25),) -> np.ndarray:
    data = data_ori.copy()
    d, h, w = data.shape
    for b in range(0, d):
        for c in range(0, h):
            if np.random.uniform() < p_per_sample:
                if callable(contrast_range):
                    factor = contrast_range()
                elif np.random.random() < 0.5 and contrast_range[0] < 1:
                    factor = np.random.uniform(contrast_range[0], 1)
                else:
                    factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])
                mn = data[b][c].mean()
                minm = data[b][c].min()
                maxm = data[b][c].max()

                data[b][c] = (data[b][c] - mn) * factor + mn
                data[b][c][data[b][c] < minm] = minm
                data[b][c][data[b][c] > maxm] = maxm
    return data

def GammaTransform_self(data_ori: np.ndarray, p_per_sample: float = 0.5,
                        gamma_range=(0.5, 2),
                        retain_stats: Union[bool, Callable[[], bool]] = False, epsilon=1e-15, invert_image=False
                        ) -> np.ndarray:
    data = data_ori.copy()
    if invert_image:
        data = - data

    d, h, w = data.shape
    for b in range(0, d):
        for c in range(0, h):
            if np.random.uniform() < p_per_sample:
                retain_stats_here = retain_stats() if callable(retain_stats) else retain_stats
                if retain_stats_here:
                    mn = data[b][c].mean()
                    sd = data[b][c].std()
                if np.random.random() < 0.5 and gamma_range[0] < 1:
                    gamma = np.random.uniform(gamma_range[0], 1)
                else:
                    gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
                minm = data[b][c].min()
                rnge = data[b][c].max() - minm
                data[b][c] = np.power(((data[b][c] - minm) / float(rnge + epsilon)), gamma) * float(rnge + epsilon) + minm
                if retain_stats_here:
                    data[b][c] = data[b][c] - data[b][c].mean()
                    data[b][c] = data[b][c] / (data[b][c].std() + 1e-8) * sd
                    data[b][c] = data[b][c] + mn
    
    if invert_image:
        data = - data
    return data


def main():
    print(f"this is augmentation file.")

if __name__ == '__main__':
    main()