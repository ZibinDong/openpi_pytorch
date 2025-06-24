from typing import Dict

import numpy as np
import torch


def dict_apply(func, d):
    """
    Apply a function to all values in a dictionary recursively.
    If the value is a dictionary, it will apply the function to its values.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            dict_apply(func, value)
        else:
            d[key] = func(value)
    return d


class Normalizer:
    def __init__(
        self,
        norm_stats: Dict[str, Dict[str, np.ndarray]],
        norm_type: Dict[str, str] | None = None,
    ):
        self.norm_stats = dict_apply(lambda x: x.astype(np.float32), norm_stats)
        self.norm_type = norm_type or {}

    def normalize(self, data: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        normalized_data = {}
        for key, value in data.items():
            if key in self.norm_stats:
                norm_type = self.norm_type.get(key, "identity")
                if norm_type == "meanstd":
                    mean = self.norm_stats[key]["mean"]
                    std = self.norm_stats[key]["std"]
                    normalized_value = (value - mean) / (std + 1e-6)
                elif norm_type == "std":
                    std = self.norm_stats[key]["std"]
                    normalized_value = value / (std + 1e-6)
                elif norm_type == "minmax":
                    min_val = self.norm_stats[key]["min"]
                    max_val = self.norm_stats[key]["max"]
                    normalized_value = (value - min_val) / (
                        max_val - min_val + 1e-6
                    ) * 2 - 1
                elif norm_type == "identity":
                    normalized_value = value
                else:
                    raise ValueError(
                        f"Unknown normalization type: {norm_type}. Supported types are 'meanstd', 'minmax', and 'identity'."
                    )
                normalized_data[key] = normalized_value
            else:
                # If the key is not in norm_stats, we assume no normalization is needed
                normalized_data[key] = value
        return normalized_data

    def unnormalize(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        unnormalized_data = {}
        for key, value in data.items():
            if key in self.norm_stats:
                norm_type = self.norm_type.get(key, "identity")
                if norm_type == "meanstd":
                    mean = self.norm_stats[key]["mean"]
                    std = self.norm_stats[key]["std"]
                    unnormalized_value = value * (std + 1e-6) + mean
                elif norm_type == "std":
                    std = self.norm_stats[key]["std"]
                    unnormalized_value = value * (std + 1e-6)
                elif norm_type == "minmax":
                    min_val = self.norm_stats[key]["min"]
                    max_val = self.norm_stats[key]["max"]
                    unnormalized_value = (value + 1) / 2 * (
                        max_val - min_val + 1e-6
                    ) + min_val
                elif norm_type == "identity":
                    unnormalized_value = value
                else:
                    raise ValueError(
                        f"Unknown normalization type: {norm_type}. Supported types are 'meanstd', 'minmax', and 'identity'."
                    )
                unnormalized_data[key] = unnormalized_value
            else:
                # If the key is not in norm_stats, we assume no unnormalization is needed
                unnormalized_data[key] = value
        return unnormalized_data
