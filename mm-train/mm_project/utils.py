import random
import os
import time
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict

def set_seed(seed: int):
    """Fija la semilla para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"🌱 Semilla aleatoria fijada en: {seed}")

def create_network(layers_config: List[Dict], in_features: int) -> (nn.Sequential, int):
    """Crea una red nn.Sequential a partir de una lista de configuración."""
    layers = []
    current_in_features = in_features
    for layer_conf in layers_config:
        layer_type = layer_conf['type']
        if layer_type == 'linear':
            out_features = layer_conf['out_features']
            layers.append(nn.Linear(current_in_features, out_features))
            current_in_features = out_features
        elif layer_type == 'relu':
            layers.append(nn.ReLU())
        elif layer_type == 'dropout':
            layers.append(nn.Dropout(p=layer_conf['p']))
    return nn.Sequential(*layers), current_in_features

def format_time(seconds):
    """Convierte segundos en un formato legible H/M/S."""
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"