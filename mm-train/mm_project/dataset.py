# mm_project/dataset.py

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import List, Dict

class MultimodalDataset(Dataset):
    """Clase de Dataset para datos multimodales (texto + tabular)."""
    def __init__(self, texts: List[str], tabular_data: np.ndarray, labels: np.ndarray, tokenizer: PreTrainedTokenizer, max_len: int):
        self.texts = texts
        self.tabular_data = tabular_data
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True, max_length=self.max_len, return_token_type_ids=False,
            padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'tabular_data': torch.tensor(self.tabular_data[idx], dtype= torch.float),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }