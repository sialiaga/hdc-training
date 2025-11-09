from dataclasses import dataclass
from typing import List, Dict, Any, Type
from transformers import PreTrainedTokenizer, PreTrainedModel

@dataclass
class ExperimentSettings:
    """Configuración centralizada para un experimento multimodal."""
    
    # --- Requeridos ---
    experiment_name: str
    results_base_folder: str
    conjunto_name: str
    text_file_path: str
    tab_file_path: str
    text_input_cols: List[str]
    target_col: str
    split_col: str
    model_name: str
    tokenizer_class: Type[PreTrainedTokenizer]
    model_class: Type[PreTrainedModel]
    num_classes: int
    max_len: int
    special_tokens: List[str]
    tabular_net_layers: List[Dict]
    fusion_head_layers: List[Dict]
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    patience: int
    min_delta: float
    # --- Opcionales ---
    select_k_best: int = None
    fusion_strategy: str = 'concat'
    model_type: str = 'cls'
    use_lr_scheduler: bool = False
    warmup_ratio: float = 0.0
    use_differential_lr: bool = False
    text_model_lr: float = None
    tabular_lr: float = None
    fusion_lr: float = None

    def __post_init__(self):
        """Asigna LRs diferenciales si no se especifican."""
        if self.use_differential_lr:
            self.text_model_lr = self.text_model_lr if self.text_model_lr is not None else self.learning_rate
            self.tabular_lr = self.tabular_lr if self.tabular_lr is not None else self.learning_rate
            self.fusion_lr = self.fusion_lr if self.fusion_lr is not None else self.learning_rate