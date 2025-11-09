# hpc_config_factory.py

# Importa la config desde el nuevo paquete
from mm_project.config import ExperimentSettings 
# Importa los modelos/tokenizers desde transformers directamente
from transformers import (
    BertTokenizer, BertModel, XLMRobertaTokenizer, XLMRobertaModel,
    RobertaModel, AutoModel, AutoTokenizer
)

# --- 1. DEFINICIÓN DE ARQUITECTURAS DE RED ---
tabular_net_768_out = [
    {'type': 'linear', 'out_features': 256}, {'type': 'relu'}, {'type': 'dropout', 'p': 0.3},
    {'type': 'linear', 'out_features': 512}, {'type': 'relu'}, {'type': 'dropout', 'p': 0.3},
    {'type': 'linear', 'out_features': 768}, {'type': 'relu'}
]
tabular_net_32_out = [
    {'type': 'linear', 'out_features': 64}, {'type': 'relu'},
    {'type': 'dropout', 'p': 0.25},
    {'type': 'linear', 'out_features': 32}, {'type': 'relu'}
]
fusion_head_layers_small = [
    {'type': 'linear', 'out_features': 128}, {'type': 'relu'},
    {'type': 'dropout', 'p': 0.35}
]


# --- 2. DEFINICIÓN DE EJES EXPERIMENTALES ---
MODELS_TO_RUN = [
    # {
    #     "display_name": "BETO",
    #     "model_name": "dccuchile/bert-base-spanish-wwm-uncased",
    #     "model_class": BertModel,
    #     "tokenizer_class": BertTokenizer,
    # },
    # {
    #     "display_name": "XLM-R",
    #     "model_name": "FacebookAI/xlm-roberta-base",
    #     "model_class": XLMRobertaModel,
    #     "tokenizer_class": XLMRobertaTokenizer,
    # },
    # {
    #     "display_name": "Tulio",
    #     "model_name": "dccuchile/tulio-chilean-spanish-bert",
    #     "model_class": BertModel,
    #     "tokenizer_class": BertTokenizer,
    # },
    {
        "display_name": "BERT_Multilingual",
        "model_name": "google-bert/bert-base-multilingual-uncased",
        "model_class": BertModel,
        "tokenizer_class": BertTokenizer,
    },
    # {
    #     "display_name": "RoBERTa_base",
    #     "model_name": "FacebookAI/roberta-base",
    #     "model_class": RobertaModel,
    #     "tokenizer_class": AutoTokenizer,
    # },
    # {
    #     "display_name": "BERT_base_uncased",
    #     "model_name": "google-bert/bert-base-uncased",
    #     "model_class": BertModel,
    #     "tokenizer_class": BertTokenizer,
    # },
    # {
    #     "display_name": "ModernBERT_base",
    #     "model_name": "answerdotai/ModernBERT-base",
    #     "model_class": AutoModel,
    #     "tokenizer_class": AutoTokenizer,
    # },
    # {
    #     "display_name": "ModernBERT_multi",
    #     "model_name": "neavo/modern_bert_multilingual",
    #     "model_class": AutoModel,
    #     "tokenizer_class": AutoTokenizer,
    # }
]

TEXT_SOURCES = {
    "Apertura": ["apertura"],
    "Interaccion": ["interaccion"],
    "Apertura_Interaccion": ["apertura", "interaccion"]
}

FUSION_SCENARIOS = [
    # --- Grupo 1: Simétricos (7 escenarios) ---
    # 5 para Concat + K-Best
    {
        "display_name": "Symm_Concat_K_All",
        "fusion_strategy": "concat",
        "tabular_net_layers": tabular_net_768_out,
        "select_k_best": None
    },
    {
        "display_name": "Symm_Concat_K_64",
        "fusion_strategy": "concat",
        "tabular_net_layers": tabular_net_768_out,
        "select_k_best": 64
    },
    {
        "display_name": "Symm_Concat_K_32",
        "fusion_strategy": "concat",
        "tabular_net_layers": tabular_net_768_out,
        "select_k_best": 32
    },
    {
        "display_name": "Symm_Concat_K_16",
        "fusion_strategy": "concat",
        "tabular_net_layers": tabular_net_768_out,
        "select_k_best": 16
    },
    {
        "display_name": "Symm_Concat_K_8",
        "fusion_strategy": "concat",
        "tabular_net_layers": tabular_net_768_out,
        "select_k_best": 8
    },
    # 2 para Fusión Avanzada (solo con K_All)
    {
        "display_name": "Symm_CrossAttn_K_All",
        "fusion_strategy": "cross_attention",
        "tabular_net_layers": tabular_net_768_out,
        "select_k_best": None
    },
    {
        "display_name": "Symm_WeightedAdd_K_All",
        "fusion_strategy": "weighted_add",
        "tabular_net_layers": tabular_net_768_out,
        "select_k_best": None
    },

    # --- Grupo 2: Asimétricos (5 escenarios) ---
    # 5 para Concat + K-Best (con la red pequeña)
    {
        "display_name": "Asymm_Concat_K_All",
        "fusion_strategy": "concat",
        "tabular_net_layers": tabular_net_32_out,
        "select_k_best": None
    },
    {
        "display_name": "Asymm_Concat_K_64",
        "fusion_strategy": "concat",
        "tabular_net_layers": tabular_net_32_out,
        "select_k_best": 64
    },
    {
        "display_name": "Asymm_Concat_K_32",
        "fusion_strategy": "concat",
        "tabular_net_layers": tabular_net_32_out,
        "select_k_best": 32
    },
    {
        "display_name": "Asymm_Concat_K_16",
        "fusion_strategy": "concat",
        "tabular_net_layers": tabular_net_32_out,
        "select_k_best": 16
    },
    {
        "display_name": "Asymm_Concat_K_8",
        "fusion_strategy": "concat",
        "tabular_net_layers": tabular_net_32_out,
        "select_k_best": 8
    }
]


# --- 3. CONFIGURACIÓN BASE ---
BASE_SETTINGS = {
    # ⚠️ ¡Recuerda poner tus rutas ABSOLUTAS aquí! ⚠️
    "results_base_folder": "/home/tu_usuario/proyecto_tesis/results",
    "conjunto_name": "experiments_full_factorial_v1",
    "text_file_path": "/home/tu_usuario/proyecto_tesis/data/PPR_1_data_transcription.csv",
    "tab_file_path": "/home/tu_usuario/proyecto_tesis/data/TAB_0_data_text_audio.csv",
    
    # --- El resto de la config ---
    "target_col": "FT__REAJUSTE",
    "split_col": "set",
    # ... (el resto de tu config base) ...
    "patience": 3,
    "min_delta": 0.001,
}

# --- 4. FUNCIÓN GENERADORA ---
def get_experiments_for_model(model_filter: str):
    """
    Genera la lista de configs de experimentos
    SOLO para el modelo especificado.
    """
    print(f"Generando lista de configuraciones para el modelo: {model_filter}...")
    experiments = []
    
    found_model = False
    for model_config in MODELS_TO_RUN:
        
        # ❗ CAMBIO: Si el 'display_name' no coincide, salta este modelo
        if model_config['display_name'] != model_filter:
            continue
        
        found_model = True
        for text_name, text_cols in TEXT_SOURCES.items():
            for scenario in FUSION_SCENARIOS:
                
                exp_name = f"{model_config['display_name']}_{text_name}_{scenario['display_name']}"
                current_special_tokens = BASE_SETTINGS["special_tokens"]
                
                config = ExperimentSettings(
                    experiment_name=exp_name,
                    model_name=model_config['model_name'],
                    model_class=model_config['model_class'],
                    tokenizer_class=model_config['tokenizer_class'],
                    text_input_cols=text_cols,
                    fusion_strategy=scenario['fusion_strategy'],
                    tabular_net_layers=scenario['tabular_net_layers'],
                    select_k_best=scenario['select_k_best'],
                    fusion_head_layers=fusion_head_layers_small,
                    model_type='cls',
                    **BASE_SETTINGS,
                    special_tokens=current_special_tokens
                )
                experiments.append(config)
    
    if not found_model:
        raise ValueError(f"¡Modelo '{model_filter}' no encontrado en MODELS_TO_RUN!")
        
    print(f"✅ Lista de {len(experiments)} configuraciones generada para {model_filter}.")
    return experiments