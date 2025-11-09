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
    {
        "display_name": "distilbert",
        "model_name": "distilbert/distilbert-base-uncased",
        "model_class": AutoModel,
        "tokenizer_class": AutoTokenizer,
    }
]

TEXT_SOURCES = {
    "Apertura": ["apertura"],
    "Interaccion": ["interaccion"]
}

FUSION_SCENARIOS = [
    {
        "display_name": "Asymm_Concat_K_4",
        "fusion_strategy": "concat",
        "tabular_net_layers": tabular_net_32_out,
        "select_k_best": 4
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
    "results_base_folder": "./results",
    "conjunto_name": "debug_smoke_test",
    "text_file_path": "./data/PPR_1_data_transcription.csv",
    "tab_file_path": "./data/TAB_0_data_text_audio.csv",
    "target_col": "FT__REAJUSTE",
    "split_col": "set",
    "num_classes": 2,
    "max_len": 64,
    "special_tokens": ["<captador>", "<cliente>", "<organizacion>", "<telefono>", "<correo>", "<rut>", "<monto>"],
    "seed": 42,
    "epochs": 1,
    "batch_size": 4,
    "learning_rate": 2e-5,
    "patience": 1,
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
                )
                experiments.append(config)
    
    if not found_model:
        raise ValueError(f"¡Modelo '{model_filter}' no encontrado en MODELS_TO_RUN!")
        
    print(f"✅ Lista de {len(experiments)} configuraciones generada para {model_filter}.")
    return experiments