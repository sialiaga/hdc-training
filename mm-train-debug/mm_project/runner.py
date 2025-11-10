import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, PreTrainedModel
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
from tqdm import tqdm
from dataclasses import asdict

# Importaciones internas del paquete
from .config import ExperimentSettings
from .dataset import MultimodalDataset
from .fusion_strategy import FusionModel
from .utils import set_seed, create_network, format_time

class ExperimentRunner:
    """Ejecuta un ciclo de experimentación completo."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"ExperimentRunner inicializado. Usando dispositivo: {self.device}")
        if not torch.cuda.is_available():
            print("⚠️ ADVERTENCIA: CUDA no está disponible. El entrenamiento será en CPU.")

    def _load_and_preprocess_data(self, config: ExperimentSettings, tokenizer):
        print(f"Cargando datos: {config.text_file_path} y {config.tab_file_path}")
        text_data_path = config.text_file_path
        tab_data_path = config.tab_file_path
        
        if not os.path.exists(text_data_path) or not os.path.exists(tab_data_path):
            raise FileNotFoundError(f"¡Error! No se encontraron los archivos de datos en las rutas HPC: {text_data_path} o {tab_data_path}")

        df_text = pd.read_csv(text_data_path)
        
        df_text['TEXTO_COMBINADO'] = ''
        for col in config.text_input_cols:
            df_text[col] = df_text[col].astype(str).replace('nan', '')
            df_text['TEXTO_COMBINADO'] += (df_text[col] + ' ')
        df_text['TEXTO_COMBINADO'] = df_text['TEXTO_COMBINADO'].str.strip()
        df_text = df_text.drop(columns=['APORTE_ORIGINAL', 'APORTE_REAJUSTADO'], errors='ignore')

        df_tab = pd.read_csv(tab_data_path, sep=";", index_col=0)
        df_tab = df_tab.drop(columns=[config.target_col], errors='ignore')
        
        df_tab_unique = df_tab.drop_duplicates(subset=['ID_LLAMADA'], keep='first')
        df = pd.merge(df_text, df_tab_unique, on='ID_LLAMADA', how='inner')
        all_tabular_features = [col for col in df_tab.columns if col != 'ID_LLAMADA']
        
        df_train = df[df[config.split_col] == 'TRAIN'].copy().head(1000)
        df_val = df[df[config.split_col] == 'VAL'].copy().head(250)
        df_test = df[df[config.split_col] == 'TEST'].copy().head(250)
        
        y_train = df_train[config.target_col].values
        y_val = df_val[config.target_col].values
        y_test = df_test[config.target_col].values
        
        X_text_train = df_train['TEXTO_COMBINADO'].to_list()
        X_text_val = df_val['TEXTO_COMBINADO'].to_list()
        X_text_test = df_test['TEXTO_COMBINADO'].to_list()
        
        X_tabular_train = df_train[all_tabular_features].values.astype(np.float32)
        X_tabular_val = df_val[all_tabular_features].values.astype(np.float32)
        X_tabular_test = df_test[all_tabular_features].values.astype(np.float32)
        
        if config.select_k_best is not None and config.select_k_best > 0 and config.select_k_best < len(all_tabular_features):
            print(f"Aplicando SelectKBest, k={config.select_k_best}...")
            X_tabular_train_safe = np.nan_to_num(X_tabular_train)
            X_tabular_val_safe = np.nan_to_num(X_tabular_val)
            X_tabular_test_safe = np.nan_to_num(X_tabular_test)
            
            selector = SelectKBest(f_classif, k=config.select_k_best)
            X_tabular_train_kbest = selector.fit_transform(X_tabular_train_safe, y_train)
            X_tabular_val_kbest = selector.transform(X_tabular_val_safe)
            X_tabular_test_kbest = selector.transform(X_tabular_test_safe)
            
            selected_features = np.array(all_tabular_features)[selector.get_support()]
            print(f"Features seleccionadas: {selected_features.tolist()}")
            
            X_tabular_train = X_tabular_train_kbest
            X_tabular_val = X_tabular_val_kbest
            X_tabular_test = X_tabular_test_kbest
            num_tabular_features = config.select_k_best
        else:
            print(f"Usando todas las {len(all_tabular_features)} features tabulares.")
            num_tabular_features = len(all_tabular_features)
            
        print(f"Datos cargados: {len(df_train)} train, {len(df_val)} val, {len(df_test)} test.")
        print(f"Número de features tabulares: {num_tabular_features}")
        
        train_dataset = MultimodalDataset(X_text_train, X_tabular_train, y_train, tokenizer, config.max_len)
        val_dataset = MultimodalDataset(X_text_val, X_tabular_val, y_val, tokenizer, config.max_len)
        test_dataset = MultimodalDataset(X_text_test, X_tabular_test, y_test, tokenizer, config.max_len)
        print("✅ Datasets de PyTorch (Train/Val/Test) creados.")

        return train_dataset, val_dataset, test_dataset, num_tabular_features

    def _build_model(self, config: ExperimentSettings, num_tabular_features: int, base_text_model: PreTrainedModel):
        print("Construyendo modelo...")
        text_hidden_size = base_text_model.config.hidden_size
        tabular_net, tab_out = create_network(config.tabular_net_layers, num_tabular_features)

        if config.fusion_strategy == 'concat':
            fusion_input_size = text_hidden_size + tab_out
        elif config.fusion_strategy in ['add', 'multiply', 'weighted_add']:
            fusion_input_size = text_hidden_size
            if tab_out != text_hidden_size:
                raise ValueError(f"Fusión '{config.fusion_strategy}' requiere tab_out ({tab_out}) == text_hidden_size ({text_hidden_size}).")
        elif config.fusion_strategy == 'cross_attention':
            fusion_input_size = text_hidden_size * 2
            if tab_out != text_hidden_size:
                raise ValueError(f"Fusión '{config.fusion_strategy}' requiere tab_out ({tab_out}) == text_hidden_size ({text_hidden_size}).")
        elif config.fusion_strategy == 'hyper':
             if tab_out != text_hidden_size:
                 raise ValueError(f"Fusión 'hyper' requiere tab_out ({tab_out}) == text_hidden_size ({text_hidden_size}).")
             fusion_input_size = (text_hidden_size + tab_out) + text_hidden_size + text_hidden_size
        else:
            raise ValueError(f"Estrategia de fusión desconocida: {config.fusion_strategy}")

        fusion_head, fusion_out = create_network(config.fusion_head_layers, fusion_input_size)
        
        model = FusionModel(
            text_model=base_text_model, 
            num_classes=config.num_classes,
            tabular_net=tabular_net,
            tabular_out_features=tab_out,
            fusion_head=fusion_head,
            fusion_head_out_features=fusion_out,
            fusion_strategy=config.fusion_strategy,
            model_type=config.model_type
        )
        return model.to(self.device)

    def _train_epoch(self, model, loader, loss_fn, optimizer, device, scheduler):
        model.train()
        total_train_loss = 0
        for batch in tqdm(loader, desc="[Train]"):
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                tabular_data=batch['tabular_data'].to(device)
            )
            labels = batch['label'].to(device)
            loss = loss_fn(outputs, labels)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
        return total_train_loss / len(loader)

    def _val_epoch(self, model, loader, loss_fn, device):
        model.eval()
        total_val_loss, val_probs, val_true, val_preds = 0, [], [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc="[Eval]"):
                outputs = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    tabular_data=batch['tabular_data'].to(device)
                )
                labels = batch['label'].to(device)
                loss = loss_fn(outputs, labels)
                total_val_loss += loss.item()
                probabilities = F.softmax(outputs, dim=1)
                preds = torch.argmax(probabilities, dim=1)

                val_probs.extend(probabilities[:, 1].cpu().numpy())
                val_true.extend(labels.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(loader)
        val_auc = roc_auc_score(val_true, val_probs)
        return avg_val_loss, val_auc, val_true, val_preds

    def _final_test_evaluation(self, model: nn.Module, test_loader: DataLoader, experiment_path: str):
        print("\n🧪 Realizando evaluación final sobre el conjunto de TEST...")
        model.eval()
        test_probs, test_true, test_preds = [], [], []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="[Final Test Evaluation]"):
                outputs = model(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                    tabular_data=batch['tabular_data'].to(self.device)
                )
                labels = batch['label'].to(self.device)
                probabilities = F.softmax(outputs, dim=1)
                preds = torch.argmax(probabilities, dim=1)
                test_probs.extend(probabilities[:, 1].cpu().numpy())
                test_true.extend(labels.cpu().numpy())
                test_preds.extend(preds.cpu().numpy())

        test_auc = roc_auc_score(test_true, test_probs)
        report = classification_report(test_true, test_preds, output_dict=True, zero_division=0)
        print(f"Resultados en TEST -> AUC: {test_auc:.4f}, Accuracy: {report['accuracy']:.4f}")

        test_results_path = os.path.join(experiment_path, "test_evaluation")
        os.makedirs(test_results_path, exist_ok=True)
        with open(os.path.join(test_results_path, 'test_classification_report.json'), 'w') as f:
            json.dump(report, f, indent=4)
        with open(os.path.join(test_results_path, 'test_metrics.json'), 'w') as f:
            json.dump({'test_auc': test_auc, 'test_accuracy': report['accuracy']}, f, indent=4)
        print(f"Resultados de TEST guardados en: {test_results_path}")

    def run(self, config: ExperimentSettings, num_workers: int = 0):
        """Orquesta la ejecución completa de un experimento."""
        print(f"\n{'='*20} INICIANDO EXPERIMENTO: {config.experiment_name} {'='*20}")
        start_time = time.time()
        set_seed(config.seed)
        
        conjunto_path = os.path.join(config.results_base_folder, config.conjunto_name)
        experiment_path = os.path.join(conjunto_path, config.experiment_name)
        os.makedirs(experiment_path, exist_ok=True)

        config_dict = asdict(config)
        config_dict['tokenizer_class'] = str(config_dict['tokenizer_class'])
        config_dict['model_class'] = str(config_dict['model_class'])

        with open(os.path.join(experiment_path, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=4)

        print(f"Cargando tokenizer y modelo base: {config.model_name}")
        tokenizer = config.tokenizer_class.from_pretrained(config.model_name)
        base_text_model = config.model_class.from_pretrained(config.model_name)

        if config.special_tokens:
            print(f"Añadiendo {len(config.special_tokens)} tokens especiales.")
            tokenizer.add_special_tokens({'additional_special_tokens': config.special_tokens})
            base_text_model.resize_token_embeddings(len(tokenizer))

        train_dataset, val_dataset, test_dataset, num_tabular_features = self._load_and_preprocess_data(config, tokenizer)

        print(f"Creando DataLoaders con {num_workers} workers.")
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True 
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config.batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )

        model = self._build_model(config, num_tabular_features, base_text_model)

        if config.use_differential_lr:
            print("Usando learning rates diferenciales.")
            optimizer = torch.optim.AdamW([
                {'params': model.text_model.parameters(), 'lr': config.text_model_lr},
                {'params': model.tabular_net.parameters(), 'lr': config.tabular_lr},
                {'params': model.fusion_head.parameters(), 'lr': config.fusion_lr}
            ])
        else:
            print(f"Usando learning rate único: {config.learning_rate}")
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

        loss_fn = nn.CrossEntropyLoss()
        scheduler = None
        if config.use_lr_scheduler:
            total_steps = len(train_loader) * config.epochs
            warmup_steps = int(total_steps * config.warmup_ratio)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
            print("Usando scheduler lineal con warmup.")

        epoch_history = []
        best_val_auc = -1.0
        best_model_state = None
        epochs_no_improve = 0

        for epoch in range(config.epochs):
            print(f"\n--- Epoch {epoch + 1}/{config.epochs} ---")
            
            avg_train_loss = self._train_epoch(model, train_loader, loss_fn, optimizer, self.device, scheduler)
            avg_val_loss, val_auc, val_true, val_preds = self._val_epoch(model, val_loader, loss_fn, self.device)
            
            print(f"Epoch {epoch + 1} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val AUC: {val_auc:.4f}")

            epoch_results = {'epoch': epoch + 1, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss, 'val_auc': val_auc}
            epoch_history.append({**epoch_results, 'experiment_name': config.experiment_name})

            if val_auc > best_val_auc + config.min_delta:
                best_val_auc = val_auc
                epochs_no_improve = 0
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                print(f"✨ Nuevo mejor AUC de validación: {best_val_auc:.4f}. Estado del modelo guardado.")

                report = classification_report(val_true, val_preds, output_dict=True, zero_division=0)
                best_metrics = {
                    'best_epoch': epoch + 1, 'validation_auc': best_val_auc,
                    'val_accuracy': report['accuracy'], 'classification_report': report
                }
                metrics_path = os.path.join(experiment_path, 'best_epoch_metrics.json')
                with open(metrics_path, 'w') as f:
                    json.dump(best_metrics, f, indent=4)

            else:
                epochs_no_improve += 1
                print(f"😕 No hubo mejora significativa. Paciencia: {epochs_no_improve}/{config.patience}")

            if epochs_no_improve >= config.patience:
                print(f"\n🛑 Early stopping activado. No hubo mejora en {config.patience} épocas.")
                break

        print(f"\n✅ Experimento '{config.experiment_name}' completado.")
        if best_model_state:
            print(f"Cargando el mejor modelo (Val AUC: {best_val_auc:.4f}) para la evaluación final.")
            model.load_state_dict(best_model_state)
            model.to(self.device)

            model_path = os.path.join(experiment_path, 'best_model.pth')
            torch.save(best_model_state, model_path)
            print(f"Mejor modelo guardado en: {model_path}")

            self._final_test_evaluation(model, test_loader, experiment_path)
        else:
            print("⚠️ No se guardó ningún estado de modelo.")
        
        end_time = time.time()
        print(f"Tiempo total del experimento: {format_time(end_time - start_time)}")
        return epoch_history