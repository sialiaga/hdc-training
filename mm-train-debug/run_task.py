# hpc_run_task.py

import sys
import argparse
import traceback
import time
import json 
import os   

from config_factory import get_experiments_for_model
from mm_project import ExperimentRunner, enviar_notificacion, format_time



def main():
    parser = argparse.ArgumentParser(description="Ejecutor de Tareas de Experimento Multimodal HPC")
    parser.add_argument("--task-id", type=int, required=True, help="ID de la tarea (de 0 a 35).")
    parser.add_argument("--num-workers", type=int, default=0, help="Número de workers.")
    parser.add_argument("--model-name", type=str, required=True, help="Nombre del modelo (ej. 'BETO').")
    
    args = parser.parse_args()
    
    task_id = args.task_id
    num_workers = args.num_workers
    model_name = args.model_name

    print(f"--- 🚀 Iniciando Trabajador de Tarea ---")
    print(f"Modelo: {model_name} | Tarea: {task_id}")
    print("-------------------------------------")

    config = None # Definimos config aquí para usarlo en el except
    try:
        model_experiments = get_experiments_for_model(model_filter=model_name)
        if task_id >= len(model_experiments):
            print(f"¡Error! Task ID {task_id} está fuera de rango.")
            sys.exit(1)
        config = model_experiments[task_id]

        # --- 1. NOTIFICACIÓN DE INICIO DE TAREA ---
        start_msg = f"🚀 Tarea {task_id} ({config.experiment_name}) INICIADA."
        enviar_notificacion(start_msg)
        
        start_time = time.time()
        runner = ExperimentRunner()
        
        # --- EJECUCIÓN ---
        runner.run(config, num_workers=num_workers)
        
        # --- 2. NOTIFICACIÓN DE ÉXITO (con métricas) ---
        duration_str = format_time(time.time() - start_time)
        
        # Cargar el JSON de resultados que se acaba de guardar
        test_auc = "N/A"
        test_acc = "N/A"
        try:
            conjunto_path = os.path.join(config.results_base_folder, config.conjunto_name)
            experiment_path = os.path.join(conjunto_path, config.experiment_name)
            metrics_path = os.path.join(experiment_path, "test_evaluation", "test_metrics.json")
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            test_auc = f"{metrics['test_auc']:.4f}"
            test_acc = f"{metrics['test_accuracy']:.4f}"
        except Exception as e:
            print(f"WARNING: No se pudo leer el JSON de métricas para la notificación. {e}")

        success_msg = (
            f"✅ Tarea {task_id} ({config.experiment_name}) FINALIZADA\n"
            f"Duración: {duration_str}\n"
            f"Test AUC: {test_auc}\n"
            f"Test Acc: {test_acc}"
        )
        enviar_notificacion(success_msg)

    except Exception as e:
        print(f"\n¡Ha ocurrido un error crítico!")
        traceback.print_exc()
        
        # --- 3. NOTIFICACIÓN DE FALLO ---
        exp_name = config.experiment_name if config else f"Modelo {model_name} (Tarea {task_id})"
        error_msg = f"🛑 ¡FALLO CRÍTICO! 🛑\n\n*Tarea*: {exp_name}\n*Error*: {str(e)}"
        enviar_notificacion(error_msg)
        
        sys.exit(1) # Asegura que SLURM marque la tarea como FAILED

    print(f"\n--- ✅ Tarea {task_id} ({config.experiment_name}) finalizada exitosamente ---")

if __name__ == "__main__":
    main()