# hpc_run_task.py

import sys
import argparse
import traceback

# ❗ CAMBIO: Importa la nueva función
from config_factory import get_experiments_for_model
from mm_project import ExperimentRunner

def main():
    parser = argparse.ArgumentParser(description="Ejecutor de Tareas de Experimento Multimodal HPC")
    parser.add_argument(
        "--task-id",
        type=int,
        required=True,
        help="ID de la tarea (de 0 a 35) del array de SLURM."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Número de workers para el DataLoader."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Nombre del modelo a ejecutar (ej. 'BETO', 'XLM-R')."
    )
    args = parser.parse_args()
    
    task_id = args.task_id
    num_workers = args.num_workers
    model_name = args.model_name 

    print(f"--- 🚀 Iniciando Trabajador de Tarea ---")
    print(f"Modelo a ejecutar: {model_name}")
    print(f"ID de Tarea (relativo): {task_id}")
    print(f"Workers para DataLoader: {num_workers}")
    print("-------------------------------------")

    try:
        model_experiments = get_experiments_for_model(model_filter=model_name)
        
        if task_id >= len(model_experiments):
            print(f"¡Error! Task ID {task_id} está fuera de rango. El modelo {model_name} solo tiene {len(model_experiments)} experimentos (de 0 a {len(model_experiments)-1}).")
            sys.exit(1)
            
        # Selecciona la tarea de la lista filtrada de 36
        config = model_experiments[task_id]

    except Exception as e:
        print(f"Error fatal al generar la configuración: {e}")
        traceback.print_exc()
        sys.exit(1)

    try:
        runner = ExperimentRunner()
        runner.run(config, num_workers=num_workers)
        
    except Exception as e:
        print(f"\n¡Ha ocurrido un error crítico durante la ejecución del experimento: {config.experiment_name}!")
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1) 

    print(f"\n--- ✅ Tarea {task_id} ({config.experiment_name}) finalizada exitosamente ---")

if __name__ == "__main__":
    main()