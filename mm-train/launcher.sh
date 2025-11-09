#!/bin/bash

# --- 1. DIRECTIVAS DE SLURM ---

#SBATCH --job-name=mm_train_BERT_Multilingual
#SBATCH --array=0-35
#SBATCH --partition=v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=36:00:00
#SBATCH --output=logs/mm_train_BERT_Multilingual-%A_%a.out
#SBATCH --error=logs/mm_train_BERT_Multilingual-%A_%a.err

# --- 2. CAPTURAR ARGUMENTO DE MODELO ---
MODEL_NAME="BERT_Multilingual"

echo "--- Iniciando Tarea $SLURM_ARRAY_TASK_ID para el Modelo: $MODEL_NAME ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodo: $SLURM_NODELIST"
echo "-------------------------------------"

mkdir -p logs
module purge

# --- ¡EDITAR AQUÍ! ---
module load Anaconda3/2022.10 
source activate /home/tu_usuario/ruta/a/mi_entorno_tesis 
# ---------------------

echo "Entorno de Python cargado:"
which python

# --- 3. EJECUCIÓN DEL SCRIPT DE PYTHON ---
echo "Iniciando script de Python: hpc_run_task.py"

# ❗ CAMBIO: Pasamos el --model-name al script de Python
python run_task.py \
    --task-id $SLURM_ARRAY_TASK_ID \
    --num-workers $SLURM_CPUS_PER_TASK \
    --model-name $MODEL_NAME

echo "--- Tarea $SLURM_ARRAY_TASK_ID ($MODEL_NAME) Finalizada ---"