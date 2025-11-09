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
#SBATCH --mail-user=tu_correo@universidad.cl
#SBATCH --mail-type=BEGIN,END,FAIL

# --- 2. CAPTURAR ARGUMENTO DE MODELO ---
MODEL_NAME="BERT_Multilingual"

echo "======================================="
echo "Job iniciado en: $(date)"
echo "Usuario: $USER"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Partición: $SLURM_JOB_PARTITION"
echo "Nodos asignados: $SLURM_JOB_NODELIST"
echo "N° de CPUs: $SLURM_CPUS_PER_TASK"
echo "Modelo: $MODEL_NAME"
echo "======================================="

# --- CONFIGURACIÓN DE TELEGRAM ---
export TELEGRAM_BOT_TOKEN=8199209755:AAEOg5uoQ0rUixAL9EntGTs32JQWK7L9xOE
export TELEGRAM_CHAT_ID=878432149

mkdir -p logs
module purge

# --- ¡EDITAR AQUÍ! ---
module load python/3.13
module load CUDA/12.0.0
# ---------------------

echo "Entorno de Python cargado:"
which python

# --- 3. EJECUCIÓN DEL SCRIPT DE PYTHON ---
echo "Iniciando script de Python: hpc_run_task.py"

python run_task.py \
    --task-id $SLURM_ARRAY_TASK_ID \
    --num-workers $SLURM_CPUS_PER_TASK \
    --model-name $MODEL_NAME

echo "--- Tarea $SLURM_ARRAY_TASK_ID ($MODEL_NAME) Finalizada ---"