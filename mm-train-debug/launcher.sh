#!/bin/bash

# --- 1. DIRECTIVAS DE SLURM ---

#SBATCH --job-name=mm_debug_DistilBERT
#SBATCH --array=0-3
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/mm_train_DistilBERT_Multilingual-%A_%a.out
#SBATCH --error=logs/mm_train_DistilBERT_Multilingual-%A_%a.err
#SBATCH --mail-user=sialiaga@miuandes.cl
#SBATCH --mail-type=BEGIN,END,FAIL

# --- 2. CAPTURAR ARGUMENTO DE MODELO ---
MODEL_NAME="distilbert"

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
export TELEGRAM_BOT_TOKEN=8199209755:AAG3169tZmAJbAsDDgi0OIK173K-avq2_BA
export TELEGRAM_CHAT_ID=878432149

echo "Limpiando..."
mkdir -p logs
module purge

echo "Cargando Python..."
module load python/3.10

echo "Cargando CUDA 12.0..."
module load CUDA/12.0.0

echo "Activando entorno virtual..."
source venv/bin/activate


echo "Iniciando script de Python: run_task.py"

python run_task.py \
    --task-id $SLURM_ARRAY_TASK_ID \
    --num-workers $SLURM_CPUS_PER_TASK \
    --model-name $MODEL_NAME

echo "--- Tarea $SLURM_ARRAY_TASK_ID ($MODEL_NAME) Finalizada ---"