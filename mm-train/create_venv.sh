#!/bin/bash
#SBATCH --job-name=build_venv
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/build_env.out
#SBATCH --error=logs/build_env.err

echo "--- ¡Iniciando construcción de VENV en un nodo 'general' (Intel Xeon)! ---"

module purge
module load intel/2019b     
module load CUDA/12.0.0    

echo "Módulos cargados:"
module list

echo "Ruta de Python (debería ser de 'intel/2019b'):"
which python

echo "Versión de Python:"
python -V 

echo "Eliminando venv_intel antiguo..."
rm -rf venv_intel

echo "Creando venv_intel..."
python -m venv venv_intel

echo "Activando venv..."
source venv_intel/bin/activate

# 6. Reinstala tus librerías
echo "Instalando paquetes (esto puede tardar)..."
pip install --upgrade pip
pip install torch pandas numpy transformers scikit-learn tqdm requests

echo "--- ¡VENV construido exitosamente! ---"