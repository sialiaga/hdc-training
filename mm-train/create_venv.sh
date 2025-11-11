#!/bin/bash
#SBATCH --job-name=build_venv
#SBATCH --partition=general        
#SBATCH --cpus-per-task=8        
#SBATCH --mem-per-cpu=8G        
#SBATCH --time=01:00:00
#SBATCH --output=logs/build_env.out
#SBATCH --error=logs/build_env.err

echo "--- ¡Iniciando construcción de VENV en un nodo 'general' (Intel Xeon)! ---"

# 1. Carga los módulos en el orden correcto
module purge
module load intel/2019b     
module load python/3.10    
module load CUDA/12.0.0   

echo "Módulos cargados:"
module list

# 2. Navega al directorio del proyecto (asumiendo que ejecutas desde él)
# cd /home/saliaga/hdc-training/mm-train

echo "Eliminando venv_intel antiguo..."
rm -rf venv_intel

# 4. Crea el venv nuevo (ahora se construye en el CPU Intel)
echo "Creando venv_intel..."
python3 -m venv venv_intel

# 5. Actívalo
echo "Activando venv..."
source venv_intel/bin/activate

# 6. Reinstala tus librerías
echo "Instalando paquetes (esto puede tardar)..."
pip install --upgrade pip
pip install torch pandas numpy transformers scikit-learn tqdm requests

echo "--- ¡VENV construido exitosamente! ---"