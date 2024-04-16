#!/bin/bash
 
#SBATCH --job-name=PART          # Job name
#SBATCH --get-user-env                          # Tells the system to use the submitting user's environment
#SBATCH --mail-type=ALL                         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=lvegna@cra.com          # Where to send mail 
#SBATCH --ntasks=1                              # Run on a single CPU
#SBATCH --gres=gpu:a100_80:1                              # Request GPU
#SBATCH --mem=30GB                               # Job memory request
#SBATCH --time=120:00:00                         # Time limit hrs:min:sec
#SBATCH --partition=hai               # Specifying partition
#SBATCH --output=/share/lvegna/logs/PART_%j.log                      # Standard output and error log
cd /share/lvegna/Repos/author/authorship-embeddings
nvidia-smi
# First line of the script lets you know where you which directory you initialized in, the host you are using, and the
# date of the run
pwd; hostname; date; echo ""
 
# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1
 
# Load anaconda3
echo "Loading anaconda3"s
module load anaconda3

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/anaconda3/lib:/home/lvegna/bin

# Creating conda env if not already created, loading if it is created
CONDA_ENV_NAME=hiatus_scratch
if [[ ! -d "/share/lvegna/anaconda3/envs/$CONDA_ENV_NAME" ]]; then
    echo "Creating anaconda3 environment $CONDA_ENV_NAME..."
    conda create -p /share/lvegna/anaconda3/envs/$CONDA_ENV_NAME python=3.9 -y
    conda activate $CONDA_ENV_NAME 
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch-nightly -c nvidia
    pip3 install transformers
else
    echo "Activating anaconda3 environment $CONDA_ENV_NAME"
    conda activate $CONDA_ENV_NAME
fi
 
SWEEP_ID=$(python scripts/sweep_config_temperature.py)
# Start the wandb agent with the captured sweep ID
CUDA_VISIBLE_DEVICES=0 wandb agent $SWEEP_ID 
# &s
# CUDA_VISIBLE_DEVICES=1 wandb agent $SWEEP_ID