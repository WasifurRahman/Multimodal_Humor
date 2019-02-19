#!/bin/sh

#SBATCH —reservation=echowdh2-20190218
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=30gb
#SBATCH -c 4
#SBATCH -a 0-7
#SBATCH -t 5-0:00:00  
#SBATCH -J humor_res_echowdh2
#SBATCH -o /scratch/echowdh2/output/humor_res_output%j
#SBATCH -e /scratch/echowdh2/output/humor_res_error%j
#SBATCH --mail-type=all    

module load anaconda3/5.3.0b
source activate wasifur
module load git
python running_different_configs.py --dataset=TED_humor


