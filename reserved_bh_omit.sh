#!/bin/sh

#SBATCH --reservation=echowdh2-20190220
#SBATCH -p reserved --gres=gpu:1
#SBATCH --mem=30gb
#SBATCH -c 4
#SBATCH -a 10-17
#SBATCH -t 3-0:00:00  
#SBATCH -J humor_omit_echowdh2
#SBATCH -o /scratch/echowdh2/output/humor_omit_output%j
#SBATCH -e /scratch/echowdh2/output/humor_omit_error%j
#SBATCH --mail-type=all    

module load anaconda3/5.3.0b
source activate wasifur
module load git
python running_different_configs.py --dataset=TED_humor


