#!/bin/sh


#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=20gb
#SBATCH -c 4
#SBATCH -a 31-35
#SBATCH -t 0-2:00:00  
#SBATCH -J humor_normal_echowdh2
#SBATCH -o /scratch/echowdh2/output/humor_normal_output%j
#SBATCH -e /scratch/echowdh2/output/humor_normal_error%j
#SBATCH --mail-type=all    

module load anaconda3/5.3.0b
source activate wasifur
module load git
python running_different_configs.py --dataset=TED_humor


