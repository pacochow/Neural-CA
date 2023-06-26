#!/bin/bash
#$ -q long.qc 
#$ -j y
#$ -r no 
#$ -wd /cs/student/msc/ml/2022/pacochow/Neural-CA

echo "------------------------------------------------"
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "------------------------------------------------"

source ~/.bashrc
conda activate cpu
python train_with_env.py