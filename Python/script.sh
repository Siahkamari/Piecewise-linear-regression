module load python3
module load pytorch
module load cuda

jupyter nbconvert --execute --to notebook --inplace example_CPU.ipynb