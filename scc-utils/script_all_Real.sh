#!/bin/bash -l

for i in 1 2 3 4 5 6 7 8 9 10
do

    qsub -pe omp 16 -N "outReal_n$i" -o "outReal_n$i" -j y script_Real.sh $i

done
