#!/bin/bash -l

for i in 1 2 3 4 5 6 7 8 9
do

    qsub -pe omp 16 -N "outclass_n$i" -o "outclass_n$i" -j y script_class.sh $i

done
