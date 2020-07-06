#!/bin/bash -l
 
# Specify the version of MATLAB to be used
module load gurobi/8.0.0
module load matlab/2018b

matlab -nodisplay -singleCompThread -r "addpath(getenv('SCC_GUROBI_BIN')),cF = cd('~/Documents/gurobi811/linux64/matlab/'), gurobi_setup, cd(cF), dc_test_multi_class($1), exit"
