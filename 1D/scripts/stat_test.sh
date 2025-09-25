#!/bin/bash

num_it=10     
num_abl=11

for ((i=1; i<=num_it; i++))
do
    for N in 3 7 9        
    do
        for ((j=1; j<num_abl; j++))
        do
            echo "Testing ablation_idx=$j for N=$N, iteration $i"
            python3 scripts/test_all.py $N $j $i
        done
    done
done
