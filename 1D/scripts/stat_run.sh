#!/bin/bash

num_it=2   
num_abl=1

for ((i=1; i<=num_it; i++))
do
    for N in 3 7 9        
    do
        echo "Iteration $i for N=$N"
        python3 scripts/train_driver.py $N   
    
        for ((j=0; j<num_abl; j++))
        do
            echo "Testing ablation_idx=$j for N=$N, iteration $i"
            python3 scripts/test_all.py $N ablation_idx $j
        done

        echo "Training const_net for N=$N (iteration $i)"
        python3 scripts/train_driver.py $N --const_net

        echo "Testing const_net for N=$N (iteration $i)"
        python3 scripts/test_all.py $N --const_net
    done
done
