#!/bin/bash

num_it=10     
num_abl=11

for ((i=1; i<=num_it; i++))
do
    for N in 3 7 9        
    do
        echo "Iteration $i for N=$N"
        python3 scripts/train_driver.py --N $N   
    
        for ((j=0; j<num_abl; j++))
        do
            echo "Testing ablation_idx=$j for N=$N, iteration $i"
            python3 scripts/test_all.py --N $N --ablation_idx $j
        done

        echo "Training const_net for N=$N (iteration $i)"
        python3 scripts/train_driver.py --N $N --const_net

        echo "Testing const_net for N=$N (iteration $i)"
        python3 scripts/test_all.py --N $N --const_net
    done
done
