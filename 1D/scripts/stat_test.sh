#!/bin/bash
num_it=1 

for ((i=1; i<=num_it; i++))
do
    for N in 3 7 9    
    do
        echo "Testing for N=$N, iteration $i"
        python3 scripts/test_all.py $N 0 $i --const_net
    done
done
