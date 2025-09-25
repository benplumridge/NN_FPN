#!/bin/bash

for i in 7 8 9 10; do # 

    for N in 3 7 9; do
        python scripts/train_driver.py -N $N  -run $i
        # python scripts/train_driver.py $N $i --const_net
        # python scripts/test_all.py $N --const_net
        python scripts/test_all.py $N
    done
done
