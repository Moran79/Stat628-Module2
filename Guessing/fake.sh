#!/bin/bash
echo ID,Expected > submit.csv

for i in {1..1321274}; do
    echo $i,3.7 >> submit.csv
done
