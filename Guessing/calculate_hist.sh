#!/bin/bash

for i in {1..10}; do
    ARRAY=(0 0 0 0 0)
    gshuf -n 1000 review_train.json > tmp.json
    while read line
    do
	number=$(echo "$line" | grep -o "stars\": [0-9]" | grep -o "[0-9]")
	ARRAY[$(($number-1))]=$((${ARRAY[$(($number-1))]} + 1))
    done < tmp.json
    echo ${ARRAY[*]} >> final.txt
done
