#!/bin/bash
echo ID,Expected > submit.csv

i=1
while read line
    do
	echo $i,$line >> submit.csv
	i=$(($i+1))
    done < tmp.txt
