#!/bin/bash

OUTPUT_CSV="docs/final_metrics.csv"
DATASETS=("data/easy.txt" "data/medium.txt" "data/hard.txt")
THREADS=(64 128 256)
DEPTH=50

#print csv headers
echo "dataset, threads, boards_per_second" > $OUTPUT_CSV
echo "gathering clean performance metrics... (saving to $OUTPUT_CSV)"

for data_file in "${DATASETS[@]}"; do #loop thru every combination
    dataset_name=$(basename "$data_file" .txt) #extract only difficulty word 

    for t_count in "${THREADS[@]}"; do
        #solver grabbing only line with thruput number
        bps=$(./bin/sudoku_solver "$data_file" $DEPTH $t_count | grep "boards per second" | awk '{print $4}')
        
        if [ -z "$bps" ]; then #if failed then no bps printed & default to 0
            bps="0.0"
        fi
        
        echo "$dataset_name,$t_count,$bps" >> $OUTPUT_CSV #append to csv file
        
    done
done

echo "collection complete. here is the raw data:"
echo "-------------------------------------------"
cat $OUTPUT_CSV
echo "-------------------------------------------"