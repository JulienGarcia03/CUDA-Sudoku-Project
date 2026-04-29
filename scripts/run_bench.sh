#!/bin/bash

#datasets & thread configurations to be tested
DATASETS=("data/easy.txt" "data/medium.txt" "data/hard.txt")
THREAD=(64 128 256)
DEPTH=50 #max search depth, arbitrary, to prevent infinite loops on hard boards

echo "starting gpu sudoku benchmarks..."
echo "------------------------------------------------"

#loop thru every combination of dataset & thread count
for data_file in "${DATASETS[@]}"; do
  for t_count in "${THREAD[@]}"; do
    echo "Testing: File=$data_file | Threads=$t_count"

    #run solver & capture output
    ./bin/sudoku_solver "$data_file" $DEPTH $t_count

    echo "------------------------------------------------"
  done
done

echo "benchmarking complete"