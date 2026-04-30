#!/usr/bin/env bash

set -e

mkdir -p results
mkdir -p data/generated

echo "building project"
make clean
make

echo "creating scaled test batches"

cp data/easy.txt data/generated/easy_1x.txt
cat data/easy.txt data/easy.txt > data/generated/easy_2x.txt
cat data/generated/easy_2x.txt data/generated/easy_2x.txt > data/generated/easy_4x.txt
cat data/generated/easy_4x.txt data/generated/easy_4x.txt > data/generated/easy_8x.txt

echo "batch,gpu_time_ms,boards_per_second,solved,invalid,depth_limit,no_solution" > results/gpu_results.csv

run_case () {
    label=$1
    file=$2

    echo "running $label"

    output=$(./bin/sudoku_solver "$file")

    echo "$output"

    batch=$(echo "$output" | awk '/batch size/ {print $3}')
    gpu_ms=$(echo "$output" | awk '/gpu time ms/ {print $4}')
    bps=$(echo "$output" | awk '/boards per second/ {print $4}')
    solved=$(echo "$output" | awk '/solved/ {print $2}')
    invalid=$(echo "$output" | awk '/invalid/ {print $2}')
    depth_limit=$(echo "$output" | awk '/depth limit/ {print $3}')
    no_solution=$(echo "$output" | awk '/no solution/ {print $3}')

    echo "$batch,$gpu_ms,$bps,$solved,$invalid,$depth_limit,$no_solution" >> results/gpu_results.csv
}

run_case "easy 1x" data/generated/easy_1x.txt
run_case "easy 2x" data/generated/easy_2x.txt
run_case "easy 4x" data/generated/easy_4x.txt
run_case "easy 8x" data/generated/easy_8x.txt

run_case "medium" data/medium.txt 32 128
run_case "hard" data/hard.txt 32 128

echo ""
echo "results saved to results/gpu_results.csv"
cat results/gpu_results.csv