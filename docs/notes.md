# gpu implementation notes

this patch owns the person 2 work

## files in this change

- src/solver_gpu.cuh
- src/solver_gpu.cu
- src/main.cu
- Makefile
- tests/test_correctness.cpp
- data/easy.txt

## kernel shape

- one block owns one board
- threads 0 through 80 map to cells
- extra threads in a 128 thread block are still useful for barriers and shared loops
- board state row masks col masks box masks and branch checkpoints live in shared memory
- propagation uses naked singles and hidden singles
- search uses a bounded shared stack with rollback

## cpu merge contract

if person 1 adds the cpu baseline keep this function signature in `src/solver_cpu.hpp`

```cpp
bool solve_sudoku_cpu_batch(const int* in_boards, int* out_boards, int batch_size);
	