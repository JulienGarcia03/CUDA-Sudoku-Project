#ifndef SOLVER_CPU_HPP
#define SOLVER_CPU_HPP

bool is_board_valid(const int* board);
bool solve_sudoku_cpu_batch(const int* in_boards, int* out_boards, int batch_size);

#endif