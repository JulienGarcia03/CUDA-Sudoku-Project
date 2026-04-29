#include "solver_cpu.hpp"
#include <cstring> // For memcpy

namespace {
    // constants matching project scale
    const int SIDE = 9;
    const int CELLS = 81;

    // check if placing 'val' at 'idx' is legal
    bool is_valid(const int* board, int idx, int val) {
        int row = idx / SIDE;
        int col = idx % SIDE;
        int box_row = (row / 3) * 3;
        int box_col = (col / 3) * 3;

        for (int i = 0; i < SIDE; ++i) {
            // check row, column, and 3x3 box
            if (board[row * SIDE + i] == val || board[i * SIDE + col] == val) return false;
            
            int r = box_row + (i / 3);
            int c = box_col + (i % 3);
            if (board[r * SIDE + c] == val) return false;
        }
        return true;
    }

    // recursive backtracking solver for a single board
    bool solve_recursive(int* board) {
        for (int i = 0; i < CELLS; ++i) {
            if (board[i] == 0) { // find empty cell
                for (int val = 1; val <= 9; ++val) {
                    if (is_valid(board, i, val)) {
                        board[i] = val;
                        if (solve_recursive(board)) return true;
                        board[i] = 0;
                    }
                }
                return false;
            }
        }
        return true; // all cells filled
    }
}

// batch solver for the performance baseline
bool solve_sudoku_cpu_batch(const int* in_boards, int* out_boards, int batch_size) {
    if (!in_boards || !out_boards || batch_size <= 0) return false;

    for (int i = 0; i < batch_size; ++i) {
        const int* src = in_boards + (i * CELLS);
        int* dst = out_boards + (i * CELLS);

        // copy input puzzle to the output buffer to solve it in place
        std::memcpy(dst, src, CELLS * sizeof(int));

        if (!solve_recursive(dst)) {
        }
    }
    return true;
}