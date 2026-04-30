#include "board_utils.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>

namespace sudoku_utils {

// parse one line into 81 cell board
bool parse_board_line(const std::string& line, int* dst) {
    if (!dst) return false;

    std::string compact;

    for (char ch : line) {
        if (ch == ' ' || ch == '\t' || ch == '\r') continue;
        compact.push_back(ch);
    }

    if (compact.empty() || compact[0] == '#') return false;

    if (compact.size() != CELLS) {
        throw std::runtime_error("line must contain 81 cells");
    }

    for (int i = 0; i < CELLS; ++i) {
        char ch = compact[i];

        if (ch == '.' || ch == '0') dst[i] = 0;
        else if (ch >= '1' && ch <= '9') dst[i] = ch - '0';
        else throw std::runtime_error("invalid sudoku character");
    }

    return true;
}

// load all boards from file
std::vector<int> load_boards(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("failed to open file");

    std::vector<int> boards;
    std::string line;

    while (std::getline(in, line)) {
        int board[CELLS];
        if (parse_board_line(line, board)) {
            boards.insert(boards.end(), board, board + CELLS);
        }
    }

    if (boards.empty()) {
        throw std::runtime_error("no boards loaded");
    }

    return boards;
}

// print 9x9 board
void print_board(const int* board) {
    if (!board) return;

    for (int r = 0; r < SIDE; ++r) {
        for (int c = 0; c < SIDE; ++c) {
            std::cout << board[r * SIDE + c];
            if (c != SIDE - 1) std::cout << ' ';
        }
        std::cout << '\n';
    }
}

// check if placing val at idx is valid
bool is_valid_at(const int* board, int idx, int val) {
    int row = idx / SIDE;
    int col = idx % SIDE;

    int box_r = (row / 3) * 3;
    int box_c = (col / 3) * 3;

    for (int i = 0; i < SIDE; ++i) {
        if (board[row * SIDE + i] == val) return false;
        if (board[i * SIDE + col] == val) return false;

        int r = box_r + i / 3;
        int c = box_c + i % 3;
        if (board[r * SIDE + c] == val) return false;
    }

    return true;
}

} // namespace sudoku_utils