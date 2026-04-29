#ifndef BOARD_UTILS_HPP
#define BOARD_UTILS_HPP

#include <string>
#include <vector>

namespace sudoku_utils {
    const int SIDE = 9;
    const int CELLS = 81;

    bool parse_board_line(const std::string& line, int* dst);
    std::vector<int> load_boards(const std::string& path);
    void print_board(const int* board);
    bool is_valid_at(const int* board, int idx, int val);
}

#endif