#include "solver_gpu.cuh"

#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

    bool parse_board_line(const std::string& line, std::array<int, sudoku_gpu::kCells>& board) {
        std::string compact;

        for (char ch : line) {
            if (ch == ' ' || ch == '\t' || ch == '\r') {
                continue;
            }
            compact.push_back(ch);
        }

        if (compact.empty() || compact[0] == '#') {
            return false;
        }

        if (compact.size() != sudoku_gpu::kCells) {
            throw std::runtime_error("bad puzzle width");
        }

        for (int i = 0; i < sudoku_gpu::kCells; ++i) {
            const char ch = compact[static_cast<std::size_t>(i)];
            if (ch == '.' || ch == '0') {
                board[static_cast<std::size_t>(i)] = 0;
            }
            else if (ch >= '1' && ch <= '9') {
                board[static_cast<std::size_t>(i)] = ch - '0';
            }
            else {
                throw std::runtime_error("bad puzzle character");
            }
        }

        return true;
    }

    std::vector<int> load_boards_from_any_path() {
        const char* candidates[] = {
            "data/easy.txt",
            "../data/easy.txt",
            "../../data/easy.txt"
        };

        for (const char* path : candidates) {
            std::ifstream in(path);
            if (!in) {
                continue;
            }

            std::vector<int> boards;
            std::string line;

            while (std::getline(in, line)) {
                std::array<int, sudoku_gpu::kCells> board{};
                if (parse_board_line(line, board)) {
                    boards.insert(boards.end(), board.begin(), board.end());
                }
            }

            if (!boards.empty()) {
                return boards;
            }
        }

        throw std::runtime_error("could not find data/easy.txt");
    }

    bool can_place(const std::array<int, sudoku_gpu::kCells>& board, int idx, int value) {
        const int row = idx / sudoku_gpu::kSide;
        const int col = idx % sudoku_gpu::kSide;
        const int box_row = (row / 3) * 3;
        const int box_col = (col / 3) * 3;

        for (int c = 0; c < sudoku_gpu::kSide; ++c) {
            if (board[static_cast<std::size_t>(row * sudoku_gpu::kSide + c)] == value) {
                return false;
            }
        }

        for (int r = 0; r < sudoku_gpu::kSide; ++r) {
            if (board[static_cast<std::size_t>(r * sudoku_gpu::kSide + col)] == value) {
                return false;
            }
        }

        for (int off = 0; off < sudoku_gpu::kSide; ++off) {
            const int r = box_row + off / 3;
            const int c = box_col + off % 3;
            if (board[static_cast<std::size_t>(r * sudoku_gpu::kSide + c)] == value) {
                return false;
            }
        }

        return true;
    }

    // simple cpu baseline for correctness only

    bool cpu_solve(std::array<int, sudoku_gpu::kCells>& board) {
        int empty = -1;

        for (int i = 0; i < sudoku_gpu::kCells; ++i) {
            if (board[static_cast<std::size_t>(i)] == 0) {
                empty = i;
                break;
            }
        }

        if (empty == -1) {
            return true;
        }

        for (int value = 1; value <= 9; ++value) {
            if (!can_place(board, empty, value)) {
                continue;
            }

            board[static_cast<std::size_t>(empty)] = value;
            if (cpu_solve(board)) {
                return true;
            }
            board[static_cast<std::size_t>(empty)] = 0;
        }

        return false;
    }

    bool board_is_complete_and_valid(const int* board) {
        for (int row = 0; row < sudoku_gpu::kSide; ++row) {
            bool seen_row[10] = {};
            bool seen_col[10] = {};

            for (int col = 0; col < sudoku_gpu::kSide; ++col) {
                const int row_value = board[row * sudoku_gpu::kSide + col];
                const int col_value = board[col * sudoku_gpu::kSide + row];

                if (row_value < 1 || row_value > 9 || seen_row[row_value]) {
                    return false;
                }
                if (col_value < 1 || col_value > 9 || seen_col[col_value]) {
                    return false;
                }

                seen_row[row_value] = true;
                seen_col[col_value] = true;
            }
        }

        for (int box = 0; box < sudoku_gpu::kSide; ++box) {
            bool seen_box[10] = {};
            const int base_row = (box / 3) * 3;
            const int base_col = (box % 3) * 3;

            for (int off = 0; off < sudoku_gpu::kSide; ++off) {
                const int row = base_row + off / 3;
                const int col = base_col + off % 3;
                const int value = board[row * sudoku_gpu::kSide + col];

                if (value < 1 || value > 9 || seen_box[value]) {
                    return false;
                }

                seen_box[value] = true;
            }
        }

        return true;
    }

}  // namespace

int main() {
    int device_count = 0;
    const cudaError_t init_err = cudaGetDeviceCount(&device_count);

    if (init_err != cudaSuccess || device_count == 0) {
        std::cout << "skipping test no cuda device\n";
        return 0;
    }

    std::vector<int> boards = load_boards_from_any_path();
    const int batch_size = static_cast<int>(boards.size() / sudoku_gpu::kCells);

    std::vector<int> gpu_out(boards.size(), 0);
    std::vector<int> gpu_status(static_cast<std::size_t>(batch_size), sudoku_gpu::kCudaError);

    sudoku_gpu::LaunchConfig config;
    config.threads_per_block = sudoku_gpu::kDefaultThreadsPerBlock;
    config.max_search_depth = sudoku_gpu::kMaxBlockStackDepth;

    float elapsed_ms = 0.0f;
    const cudaError_t solve_err = sudoku_gpu::solve_batch(
        boards.data(),
        gpu_out.data(),
        gpu_status.data(),
        batch_size,
        config,
        &elapsed_ms
    );

    if (solve_err != cudaSuccess) {
        std::cerr << "gpu solver failed " << cudaGetErrorString(solve_err) << "\n";
        return 1;
    }

    for (int board_idx = 0; board_idx < batch_size; ++board_idx) {
        std::array<int, sudoku_gpu::kCells> expected{};

        std::copy_n(
            boards.data() + static_cast<std::size_t>(board_idx) * sudoku_gpu::kCells,
            sudoku_gpu::kCells,
            expected.begin()
        );

        if (!cpu_solve(expected)) {
            std::cerr << "cpu baseline could not solve sample puzzle " << board_idx << "\n";
            return 1;
        }

        if (gpu_status[static_cast<std::size_t>(board_idx)] != sudoku_gpu::kSolved) {
            std::cerr << "gpu status was not solved for puzzle " << board_idx << "\n";
            return 1;
        }

        const int* gpu_board = gpu_out.data() + static_cast<std::size_t>(board_idx) * sudoku_gpu::kCells;

        if (!board_is_complete_and_valid(gpu_board)) {
            std::cerr << "gpu board is not a valid completed sudoku " << board_idx << "\n";
            return 1;
        }

        for (int cell = 0; cell < sudoku_gpu::kCells; ++cell) {
            if (gpu_board[cell] != expected[static_cast<std::size_t>(cell)]) {
                std::cerr << "gpu board did not match cpu baseline " << board_idx << "\n";
                return 1;
            }
        }
    }

    // reject a row duplicate

    std::array<int, sudoku_gpu::kCells> invalid_board{};
    invalid_board.fill(0);
    invalid_board[0] = 5;
    invalid_board[1] = 5;

    std::vector<int> invalid_in(invalid_board.begin(), invalid_board.end());
    std::vector<int> invalid_out(static_cast<std::size_t>(sudoku_gpu::kCells), 0);
    std::vector<int> invalid_status(1, sudoku_gpu::kCudaError);

    const cudaError_t invalid_err = sudoku_gpu::solve_batch(
        invalid_in.data(),
        invalid_out.data(),
        invalid_status.data(),
        1,
        config,
        nullptr
    );

    if (invalid_err != cudaSuccess) {
        std::cerr << "invalid board launch failed " << cudaGetErrorString(invalid_err) << "\n";
        return 1;
    }

    if (invalid_status[0] != sudoku_gpu::kInvalidBoard) {
        std::cerr << "invalid board was not flagged invalid\n";
        return 1;
    }

    std::cout << "all correctness checks passed in " << elapsed_ms << " ms\n";
    return 0;
}
