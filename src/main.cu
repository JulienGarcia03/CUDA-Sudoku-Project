#include "solver_gpu.cuh"

#ifdef HAVE_CPU_BASELINE
#include "solver_cpu.hpp"
#endif

#include <cstddef>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// one 81 cell board per non empty line

bool parse_board_line(const std::string& line, int* dst) {
    std::string compact;
    compact.reserve(line.size());

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
        throw std::runtime_error("each puzzle line must contain 81 cells");
    }

    for (int i = 0; i < sudoku_gpu::kCells; ++i) {
        const char ch = compact[static_cast<std::size_t>(i)];
        if (ch == '.' || ch == '0') {
            dst[i] = 0;
        } else if (ch >= '1' && ch <= '9') {
            dst[i] = ch - '0';
        } else {
            throw std::runtime_error("puzzle file contains a non sudoku character");
        }
    }

    return true;
}

std::vector<int> load_boards(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("could not open puzzle file");
    }

    std::vector<int> boards;
    std::string line;

    while (std::getline(in, line)) {
        int board[sudoku_gpu::kCells];
        if (parse_board_line(line, board)) {
            boards.insert(boards.end(), board, board + sudoku_gpu::kCells);
        }
    }

    if (boards.empty()) {
        throw std::runtime_error("no puzzles were loaded");
    }

    return boards;
}

void print_board(const int* board) {
    for (int row = 0; row < sudoku_gpu::kSide; ++row) {
        for (int col = 0; col < sudoku_gpu::kSide; ++col) {
            const int value = board[row * sudoku_gpu::kSide + col];
            std::cout << value;
            if (col + 1 != sudoku_gpu::kSide) {
                std::cout << ' ';
            }
        }
        std::cout << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "usage  ./bin/sudoku_solver data/easy.txt [depth] [threads]\n";
            return 1;
        }

        const std::string data_path = argv[1];
        std::vector<int> boards = load_boards(data_path);
        const int batch_size = static_cast<int>(boards.size() / sudoku_gpu::kCells);

        sudoku_gpu::LaunchConfig config;
        if (argc >= 3) {
            config.max_search_depth = std::stoi(argv[2]);
        }
        if (argc >= 4) {
            config.threads_per_block = std::stoi(argv[3]);
            if(config.threads_per_block<81){ // guard added for proper mapping of threads
                std::cerr << "warning: threads per block must be >= 81 to map to board. defaulting to 128.\n";
                config.threads_per_block = 128;
            }
        }

        std::vector<int> gpu_out(boards.size(), 0);
        std::vector<int> gpu_status(static_cast<std::size_t>(batch_size), sudoku_gpu::kCudaError);

        float elapsed_ms = 0.0f;
        const cudaError_t err = sudoku_gpu::solve_batch(
            boards.data(),
            gpu_out.data(),
            gpu_status.data(),
            batch_size,
            config,
            &elapsed_ms
        );

        if (err != cudaSuccess) {
            std::cerr << "cuda launch failed  " << cudaGetErrorString(err) << "\n";
            return 1;
        }

        int solved = 0;
        int invalid = 0;
        int depth_limit = 0;
        int no_solution = 0;

        for (int i = 0; i < batch_size; ++i) {
            switch (gpu_status[static_cast<std::size_t>(i)]) {
                case sudoku_gpu::kSolved:
                    ++solved;
                    break;
                case sudoku_gpu::kInvalidBoard:
                    ++invalid;
                    break;
                case sudoku_gpu::kDepthLimit:
                    ++depth_limit;
                    break;
                default:
                    ++no_solution;
                    break;
            }
        }

        std::cout << "batch size          " << batch_size << "\n";
        std::cout << "threads per block   " << config.threads_per_block << "\n";
        std::cout << "max search depth    " << config.max_search_depth << "\n";
        std::cout << "gpu time ms         " << std::fixed << std::setprecision(3) << elapsed_ms << "\n";

        if (elapsed_ms > 0.0f) {
            const double boards_per_second = 1000.0 * static_cast<double>(batch_size) / static_cast<double>(elapsed_ms);
            std::cout << "boards per second   " << std::setprecision(1) << boards_per_second << "\n";
        }

        std::cout << "solved              " << solved << "\n";
        std::cout << "invalid             " << invalid << "\n";
        std::cout << "depth limit         " << depth_limit << "\n";
        std::cout << "no solution         " << no_solution << "\n";

#ifdef HAVE_CPU_BASELINE
        std::vector<int> cpu_out(boards.size(), 0);

        // merge contract with person 1
        const bool cpu_ok = solve_sudoku_cpu_batch(boards.data(), cpu_out.data(), batch_size);

        if (cpu_ok) {
            int mismatches = 0;

            for (int i = 0; i < batch_size; ++i) {
                if (gpu_status[static_cast<std::size_t>(i)] != sudoku_gpu::kSolved) {
                    continue;
                }

                const int* gpu_board = gpu_out.data() + static_cast<std::size_t>(i) * sudoku_gpu::kCells;

                if(!is_board_valid(gpu_board)){  //using cpu baseline logic to verify gpu's math
                    ++mismatches;
                }
            }

            std::cout << "validation status      " << (mismatches == 0 ? "passed" : "failed") << "\n";
        } else {
            std::cout << "cpu validation      skipped by cpu baseline\n";
        }
#else
        std::cout << "cpu validation      skipped no cpu baseline in build\n";
#endif

        if (batch_size > 0) {
            std::cout << "\nfirst gpu board output\n";
            print_board(gpu_out.data());
            std::cout << "status              " << sudoku_gpu::status_to_string(gpu_status[0]) << "\n";
        }
    } catch (const std::exception& ex) {
        std::cerr << "error  " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
