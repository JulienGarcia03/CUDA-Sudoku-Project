#pragma once

#include <cuda_runtime.h>

namespace sudoku_gpu {

constexpr int kSide = 9;
constexpr int kCells = 81;
constexpr unsigned int kFullMask = 0x1ffu;
constexpr int kDefaultThreadsPerBlock = 128;
constexpr int kFallbackThreadsPerBlock = 81;
constexpr int kMaxBlockStackDepth = 32;

enum SolveStatus : int {
    kSolved = 0,
    kNoSolution = 1,
    kInvalidBoard = 2,
    kDepthLimit = 3,
    kCudaError = 4
};

struct LaunchConfig {
    int threads_per_block = kDefaultThreadsPerBlock;
    int max_search_depth = kMaxBlockStackDepth;
    bool enable_timing = false;
};

cudaError_t launch_sudoku_kernel(
    const int* d_in_boards,
    int* d_out_boards,
    int* d_status,
    int batch_size,
    const LaunchConfig& config = {},
    cudaStream_t stream = 0
);

cudaError_t solve_batch(
    const int* h_in_boards,
    int* h_out_boards,
    int* h_status,
    int batch_size,
    const LaunchConfig& config = {},
    float* elapsed_ms = nullptr
);

const char* status_to_string(int status);

}  // namespace sudoku_gpu
