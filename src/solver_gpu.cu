#include "solver_gpu.cuh"
#include <cstddef>

namespace sudoku_gpu {
namespace {

// shared working set for one board

struct BlockState {
    int board[kCells];
    unsigned int row_used[kSide];
    unsigned int col_used[kSide];
    unsigned int box_used[kSide];

    unsigned int cell_mask[kCells];
    int cell_count[kCells];
    int single_value[kCells];

    int final_status;
    int depth;
    int hit_depth_limit;

    int branch_cell;
    unsigned int branch_mask;
    int next_guess;

    int stack_board[kMaxBlockStackDepth][kCells];
    int stack_guess_cell[kMaxBlockStackDepth];
    unsigned int stack_remaining_mask[kMaxBlockStackDepth];
};

__device__ __forceinline__ int cell_row(int idx) {
    return idx / kSide;
}

__device__ __forceinline__ int cell_col(int idx) {
    return idx % kSide;
}

__device__ __forceinline__ int cell_box(int row, int col) {
    return (row / 3) * 3 + (col / 3);
}

__device__ __forceinline__ unsigned int digit_bit(int value) {
    return 1u << (value - 1);
}

__device__ __forceinline__ int digit_from_mask(unsigned int mask) {
    return __ffs(static_cast<int>(mask));
}

// rebuild row col and box masks from the current shared board

__device__ bool rebuild_constraints(BlockState& s) {
    const int tid = static_cast<int>(threadIdx.x);

    if (tid < kSide) {
        s.row_used[tid] = 0u;
        s.col_used[tid] = 0u;
        s.box_used[tid] = 0u;
    }

    __syncthreads();

    int local_invalid = 0;

    if (tid < kCells) {
        const int value = s.board[tid];

        if (value < 0 || value > 9) {
            local_invalid = 1;
        } else if (value != 0) {
            const int row = cell_row(tid);
            const int col = cell_col(tid);
            const int box = cell_box(row, col);
            const unsigned int bit = digit_bit(value);

            const unsigned int old_row = atomicOr(&s.row_used[row], bit);
            const unsigned int old_col = atomicOr(&s.col_used[col], bit);
            const unsigned int old_box = atomicOr(&s.box_used[box], bit);

            if ((old_row & bit) != 0u || (old_col & bit) != 0u || (old_box & bit) != 0u) {
                local_invalid = 1;
            }
        }
    }

    return __syncthreads_or(local_invalid) != 0;
}

// compute candidate masks for all empty cells

__device__ bool compute_candidates(BlockState& s) {
    const int tid = static_cast<int>(threadIdx.x);
    int local_invalid = 0;

    if (tid < kCells) {
        s.single_value[tid] = 0;

        if (s.board[tid] == 0) {
            const int row = cell_row(tid);
            const int col = cell_col(tid);
            const int box = cell_box(row, col);

            const unsigned int used = s.row_used[row] | s.col_used[col] | s.box_used[box];
            const unsigned int mask = kFullMask & ~used;
            const int count = __popc(mask);

            s.cell_mask[tid] = mask;
            s.cell_count[tid] = count;

            if (count == 0) {
                local_invalid = 1;
            } else if (count == 1) {
                s.single_value[tid] = digit_from_mask(mask);
            }
        } else {
            s.cell_mask[tid] = digit_bit(s.board[tid]);
            s.cell_count[tid] = 1;
        }
    }

    return __syncthreads_or(local_invalid) != 0;
}

// add hidden singles using the candidate masks already in shared memory

__device__ bool add_hidden_singles(BlockState& s) {
    int conflict = 0;

    if (threadIdx.x == 0) {
        for (int row = 0; row < kSide; ++row) {
            for (int digit = 1; digit <= 9; ++digit) {
                const unsigned int bit = digit_bit(digit);
                int found = -1;
                int hits = 0;

                for (int col = 0; col < kSide; ++col) {
                    const int idx = row * kSide + col;
                    if (s.board[idx] == 0 && (s.cell_mask[idx] & bit) != 0u) {
                        found = idx;
                        ++hits;
                        if (hits > 1) {
                            break;
                        }
                    }
                }

                if (hits == 1) {
                    if (s.single_value[found] == 0 || s.single_value[found] == digit) {
                        s.single_value[found] = digit;
                    } else {
                        conflict = 1;
                    }
                }
            }
        }

        for (int col = 0; col < kSide; ++col) {
            for (int digit = 1; digit <= 9; ++digit) {
                const unsigned int bit = digit_bit(digit);
                int found = -1;
                int hits = 0;

                for (int row = 0; row < kSide; ++row) {
                    const int idx = row * kSide + col;
                    if (s.board[idx] == 0 && (s.cell_mask[idx] & bit) != 0u) {
                        found = idx;
                        ++hits;
                        if (hits > 1) {
                            break;
                        }
                    }
                }

                if (hits == 1) {
                    if (s.single_value[found] == 0 || s.single_value[found] == digit) {
                        s.single_value[found] = digit;
                    } else {
                        conflict = 1;
                    }
                }
            }
        }

        for (int box = 0; box < kSide; ++box) {
            const int base_row = (box / 3) * 3;
            const int base_col = (box % 3) * 3;

            for (int digit = 1; digit <= 9; ++digit) {
                const unsigned int bit = digit_bit(digit);
                int found = -1;
                int hits = 0;

                for (int off = 0; off < kSide; ++off) {
                    const int row = base_row + off / 3;
                    const int col = base_col + off % 3;
                    const int idx = row * kSide + col;

                    if (s.board[idx] == 0 && (s.cell_mask[idx] & bit) != 0u) {
                        found = idx;
                        ++hits;
                        if (hits > 1) {
                            break;
                        }
                    }
                }

                if (hits == 1) {
                    if (s.single_value[found] == 0 || s.single_value[found] == digit) {
                        s.single_value[found] = digit;
                    } else {
                        conflict = 1;
                    }
                }
            }
        }
    }

    return __syncthreads_or(conflict) != 0;
}

// repeat singles until the board stops changing

__device__ bool propagate_to_fixpoint(BlockState& s) {
    while (true) {
        if (rebuild_constraints(s)) {
            return false;
        }

        if (compute_candidates(s)) {
            return false;
        }

        __syncthreads();

        if (add_hidden_singles(s)) {
            return false;
        }

        __syncthreads();

        int made_progress = 0;

        if (threadIdx.x < kCells) {
            const int tid = static_cast<int>(threadIdx.x);
            if (s.board[tid] == 0 && s.single_value[tid] != 0) {
                s.board[tid] = s.single_value[tid];
                made_progress = 1;
            }
        }

        if (__syncthreads_or(made_progress) == 0) {
            break;
        }
    }

    return true;
}

__device__ bool board_is_complete(const BlockState& s) {
    int has_empty = 0;

    if (threadIdx.x < kCells) {
        has_empty = (s.board[threadIdx.x] == 0) ? 1 : 0;
    }

    return __syncthreads_or(has_empty) == 0;
}

__device__ void choose_branch_cell(BlockState& s) {
    if (threadIdx.x == 0) {
        int best_cell = -1;
        int best_count = 10;
        unsigned int best_mask = 0u;

        for (int idx = 0; idx < kCells; ++idx) {
            if (s.board[idx] == 0) {
                const int count = s.cell_count[idx];
                if (count > 1 && count < best_count) {
                    best_count = count;
                    best_cell = idx;
                    best_mask = s.cell_mask[idx];
                }
            }
        }

        s.branch_cell = best_cell;
        s.branch_mask = best_mask;
    }

    __syncthreads();
}

// save the current board before a speculative guess

__device__ void save_checkpoint(BlockState& s, int depth) {
    for (int idx = static_cast<int>(threadIdx.x); idx < kCells; idx += static_cast<int>(blockDim.x)) {
        s.stack_board[depth][idx] = s.board[idx];
    }

    __syncthreads();
}

// restore a previous board and try the next remaining guess

__device__ void restore_checkpoint(BlockState& s, int depth) {
    for (int idx = static_cast<int>(threadIdx.x); idx < kCells; idx += static_cast<int>(blockDim.x)) {
        s.board[idx] = s.stack_board[depth][idx];
    }

    __syncthreads();
}

__device__ bool backtrack(BlockState& s) {
    while (true) {
        if (threadIdx.x == 0) {
            if (s.depth == 0) {
                s.branch_cell = -1;
                s.branch_mask = 0u;
                s.next_guess = 0;
            } else {
                const int checkpoint = s.depth - 1;
                s.depth = checkpoint;
                s.branch_cell = s.stack_guess_cell[checkpoint];
                s.branch_mask = s.stack_remaining_mask[checkpoint];
                s.next_guess = 0;
            }
        }

        __syncthreads();

        if (s.depth == 0 && s.branch_cell == -1) {
            return false;
        }

        restore_checkpoint(s, s.depth);

        if (threadIdx.x == 0) {
            if (s.branch_mask != 0u) {
                const int next_digit = digit_from_mask(s.branch_mask);
                const unsigned int next_bit = digit_bit(next_digit);
                s.next_guess = next_digit;
                s.stack_remaining_mask[s.depth] = s.branch_mask & ~next_bit;
                s.depth += 1;
            }
        }

        __syncthreads();

        if (s.next_guess == 0) {
            continue;
        }

        if (threadIdx.x < kCells && static_cast<int>(threadIdx.x) == s.branch_cell) {
            s.board[threadIdx.x] = s.next_guess;
        }

        __syncthreads();
        return true;
    }
}

// one block owns one board

__global__ void sudoku_kernel(
    const int* in_boards,
    int* out_boards,
    int* out_status,
    int batch_size,
    int max_depth
) {
    const int board_id = static_cast<int>(blockIdx.x);
    if (board_id >= batch_size) {
        return;
    }

    __shared__ BlockState s;

    const int tid = static_cast<int>(threadIdx.x);
    const int base = board_id * kCells;

    if (tid < kCells) {
        s.board[tid] = in_boards[base + tid];
    }

    if (tid == 0) {
        s.final_status = kNoSolution;
        s.depth = 0;
        s.hit_depth_limit = 0;
        s.branch_cell = -1;
        s.branch_mask = 0u;
        s.next_guess = 0;
    }

    __syncthreads();

    while (true) {
        const bool ok = propagate_to_fixpoint(s);

        if (!ok) {
            if (s.depth == 0) {
                if (tid == 0) {
                    s.final_status = kInvalidBoard;
                }
                break;
            }

            if (!backtrack(s)) {
                if (tid == 0) {
                    s.final_status = (s.hit_depth_limit != 0) ? kDepthLimit : kNoSolution;
                }
                break;
            }

            continue;
        }

        if (board_is_complete(s)) {
            if (tid == 0) {
                s.final_status = kSolved;
            }
            break;
        }

        choose_branch_cell(s);

        if (s.branch_cell < 0) {
            if (tid == 0) {
                s.final_status = kNoSolution;
            }
            break;
        }

        if (s.depth >= max_depth) {
            if (tid == 0) {
                s.hit_depth_limit = 1;
            }

            __syncthreads();

            if (!backtrack(s)) {
                if (tid == 0) {
                    s.final_status = kDepthLimit;
                }
                break;
            }

            continue;
        }

        save_checkpoint(s, s.depth);

        if (tid == 0) {
            const int guess_digit = digit_from_mask(s.branch_mask);
            const unsigned int guess_bit = digit_bit(guess_digit);
            s.stack_guess_cell[s.depth] = s.branch_cell;
            s.stack_remaining_mask[s.depth] = s.branch_mask & ~guess_bit;
            s.next_guess = guess_digit;
            s.depth += 1;
        }

        __syncthreads();

        if (tid < kCells && tid == s.branch_cell) {
            s.board[tid] = s.next_guess;
        }

        __syncthreads();
    }

    for (int idx = tid; idx < kCells; idx += static_cast<int>(blockDim.x)) {
        out_boards[base + idx] = s.board[idx];
    }

    if (tid == 0) {
        out_status[board_id] = s.final_status;
    }
}

int normalized_threads_per_block(const LaunchConfig& config) {
    int threads = config.threads_per_block;

    if (threads <= 0) {
        threads = kDefaultThreadsPerBlock;
    }
    if (threads < kCells) {
        threads = kCells;
    }
    if (threads > kDefaultThreadsPerBlock) {
        threads = kDefaultThreadsPerBlock;
    }

    return threads;
}

int normalized_depth_limit(const LaunchConfig& config) {
    int depth = config.max_search_depth;

    if (depth <= 0) {
        depth = 1;
    }
    if (depth > kMaxBlockStackDepth) {
        depth = kMaxBlockStackDepth;
    }

    return depth;
}

}  // namespace

const char* status_to_string(int status) {
    switch (status) {
        case kSolved:
            return "solved";
        case kNoSolution:
            return "no_solution";
        case kInvalidBoard:
            return "invalid_board";
        case kDepthLimit:
            return "depth_limit";
        case kCudaError:
            return "cuda_error";
        default:
            return "unknown";
    }
}

cudaError_t launch_sudoku_kernel(
    const int* d_in_boards,
    int* d_out_boards,
    int* d_status,
    int batch_size,
    const LaunchConfig& config,
    cudaStream_t stream
) {
    if (batch_size < 0 || d_in_boards == nullptr || d_out_boards == nullptr || d_status == nullptr) {
        return cudaErrorInvalidValue;
    }

    if (batch_size == 0) {
        return cudaSuccess;
    }

    const int threads = normalized_threads_per_block(config);
    const int depth = normalized_depth_limit(config);

    sudoku_kernel<<<batch_size, threads, 0, stream>>>(d_in_boards, d_out_boards, d_status, batch_size, depth);
    return cudaGetLastError();
}

// host side wrapper for copy launch and timing

cudaError_t solve_batch(
    const int* h_in_boards,
    int* h_out_boards,
    int* h_status,
    int batch_size,
    const LaunchConfig& config,
    float* elapsed_ms
) {
    if (batch_size < 0 || h_in_boards == nullptr || h_out_boards == nullptr || h_status == nullptr) {
        return cudaErrorInvalidValue;
    }

    if (batch_size == 0) {
        if (elapsed_ms != nullptr) {
            *elapsed_ms = 0.0f;
        }
        return cudaSuccess;
    }

    cudaError_t err = cudaSuccess;
    const std::size_t bytes = static_cast<std::size_t>(batch_size) * kCells * sizeof(int);
    const std::size_t status_bytes = static_cast<std::size_t>(batch_size) * sizeof(int);

    int* d_in_boards = nullptr;
    int* d_out_boards = nullptr;
    int* d_status = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

#define CHECK_CUDA(call)          \
    do {                          \
        err = (call);             \
        if (err != cudaSuccess) { \
            goto cleanup;         \
        }                         \
    } while (0)

    CHECK_CUDA(cudaMalloc(&d_in_boards, bytes));
    CHECK_CUDA(cudaMalloc(&d_out_boards, bytes));
    CHECK_CUDA(cudaMalloc(&d_status, status_bytes));

    CHECK_CUDA(cudaMemcpy(d_in_boards, h_in_boards, bytes, cudaMemcpyHostToDevice));

    if (elapsed_ms != nullptr || config.enable_timing) {
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));
    }

    CHECK_CUDA(launch_sudoku_kernel(d_in_boards, d_out_boards, d_status, batch_size, config));

    if (elapsed_ms != nullptr || config.enable_timing) {
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        if (elapsed_ms != nullptr) {
            CHECK_CUDA(cudaEventElapsedTime(elapsed_ms, start, stop));
        }
    } else {
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    CHECK_CUDA(cudaMemcpy(h_out_boards, d_out_boards, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_status, d_status, status_bytes, cudaMemcpyDeviceToHost));

cleanup:
    if (start != nullptr) {
        cudaEventDestroy(start);
    }
    if (stop != nullptr) {
        cudaEventDestroy(stop);
    }
    if (d_in_boards != nullptr) {
        cudaFree(d_in_boards);
    }
    if (d_out_boards != nullptr) {
        cudaFree(d_out_boards);
    }
    if (d_status != nullptr) {
        cudaFree(d_status);
    }

#undef CHECK_CUDA

    return err;
}

}  // namespace sudoku_gpu
