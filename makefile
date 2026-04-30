CUDA_PATH ?= /usr/local/cuda
NVCC ?= $(CUDA_PATH)/bin/nvcc
CXX ?= g++

BUILD_DIR := build
BIN_DIR := bin

NVCCFLAGS := -O3 -std=c++17 -lineinfo -Isrc
CXXFLAGS := -O3 -std=c++17 -Isrc

ifneq ($(strip $(CUDA_ARCH)),)
NVCCFLAGS += -arch=$(CUDA_ARCH)
endif

CPU_SRCS := src/solver_cpu.cpp src/board_utils.cpp
CPU_OBJ := $(BUILD_DIR)/solver_cpu.o $(BUILD_DIR)/board_utils.o

APP_OBJ := $(BUILD_DIR)/main.o $(BUILD_DIR)/solver_gpu.o $(CPU_OBJ)
TEST_OBJ := $(BUILD_DIR)/test_correctness.o $(BUILD_DIR)/solver_gpu.o

.PHONY: all clean run test profile racecheck resource_usage

all: $(BIN_DIR)/sudoku_solver

$(BIN_DIR)/sudoku_solver: $(APP_OBJ) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(BIN_DIR)/test_correctness: $(TEST_OBJ) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(BUILD_DIR)/main.o: src/main.cu src/solver_gpu.cuh src/solver_cpu.hpp src/board_utils.hpp | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DHAVE_CPU_BASELINE=1 -dc $< -o $@

$(BUILD_DIR)/solver_gpu.o: src/solver_gpu.cu src/solver_gpu.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

$(BUILD_DIR)/test_correctness.o: tests/test_correctness.cpp src/solver_gpu.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

$(BUILD_DIR)/solver_cpu.o: src/solver_cpu.cpp src/solver_cpu.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/board_utils.o: src/board_utils.cpp src/board_utils.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

run: $(BIN_DIR)/sudoku_solver
	./$(BIN_DIR)/sudoku_solver data/easy.txt

test: $(BIN_DIR)/test_correctness
	./$(BIN_DIR)/test_correctness

profile: $(BIN_DIR)/sudoku_solver
	ncu --set full ./$(BIN_DIR)/sudoku_solver data/easy.txt

racecheck: $(BIN_DIR)/test_correctness
	compute-sanitizer --tool racecheck ./$(BIN_DIR)/test_correctness

resource_usage: | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --resource-usage -c src/solver_gpu.cu -o $(BUILD_DIR)/solver_gpu_res.o

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)