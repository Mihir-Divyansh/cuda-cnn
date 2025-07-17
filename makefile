# CUDA Project Makefile

# Compiler and flags
NVCC := nvcc
CFLAGS := -O3 -std=c++17
NVCC_FLAGS := -arch=sm_86 --ptxas-options=-v

# Directories
SRC_DIR := src
BUILD_DIR := build
BIN_DIR := bin

# Target executable
TARGET := $(BIN_DIR)/main

# Recursive wildcard function
rwildcard = $(foreach d,$(wildcard $(1:=/*)),$(call rwildcard,$d,$2) $(filter $(subst *,%,$2),$d))

# Find all CUDA and C++ source files recursively
CU_SOURCES := $(call rwildcard,$(SRC_DIR),*.cu)
CPP_SOURCES := $(call rwildcard,$(SRC_DIR),*.cpp)
C_SOURCES := $(call rwildcard,$(SRC_DIR),*.c)

# Generate object file names
CU_OBJECTS := $(CU_SOURCES:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)
CPP_OBJECTS := $(CPP_SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
C_OBJECTS := $(C_SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

ALL_OBJECTS := $(CU_OBJECTS) $(CPP_OBJECTS) $(C_OBJECTS)

# NCU profiler settings
NCU := /usr/local/cuda/bin/ncu
NCU_FLAGS := --set full --force-overwrite
PROFILE_OUTPUT := profile_output

# Default target
.PHONY: all clean profile run dirs

all: dirs $(TARGET)

# Create necessary directories
dirs:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(dir $(ALL_OBJECTS))

# Link target
$(TARGET): $(ALL_OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

# Compile CUDA files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) $(CFLAGS) -c $< -o $@

# Compile C++ files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(NVCC) $(CFLAGS) -c $< -o $@

# Compile C files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(NVCC) $(CFLAGS) -c $< -o $@

# Run the program
run: $(TARGET)
	./$(TARGET)

# Profile with NCU (requires sudo)
profile: $(TARGET)
	sudo $(NCU) $(NCU_FLAGS) --export $(PROFILE_OUTPUT) ./$(TARGET)

# Profile with specific metrics
profile-metrics: $(TARGET)
	sudo $(NCU) --metrics sm__cycles_elapsed.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed ./$(TARGET)

# Profile and save to file
profile-save: $(TARGET)
	sudo $(NCU) $(NCU_FLAGS) --export $(PROFILE_OUTPUT) ./$(TARGET)
	@echo "Profile saved to $(PROFILE_OUTPUT).ncu-rep"

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR) $(PROFILE_OUTPUT).ncu-rep

# Show detected files (for debugging)
show-files:
	@echo "CUDA sources: $(CU_SOURCES)"
	@echo "C++ sources: $(CPP_SOURCES)"
	@echo "C sources: $(C_SOURCES)"
	@echo "All objects: $(ALL_OBJECTS)"

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build the project"
	@echo "  clean        - Remove build artifacts"
	@echo "  run          - Run the executable"
	@echo "  profile      - Profile with NCU (requires sudo)"
	@echo "  profile-metrics - Profile with specific metrics"
	@echo "  profile-save - Profile and save to file"
	@echo "  show-files   - Show detected source files"
	@echo "  help         - Show this help message"

# Dependency tracking
-include $(ALL_OBJECTS:.o=.d)

# Generate dependency files
$(BUILD_DIR)/%.d: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	@$(NVCC) -M $< | sed 's,\($*\)\.o[ :]*,$(BUILD_DIR)/\1.o $@ : ,g' > $@

$(BUILD_DIR)/%.d: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	@$(NVCC) -M $< | sed 's,\($*\)\.o[ :]*,$(BUILD_DIR)/\1.o $@ : ,g' > $@

$(BUILD_DIR)/%.d: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	@$(NVCC) -M $< | sed 's,\($*\)\.o[ :]*,$(BUILD_DIR)/\1.o $@ : ,g' > $@
