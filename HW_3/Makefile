# === Compiler and Flags ===
CXX := g++
CXXFLAGS := -std=c++17 -Wall -I./src/utils -I./src/perceptron

# === Directories ===
SRC_DIR := src
MAIN_DIR := $(SRC_DIR)/main
UTILS_DIR := $(SRC_DIR)/utils
PERCEPTRON_DIR := $(SRC_DIR)/perceptron
BUILD_DIR := build

# === Files ===
TARGET := main.out

# Add all source files (main + perceptron .cpp files)
SRCS := $(MAIN_DIR)/main.cpp \
        $(wildcard $(PERCEPTRON_DIR)/*.cpp)

# Corresponding object files in build directory
OBJS := $(SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# === Rules ===

# Default rule
all: $(TARGET)

# Link all object files into final executable
$(TARGET): $(OBJS)
	@echo "Linking..."
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile each .cpp file into an object file under build/
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create build directory if missing
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Clean build artifacts
CLEAN_FILES := challenge_data.txt test_data.txt training_data.txt heatmap* *error.txt 

clean:
	@echo "Cleaning..."
	
	rm -rf $(BUILD_DIR) $(TARGET) $(CLEAN_FILES)

# Convenience target to run the program
run: all
	@echo "Running program..."
	./$(TARGET)
