#VARIABLE := $(shell cat compile.txt)

# Set the compiler based on the value of the variable

USE_CUDA ?= false
COMPARISON:= true

ifeq ($(strip $(USE_CUDA)), $(strip $(COMPARISON)))
CXX := nvcc
CXXFLAGS = -x cu --expt-extended-lambda --expt-relaxed-constexpr --expt-relaxed-constexpr -DUSE_CUDA=$(USE_CUDA)
TARGET:= parallel_for_gpu.ex
else
CXX := mpicxx 
CXXFLAGS = --std=c++14 
TARGET:= parallel_for_cpu.ex
endif

# Source files
SRC = ParallelFor.cpp

#%.o : %.cpp Makefile
#	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Object files
#OBJ = $(SRC:.cpp=.o)

# Header choice

#.PHONY: print_variable

#print_variable:
#	$(info $(CXX))
#	$(info $(VARIABLE))

.PHONY: all

# Rule to build the target
$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

# Rule to build object files
#%.o: %.cpp
#	$(CXX) $(CXXFLAGS) -DUSE_HEADER1=$(USE_CUDA) -c $< -o $@


# Clean rule
clean:
	rm -f  $(TARGET)
