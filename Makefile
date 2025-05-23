CXX = g++
CXXFLAGS = -Wall -Iinclude

SRC = src/main.cpp
OBJ = $(SRC:.cpp=.o)
TARGET = build/main

all: $(TARGET)

$(TARGET): $(OBJ)
	mkdir -p build
	$(CXX) $(OBJ) -o $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf build $(OBJ)

.PHONY: all clean
