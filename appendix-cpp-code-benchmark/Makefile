CXX = g++
CXXFLAGS = -O3 -Wall -Wextra -std=c++11
LDFLAGS = -lm

TARGET = benchmark
SRC = bench.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
