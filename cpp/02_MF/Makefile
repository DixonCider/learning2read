CXX = g++
CXXFLAGS = -Wall -fopenmp -O3
LDLIBS = -lpthread

HEADERS = csv.h
OBJ = Preprocess MF

.PHONY: all
all: $(OBJ)

%: %.cpp $(HEADERS)
	$(CXX) -o $@ $(CXXFLAGS) $(LDLIBS) -o $@ $^

.PHONY: clean
clean:
	rm -f $(OBJ)
