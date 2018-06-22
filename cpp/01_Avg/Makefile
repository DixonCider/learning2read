CXX = g++
CXXFLAGS = -Wall -fopenmp -O3
LDLIBS = -lpthread

HEADERS = simulated_annealing.h csv.h
OBJ = AvgSA PureAvgSub

.PHONY: all
all: $(OBJ)

%: %.cpp $(HEADERS)
	$(CXX) -o $@ $(CXXFLAGS) $(LDLIBS) -o $@ $^

.PHONY: clean
clean:
	rm -f $(OBJ)
