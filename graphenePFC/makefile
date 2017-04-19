CPPFLAGS=-std=c++11
LIBS=-fopenmp -lfftw3_omp -lfftw3 -lm -lboost_system

SRCS=graphene.cpp graphene_functions.cpp postprocessing.cpp analyze_coupled_layers.cpp analyze_coupled_layers_with_burger.cpp
OBJS=graphene.o graphene_functions.o postprocessing.o analyze_coupled_layers.o analyze_coupled_layers_with_burger.o
MATRIX=matrix.hpp matrix_types.hpp

LINK_TARGET=graphene
CC=gcc-6
CXX=g++-6
all : $(LINK_TARGET)

$(LINK_TARGET) : $(OBJS)
	$(CXX) -g $(CPPFLAGS) -o $@ $^ $(LIBS)

%.o : %.cpp
	$(CXX) -c $(CPPFLAGS) -o $@ $< $(LIBS)

analyze_coupled_layers.o: analyze_coupled_layers.cpp $(MATRIX)
analyze_coupled_layers_with_burger.o: analyze_coupled_layers_with_burger.cpp $(MATRIX)
graphene_functions.o: graphene_functions.cpp $(MATRIX)
postprocessing.o: postprocessing.cpp $(MATRIX)
graphene.o: graphene.cpp $(MATRIX)

clean:
	rm *.o $(LINK_TARGET)
