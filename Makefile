CXX ?= g++
CFLAGS = -O3  
#CFLAGS = -g  
LIBS = ./lib/
all: 
	$(CXX) $(CFLAGS) -I $(LIBS) -lgomp  dense_sort.cpp  -o dense
	$(CXX) $(CFLAGS) -I $(LIBS) -lgomp sparse.cpp -o sparse
