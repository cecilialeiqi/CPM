CXX ?= g++
CFLAGS = -O3  
#CFLAGS = -g  
LIBS = ./lib/
dense:
	$(CXX) $(CFLAGS) -I $(LIBS) -lgomp  dense_sort.cpp  -o dense
	
