main: main.cpp
	g++ main.cpp -O3

test_main: main.cpp
	g++ main.cpp -DDEBUGGING

main_mpi: main_mpi.cpp
	mpicxx main_mpi.cpp -O3

test_main_mpi: main_mpi.cpp
	mpicxx main_mpi.cpp -DDEBUGGING

main_mpi_open_mp: main_mpi.cpp
	mpicxx main_mpi.cpp -fopenmp -O3 

test_main_mpi_open_mp: main_mpi.cpp
	mpicxx main_mpi.cpp -fopenmp -DDEBUGGING 
