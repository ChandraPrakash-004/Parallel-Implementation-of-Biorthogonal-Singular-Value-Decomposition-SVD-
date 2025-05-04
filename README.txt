How to Compile and Run

Serial Version
Compile: g++ Ser_Biorthogonal_SVD.cpp
run: ./a.out


Parallel Version (OpenMP) 
Case 1: The Number of threads is hardcoded in the code
Compile: g++ -fopenmp Parr_Biorthogonal_SVD.cpp
run: ./a.out

Case 2: Set the number of threads at runtime
Compile : g++ -fopenmp Parr_Biorthogonal_SVD.cpp
Set number of threads (e.g., 32): export OMP_NUM_THREADS=32
run: ./a.out
   

 
 