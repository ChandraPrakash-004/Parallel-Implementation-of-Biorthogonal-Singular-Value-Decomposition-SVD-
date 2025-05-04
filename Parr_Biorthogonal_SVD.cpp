#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>
#include <chrono>
#include <omp.h>  

using namespace std;
using namespace chrono;

const double EPS = 1e-10;

typedef vector<vector<double>> Matrix;


Matrix transpose(const Matrix &A) {    // Transpose of a matrix
    int m = A.size(), n = A[0].size();
    Matrix T(n, vector<double>(m));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            T[j][i] = A[i][j];
    return T;
}

// Multiply two matrices
Matrix multiply(const Matrix &A, const Matrix &B) {
    int m = A.size(), n = B[0].size(), p = A[0].size();
    Matrix C(m, vector<double>(n, 0));

    #pragma omp parallel for
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < p; ++k)
                C[i][j] += A[i][k] * B[k][j];
    
    return C;
}

// Apply Givens rotation to matrix A from the right on columns i and j
void applyGivens(Matrix &A, int i, int j, double c, double s) {
    int m = A.size();

    #pragma omp parallel for
    for (int k = 0; k < m; ++k) {
        double temp1 = A[k][i];
        double temp2 = A[k][j];
        A[k][i] = c * temp1 - s * temp2;
        A[k][j] = s * temp1 + c * temp2;
    }
}

// Parallel Biorthogonalization SVD
void biorthogonalSVD(Matrix A, Matrix &U, Matrix &S, Matrix &V) {
    int m = A.size(), n = A[0].size();
    U = A;
    V = Matrix(n, vector<double>(n, 0));
    for (int i = 0; i < n; ++i) V[i][i] = 1;

    double N2 = 0;  
    #pragma omp parallel for reduction(+:N2)
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            N2 += U[i][j] * U[i][j];

    double s = 0;
    bool first = true;

    do {
        s = 0;
        first = false;

        #pragma omp parallel for reduction(+:s)
        for (int i = 0; i < n - 1; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double a = 0, b = 0, d = 0;
                #pragma omp parallel for reduction(+:a, b, d)
                for (int k = 0; k < m; ++k) {
                    a += U[k][i] * U[k][i];
                    b += U[k][j] * U[k][j];
                    d += U[k][i] * U[k][j];
                }

                s += d * d;

                if (fabs(d) < EPS) continue;

                double tau = (b - a) / (2 * d);
                double t = ((tau >= 0) ? 1.0 : -1.0) / (fabs(tau) + sqrt(1 + tau * tau));
                double c = 1.0 / sqrt(1 + t * t);
                double ss = t * c;

                applyGivens(U, i, j, c, ss);
                applyGivens(V, i, j, c, ss);
            }
        }
    } while (sqrt(s) > EPS * N2 || first);

    // Compute singular values and normalize U
    
    vector<double> sigma(n, 0);

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < m; ++k)
            sigma[i] += U[k][i] * U[k][i];
        sigma[i] = sqrt(sigma[i]);
        if (sigma[i] > EPS) {
            #pragma omp parallel for
            for (int k = 0; k < m; ++k)
                U[k][i] /= sigma[i];
        } else {
            #pragma omp parallel for
            for (int k = 0; k < m; ++k)
                U[k][i] = 0;
        }
    }


    S = Matrix(n, vector<double>(n, 0));   // Construct diagonal S
    #pragma omp parallel for
    for (int i = 0; i < n; ++i)
        S[i][i] = sigma[i];
}


void printMatrix(const Matrix &A, const string &name) {
    cout << name << " =\n";
    for (auto &row : A) {
        for (auto &val : row)
            cout << setw(10) << fixed << setprecision(5) << val << " ";
        cout << "\n";
    }
    cout << "\n";
}
// 
vector<vector<double>> generateLargeMatrix(int size) {    // For Generating Random Matrix
    vector<vector<double>> matrix(size, vector<double>(size));
    

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(-10.0, 10.0);
    
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            
            if (i == j) {
            
                matrix[i][j] = (size - i) * (1.0 + 0.1 * dist(gen));
            } else {
                matrix[i][j] = 0.1 * dist(gen) / (1.0 + abs(i - j));
            }
        }
    }
    
    return matrix;
}


int main() {   
//    Matrix A = {
//        {4, 11, 14},
//        {8, 7, -2}
//    };

    auto A = generateLargeMatrix(300);
    int num_threads = 32;
    omp_set_num_threads(num_threads);
    Matrix U, S, V;
    auto start = high_resolution_clock::now();
    biorthogonalSVD(A, U, S, V);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
     cout << "Threads: " << num_threads << ", Execution time: " << duration.count() << " microseconds" << endl;
    
 
// Testing for Multiple Number of threads
//    vector<int> threads = {1,2,4,8,16,32,48}; 
//    auto A = generateLargeMatrix(300);  // Matrix size is fixed
//    for (auto num_threads : threads) {
//        omp_set_num_threads(num_threads);
//        Matrix U, S, V;
//        auto start = high_resolution_clock::now();
//        biorthogonalSVD(A, U, S, V);
//        auto end = high_resolution_clock::now();
//        auto duration = duration_cast<microseconds>(end - start);
//
//        cout << "Threads: " << num_threads << ", Execution time: " << duration.count() << " microseconds" << endl;
//    }

    
//    printMatrix(U, "U");
//    printMatrix(S, "Sigma");
//    printMatrix(V, "V");
//

//    Matrix VT = transpose(V);
//    Matrix US = multiply(U, S);
//    Matrix USVT = multiply(US, VT);
//
//    printMatrix(USVT, "Reconstructed A (U * Sigma * V^T)");

    return 0;
}
