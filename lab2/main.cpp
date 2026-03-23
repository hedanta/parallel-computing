#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>

using Matrix = std::vector<std::vector<float>>;

Matrix GenMatrix(int N) {
    Matrix A(N, std::vector<float>(N));
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            A[i][j] = dist(gen);
        }
    }
    return A;
}

bool CompareMatrices(const Matrix& A, const Matrix& B, float eps = 1e-3) {
    size_t N = A.size();
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            if (std::abs(A[i][j] - B[i][j]) > eps)
                return false;
        }
    }
    return true;
}

Matrix MulClassic(const Matrix& A, const Matrix& B) {
    size_t N = A.size();
    Matrix C(N, std::vector<float>(N, 0));
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            float s = 0.0f;
            for (size_t k = 0; k < N; k++) {
                s += A[i][k] * B[k][j];
            }
            C[i][j] = s;
        }
    }
    return C;
}


Matrix Transpose(const Matrix& B) {
    size_t N = B.size();
    Matrix BT(N, std::vector<float>(N));
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            BT[j][i] = B[i][j];
        }
    }
    return BT;
}

Matrix MulTranspose(const Matrix& A, const Matrix& B) {
    size_t N = A.size();
    Matrix C(N, std::vector<float>(N, 0));
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            float s = 0.0f;
            for (size_t k = 0; k < N; k++) {
                s += A[i][k] * B[j][k];
            }
            C[i][j] = s;
        }
    }
    return C;
}

Matrix MulBuff(const Matrix& A, const Matrix& B) {
    size_t N = A.size();
    Matrix C(N, std::vector<float>(N, 0));
    std::vector<float> tmp(N);
    for (size_t j = 0; j < N; j++) {
        for (size_t k = 0; k < N; k++) {
            tmp[k] = B[k][j];
        }
        for (size_t i = 0; i < N; i++) {
            float s = 0.0f;
            for (size_t k = 0; k < N; k++) {
                s += A[i][k] * tmp[k];
            }
            C[i][j] = s;
        }
    }
    return C;
}

Matrix MulBuffUnroll(const Matrix& A, const Matrix& B, int M) {
    size_t N = A.size();
    Matrix C(N, std::vector<float>(N, 0));
    std::vector<float> tmp(N);

    for (size_t j = 0; j < N; j++) {
        for (size_t k = 0; k < N; k++) {
            tmp[k] = B[k][j];
        }
        for (size_t i = 0; i < N; i++) {
            float s[16] = {0};
            size_t k = 0;
            for (; k + M <= N; k += M) {
                for (int m = 0; m < M; m++) {
                    s[m] += A[i][k + m] * tmp[k + m];
                }
            }
            float sum = 0;
            for (int m = 0; m < M; m++) { 
                sum += s[m];
            }
            for (; k < N; k++) {
                sum += A[i][k] * tmp[k];
            }
            C[i][j] = sum;
        }
    }
    return C;
}

Matrix MulBlock(const Matrix& A, const Matrix& B, int S) {
    size_t N = A.size();
    Matrix C(N, std::vector<float>(N, 0));
    for (size_t ii = 0; ii < N; ii += S) {
        for (size_t jj = 0; jj < N; jj += S) {
            for (size_t kk = 0; kk < N; kk += S) {

                for (size_t i = ii; i < std::min(ii + S, N); i++) {
                    for (size_t j = jj; j < std::min(jj + S, N); j++) {
                        float s = C[i][j];
                        for (size_t k = kk; k < std::min(kk + S, N); k++) {
                            s += A[i][k] * B[k][j];
                        }
                        C[i][j] = s;
                    }
                }
            }
        }
    }
    return C;
}

Matrix MulBlockUnroll(const Matrix& A, const Matrix& B, int S, int M) {
    size_t N = A.size();
    Matrix C(N, std::vector<float>(N, 0));

    for (size_t ii = 0; ii < N; ii += S) {
        for (size_t jj = 0; jj < N; jj += S) {
            for (size_t kk = 0; kk < N; kk += S) {

                for (size_t i = ii; i < std::min(ii + S, N); i++) {
                    for (size_t j = jj; j < std::min(jj + S, N); j++) {
                        float s[16] = {0};
                        size_t k = kk;

                        size_t kend = std::min(kk + S, N);
                        for (; k + M <= kend; k += M) {
                            for (int m = 0; m < M; m++)
                                s[m] += A[i][k + m] * B[k + m][j];
                        }

                        float sum = C[i][j];
                        for (int m = 0; m < M; m++) {
                            sum += s[m];
                        }
                        for (; k < kend; k++) {
                            sum += A[i][k] * B[k][j];
                        }

                        C[i][j] = sum;
                    }
                }
            }
        }
    }
    return C;
}

template<typename Func>
double MeasureTime(Func f) {
    auto start = std::chrono::high_resolution_clock::now();
    f();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

double CalcGflops(size_t N, double time_sec) {
    double ops = 2.0 * N * N * N;
    return ops / (time_sec * 1e9);
}

int main() {
    std::vector<int> Ns = {128, 256, 512, 1024, 2048};
    std::vector<int> Ms = {1, 2, 4, 8, 16};

    std::ofstream file_all("all_algorithms.txt");
    std::ofstream file_best("best_results.txt");

    file_all << "N algo time GFLOPS\n";
    file_best << "N best_algo time GFLOPS S M\n";

    for (int N : Ns) {
        std::cout << "\nN = " << N << "\n";

        Matrix A = GenMatrix(N);
        Matrix B = GenMatrix(N);

        Matrix C_ref;
        double t_classic = MeasureTime([&]() {
            C_ref = MulClassic(A, B);
        });

        double g_classic = CalcGflops(N, t_classic);
        file_all << N << " classic " << t_classic << " " << g_classic << "\n";

        Matrix BT, C_tr;
        double t_tr_b = MeasureTime([&]() {
            BT = Transpose(B);
        });
        double t_tr = MeasureTime([&]() {
            C_tr = MulTranspose(A, BT);
        });

        if (!CompareMatrices(C_ref, C_tr))
            std::cout << "ERROR: transpose mismatch\n";

        double g_tr = CalcGflops(N, t_tr);
        double t_full = t_tr + t_tr_b;
        double g_full = CalcGflops(N, t_full);
        file_all << N << " transpose " << t_tr << " " << g_tr << "\n";
        file_all << N << " transpose_full " << t_full << " " << g_full << "\n";

        Matrix C_buf;
        double t_buf = MeasureTime([&]() {
            C_buf = MulBuff(A, B);
        });

        if (!CompareMatrices(C_ref, C_buf))
            std::cout << "ERROR: buffer mismatch\n";

        double g_buf = CalcGflops(N, t_buf);
        file_all << N << " buffer " << t_buf << " " << g_buf << "\n";

        // best buffer+unroll
        int bestM = 1;
        double best_buff_time = 1e18;

        for (int M : Ms) {

            double t = MeasureTime([&]() {
                Matrix C = MulBuffUnroll(A, B, M);
            });

            if (t < best_buff_time) {
                best_buff_time = t;
                bestM = M;
            }

            double g = CalcGflops(N, t);
            file_all << N << " buffer_unroll_M" << M << " " << t << " " << g << "\n";
        }

        std::cout << "Best M (buffer) = " << bestM << "\n";

        // best block (s)
        int bestS = 1;
        double best_clock_time = 1e18;

        for (int S = 1; S <= N; S *= 2) {
            Matrix C;
            double t = MeasureTime([&]() {
                C = MulBlock(A, B, S);
            });

            if (!CompareMatrices(C_ref, C))
                std::cout << "ERROR: block mismatch\n";

            if (t < best_clock_time) {
                best_clock_time = t;
                bestS = S;
            }

            double g = CalcGflops(N, t);
            file_all << N << " block_S" << S << " " << t << " " << g << "\n";
        }

        std::cout << "Best S = " << bestS << "\n";

        // best block (s+m)
        int bestSM_S = bestS;
        int bestSM_M = bestM;
        double bestSM_time = 1e18;

        for (int S = 8; S <= 128; S *= 2) {
            for (int M : Ms) {

                double t = MeasureTime([&]() {
                    Matrix C = MulBlockUnroll(A, B, S, M);
                });

                if (t < bestSM_time) {
                    bestSM_time = t;
                    bestSM_S = S;
                    bestSM_M = M;
                }

                double g = CalcGflops(N, t);
                file_all << N << " block_unroll_S" << S << "_M" << M
                         << " " << t << " " << g << "\n";
            }
        }

        std::cout << "Best S+M = " << bestSM_S << ", " << bestSM_M << "\n";

        double best_time = t_classic;
        std::string best_algo = "classic";
        int outS = 0, outM = 0;

        if (t_tr < best_time) {
            best_time = t_tr;
            best_algo = "transpose";
        }

        if (t_full < best_time) {
            best_time = t_full;
            best_algo = "transpose_full";
        }

        if (t_buf < best_time) {
            best_time = t_buf;
            best_algo = "buffer";
        }

        if (best_buff_time < best_time) {
            best_time = best_buff_time;
            best_algo = "buffer_unroll";
            outM = bestM;
        }

        if (best_clock_time < best_time) {
            best_time = best_clock_time;
            best_algo = "block";
            outS = bestS;
        }

        if (bestSM_time < best_time) {
            best_time = bestSM_time;
            best_algo = "block_unroll";
            outS = bestSM_S;
            outM = bestSM_M;
        }

        double best_g = CalcGflops(N, best_time);

        file_best << N << " " << best_algo << " "
                  << best_time << " " << best_g
                  << " " << outS << " " << outM << "\n";

        std::cout << "BEST: " << best_algo
                  << " | GFLOPS = " << best_g << "\n";
    }

    std::cout << "\nAll results saved.\n";
}