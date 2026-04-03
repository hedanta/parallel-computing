#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <intrin.h>
#include <windows.h>

void sum_sequential(const std::vector<int>& a) {
    size_t N = a.size();
    volatile int S = 0;
    for (size_t i = 0; i < N; i++) {
        S += a[i];
    }
}

void sum_random(
    const std::vector<int>& a, 
    std::mt19937& gen, 
    std::uniform_int_distribution<>& distr
) {
    volatile int S = 0;
    size_t N = a.size();
    for (size_t i = 0; i < N; i++) {
        int ind = distr(gen);
        S += a[ind];
    }
}

void sum_random_prearray(
    const std::vector<int>& a, 
    const std::vector<int>& index_arr,
    std::mt19937& gen, 
    std::uniform_int_distribution<>& distr
) {
    size_t N = a.size();
    volatile int S = 0;
    for (size_t i = 0; i < N; i++) {
        S += a[index_arr[i]];
    }
}

template<typename Func>
double MeasureTime(Func f) {
    unsigned __int64 start_tsc = __rdtsc();
    f();
    unsigned __int64 end_tsc = __rdtsc();
    return end_tsc - start_tsc;
}

struct range {
    size_t start_kb;
    size_t end_kb;
    size_t step;
};

int main() {
    SetThreadAffinityMask(GetCurrentThread(), 1);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

    std::ofstream fout("results.csv");
    fout << "size_kb,seq,random,random_idx\n";
    std::vector<range> ranges = {
        {1, 2048, 1},
        {2048 + 512, 32 * 1024, 512},
        {32 * 1024 + 5 * 1024, 150 * 1024, 5 * 1024}
    };

    std::mt19937 gen(42);

    for (auto r : ranges) {
        for (size_t size_kb = r.start_kb; size_kb <= r.end_kb; size_kb += r.step) {
            size_t n = size_kb * 1024 / sizeof(int);

            std::vector<int> a(n);
            std::vector<int> index_arr(n);
            std::uniform_int_distribution<> distr(0, n-1);

            for (size_t i = 0; i < n; i++) {
                a[i] = i;
                index_arr[i] = distr(gen);
            }
            
            //for (int i = 0; i < 100; i++) sum_sequential(a);

            double t1 = 1e18;
            for (int i = 0; i < 5; i++) {
                double t = MeasureTime([&]() {
                    sum_sequential(a);
                }) / n;
                if (t < t1) t1 = t;
            }

            double t2 = 1e18;
            for (int i = 0; i < 5; i++) {
                double t = MeasureTime([&]() {
                    sum_random(a, gen, distr);
                }) / n;
                if (t < t2) t2 = t;
            }

            double t3 = 1e18;
            for (int i = 0; i < 5; i++) {
                double t = MeasureTime([&]() {
                    sum_random_prearray(a, index_arr, gen, distr);
                }) / n;
                if (t < t3) t3 = t;
            }

            fout << size_kb << "," << t1 << "," << t2 << "," << t3 << "\n";
            std::cout << size_kb << " KB done\n";
        }

    }
    fout.close();
}