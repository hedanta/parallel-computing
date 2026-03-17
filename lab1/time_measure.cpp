#include <iostream>
#include <vector>
#include <random>
#include <windows.h>
#include <intrin.h>
#include <fstream>
#include <cmath>

std::vector<std::vector<int>> graph;
std::vector<bool> used;

void dfs(int v) {
    used[v] = true;
    for (int to : graph[v]) {
        if (!used[to])
            dfs(to);
    }
}

void generate_graph(int n, int num_edges) {
    graph.clear();
    graph.resize(n);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, n - 1);

    for (int i = 0; i < n; i += 1) {
        for (int j = 0; j < num_edges; j += 1) {
            int to = dis(gen);
            graph[i].push_back(to);
        }
    }
}

double avg(const std::vector<double>& v) {
    double s = 0;
    for (double x : v) {
        s += x;
    }
    return s / v.size();
}

double variance(const std::vector<double>& v, double mean) {
    double s = 0;
    for (double x : v) {
        s += (x - mean) * (x - mean);
    }
    return s / v.size();
}

int main() {
    SetThreadAffinityMask(GetCurrentThread(), 1);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

    int N = 10000;
    int edges = 10;
    int K = 5000;

    //std::cout << "K: ";
    //std::cin >> K;

    std::ofstream fout("results.txt");

    __int64 freq;
    QueryPerformanceFrequency((LARGE_INTEGER *) &freq);

    unsigned __int64 start = __rdtsc();
    Sleep(1000);
    unsigned __int64 end = __rdtsc();
    double cpu_freq = (end - start);

    for (int i = 0; i < 1000; i += 1) {
        generate_graph(N, edges);
        used.assign(N, false);
        dfs(0);
    }

    for (int i = 0; i < K; i += 1) {
        used.assign(N, false);

        // GetTickCount
        DWORD start_tick = GetTickCount();
        LARGE_INTEGER start, end;

        for (int r = 0; r < 500; r += 1) {
            used.assign(N, false);
            dfs(0);
        }

        DWORD end_tick = GetTickCount();
        double tick_time = end_tick - start_tick;
        tick_time /= 500.0;

        // QueryPerformanceCounter
        used.assign(N, false);

        __int64 start_perf, end_perf;
        QueryPerformanceCounter((LARGE_INTEGER *) &start_perf);
        dfs(0);
        QueryPerformanceCounter((LARGE_INTEGER *) &end_perf);

        double perf_time = (double) (end_perf - start_perf) / freq * 1000;

        // RDTSC
        used.assign(N, false);

        unsigned __int64 start_tsc = __rdtsc();
        dfs(0);
        unsigned __int64 end_tsc = __rdtsc();

        double tsc_ms = (end_tsc - start_tsc) / cpu_freq * 1000;

        
        std::cout << "Run " << i + 1 << ": "
                  << tick_time << " ms | "
                  << perf_time << " ms | "
                  << tsc_ms << " ms\n";
        
        

        fout << tick_time << " " << perf_time << " " << tsc_ms << "\n";
    }

    fout.close();
}