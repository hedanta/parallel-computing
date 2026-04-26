#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <immintrin.h>
#include <opencv2/opencv.hpp>

template<typename Func>
double MeasureTime(Func f, int reps = 7) {
    double best = 1e18;
    for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        f();
        auto t1 = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        if (dt < best) best = dt;
    }
    return best;
}

uint8_t* alloc64(std::size_t n) {
    void* p = _mm_malloc(n, 64);
    if (!p) throw std::bad_alloc{};
    return static_cast<uint8_t*>(p);
}
void free64(uint8_t* p) noexcept { _mm_free(p); }

inline uint8_t clip(int v) noexcept {
    return static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
}

void laplace_scalar(const uint8_t* src, uint8_t* dst, int W, int H) {
    std::memset(dst, 0, W * H);
    for (int y = 1; y < H - 1; y++) {
        const uint8_t* r0 = src + (y - 1) * W;
        const uint8_t* r1 = src +  y * W;
        const uint8_t* r2 = src + (y + 1) * W;
        uint8_t* rd = dst + y * W;
        for (int x = 1; x < W - 1; x++) {
            int sum =
                r0[x-1] + r0[x] + r0[x+1] +
                r1[x-1] - 8*r1[x] + r1[x+1] +
                r2[x-1] + r2[x] + r2[x+1];
            rd[x] = clip(std::abs(sum));
        }
    }
}

inline __m512i load32_u8_to_i16(const uint8_t* p) noexcept {
    __m256i v8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    return _mm512_cvtepu8_epi16(v8);
}

void laplace_avx512(const uint8_t* src, uint8_t* dst, int W, int H) {
    std::memset(dst, 0, size_t(W) * H);
    const __m512i v8 = _mm512_set1_epi16(8);
    for (int y = 1; y < H - 1; y++) {
        const uint8_t* r0 = src + (y - 1) * W;
        const uint8_t* r1 = src +  y      * W;
        const uint8_t* r2 = src + (y + 1) * W;
        uint8_t* rd = dst + y * W;
        int x = 1;

        for (; x <= W - 33; x += 32) {
            // y-1
            __m512i A = load32_u8_to_i16(r0 + x - 1);
            __m512i B = load32_u8_to_i16(r0 + x);
            __m512i C = load32_u8_to_i16(r0 + x + 1);
            // y
            __m512i D = load32_u8_to_i16(r1 + x - 1);
            __m512i E = load32_u8_to_i16(r1 + x);
            __m512i F = load32_u8_to_i16(r1 + x + 1);
            // y+1
            __m512i G = load32_u8_to_i16(r2 + x - 1);
            __m512i Hv = load32_u8_to_i16(r2 + x);
            __m512i I = load32_u8_to_i16(r2 + x + 1);

            // sum = A+B+C + (D+F - 8E) + G+H+I
            __m512i row0 = _mm512_add_epi16(_mm512_add_epi16(A, B), C);
            __m512i row2 = _mm512_add_epi16(_mm512_add_epi16(G, Hv), I);
            __m512i mid  = _mm512_sub_epi16(
                               _mm512_add_epi16(D, F),
                               _mm512_mullo_epi16(E, v8));
            __m512i sum  = _mm512_add_epi16(_mm512_add_epi16(row0, row2), mid);

            sum = _mm512_abs_epi16(sum);

            __m512i packed = _mm512_packus_epi16(sum, _mm512_setzero_si512());
            packed = _mm512_permutexvar_epi64(
                         _mm512_set_epi64(7,5,3,1, 6,4,2,0), packed);

            // запись 32 байт
            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(rd + x),
                _mm512_castsi512_si256(packed));
        }

        // скалярная обработка оставшихся пикселей
        for (; x < W - 1; x++) {
            int sum =
                r0[x-1] + r0[x] + r0[x+1] +
                r1[x-1] - 8*r1[x] + r1[x+1] +
                r2[x-1] + r2[x] + r2[x+1];
            rd[x] = clip(std::abs(sum));
        }
    }
}

int verify(const uint8_t* ref, const uint8_t* tst, int N) {
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (std::abs(static_cast<int>(ref[i]) - static_cast<int>(tst[i])) > 1) {
            if (errors < 10)
                std::cout << "Error [" << i << "]: "
                          << "ref=" << +ref[i]
                          << "avx512=" << +tst[i] << '\n';
            errors += 1;
        }
    }
    return errors;
}

int main(int argc, char* argv[]) {
    // .\main.exe cat.png
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    int W = img.cols;
    int H = img.rows;

    std::cout << "Laplace (diag): Scalar vs AVX-512\n\n"
              << "Size: " << W << " x " << H << " px\n\n";

    uint8_t* src      = alloc64(W * H);
    uint8_t* dst_ref  = alloc64(W * H);
    uint8_t* dst_avx  = alloc64(W * H);
    std::memcpy(src, img.data, W * H);

    laplace_scalar (src, dst_ref, W, H);
    laplace_avx512 (src, dst_avx, W, H);

    double t_scalar = MeasureTime([&]{ laplace_scalar (src, dst_ref, W, H); });
    double t_avx512 = MeasureTime([&]{ laplace_avx512 (src, dst_avx, W, H); });

    const int errors = verify(dst_ref, dst_avx, W * H);
    std::cout << "Verify results: "
              << (errors == 0 ? "good" : "bad")
              << " (" << errors << " errors, +-1)\n\n";

    const double mpix = static_cast<double>(W - 2) * (H - 2) / 1e6;
    std::cout << std::fixed << std::setprecision(3)
              << "Time (scalar):  " << t_scalar * 1e3 << " ms\n"
              << "Time (AVX-512): " << t_avx512 * 1e3 << " ms\n"
              << std::setprecision(2)
              << (t_scalar / t_avx512) << "x faster\n\n"
              << std::setprecision(1)
              << "Bandwidth:\n"
              << "Scalar:  " << (mpix / t_scalar) << " Mpx/s\n"
              << "AVX-512: " << (mpix / t_avx512) << " Mpx/s\n";

    std::cout << "\nExample (y=1, x=1..8):\n"
              << "  idx   src  scalar  avx512\n";
    for (int x = 1; x <= 8 && x < W - 1; ++x)
        std::cout << "  [" << x << "]   "
                  << std::setw(3) << +src[W + x] << "     "
                  << std::setw(3) << +dst_ref[W + x] << "     "
                  << std::setw(3) << +dst_avx[W + x] << '\n';

    free64(src);
    free64(dst_ref);

    cv::Mat out(H, W, CV_8U, dst_avx);
    cv::imwrite("result.png", out);
    free64(dst_avx);

    return 0;
}