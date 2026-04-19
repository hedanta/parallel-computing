#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <xmmintrin.h>
#include <emmintrin.h>

template<typename Func>
double MeasureTime(Func f, int reps = 7) {
    double best = 1e18;
    for (int r = 0; r < reps; r++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        f();
        auto t1 = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        if (dt < best) {
            best = dt;
        }
    }
    return best;
}

float scalar(const int8_t* a, int n) {
    int64_t s = 0;
    for (int i = 0; i < n; i++) {
        int v = a[i];
        s += v * v;
    }
    return std::sqrt((float)s);
}

float sse_scalar_u1(const int8_t* a, int n) {
    __m128 sum = _mm_setzero_ps();

    for (int i = 0; i < n; i++) {
        float v = (float)a[i];

        __m128 x = _mm_set_ss(v);
        __m128 sq = _mm_mul_ss(x, x);
        sum = _mm_add_ss(sum, sq);
    }

    return _mm_cvtss_f32(_mm_sqrt_ss(sum));
}

float sse_scalar_u2(const int8_t* a, int n) {
    __m128 s0 = _mm_setzero_ps();
    __m128 s1 = _mm_setzero_ps();

    for (int i = 0; i < n; i += 2) {
        float v0 = (float)a[i];
        float v1 = (float)a[i + 1];

        __m128 x0 = _mm_set_ss(v0);
        __m128 x1 = _mm_set_ss(v1);

        s0 = _mm_add_ss(s0, _mm_mul_ss(x0, x0));
        s1 = _mm_add_ss(s1, _mm_mul_ss(x1, x1));
    }

    float sum =
        _mm_cvtss_f32(s0) +
        _mm_cvtss_f32(s1);

    __m128 x = _mm_set_ss(sum);

    return _mm_cvtss_f32(_mm_sqrt_ss(x));
}

float sse_scalar_u4(const int8_t* a, int n) {
    __m128 s[4] = {
        _mm_setzero_ps(),
        _mm_setzero_ps(),
        _mm_setzero_ps(),
        _mm_setzero_ps()
    };

    for (int i = 0; i < n; i += 4) {
        for (int k = 0; k < 4; k++) {
            float v = (float)a[i + k];
            __m128 x = _mm_set_ss(v);
            s[k] = _mm_add_ss(s[k], _mm_mul_ss(x, x));
        }
    }

    float sum = 0;
    for (int k = 0; k < 4; k++) {
        sum += _mm_cvtss_f32(s[k]);
    }

    __m128 x = _mm_set_ss(sum);

    return _mm_cvtss_f32(_mm_sqrt_ss(x));
}

float sse_scalar_u8(const int8_t* a, int n) {
    __m128 s[8] = {
        _mm_setzero_ps(), _mm_setzero_ps(),
        _mm_setzero_ps(), _mm_setzero_ps(),
        _mm_setzero_ps(), _mm_setzero_ps(),
        _mm_setzero_ps(), _mm_setzero_ps()
    };

    for (int i = 0; i < n; i += 8) {
        for (int k = 0; k < 8; k++) {
            float v = (float)a[i + k];

            __m128 x = _mm_set_ss(v);
            s[k] = _mm_add_ss(s[k], _mm_mul_ss(x, x));
        }
    }

    float sum = 0;
    for (int k = 0; k < 8; k++) {
        sum += _mm_cvtss_f32(s[k]);
    }

    __m128 x = _mm_set_ss(sum);

    return _mm_cvtss_f32(_mm_sqrt_ss(x));
}

float sse_u1(const int8_t* a, int n) {
    __m128 acc = _mm_setzero_ps();

    for (int i = 0; i < n; i += 4) {
        __m128 v = _mm_set_ps(
            (float)a[i+3],
            (float)a[i+2],
            (float)a[i+1],
            (float)a[i]
        );

        acc = _mm_add_ps(acc, _mm_mul_ps(v, v));
    }

    alignas(16) float t[4];
    _mm_store_ps(t, acc);

    float sum = t[0] + t[1] + t[2] + t[3];
    
    return std::sqrt(sum);
}

float sse_u2(const int8_t* a, int n) {
    __m128 acc0 = _mm_setzero_ps();
    __m128 acc1 = _mm_setzero_ps();

    for (int i = 0; i < n; i += 8) {
        __m128 v0 = _mm_set_ps(
            (float)a[i+3], (float)a[i+2],
            (float)a[i+1], (float)a[i]
        );

        __m128 v1 = _mm_set_ps(
            (float)a[i+7], (float)a[i+6],
            (float)a[i+5], (float)a[i+4]
        );

        acc0 = _mm_add_ps(acc0, _mm_mul_ps(v0, v0));
        acc1 = _mm_add_ps(acc1, _mm_mul_ps(v1, v1));
    }

    alignas(16) float t[4];
    float sum = 0;

    _mm_store_ps(t, acc0);
    for (int i = 0; i < 4; i++) {
        sum += t[i];
    }

    _mm_store_ps(t, acc1);
    for (int i = 0; i < 4; i++) {
        sum += t[i];
    }

    return std::sqrt(sum);
}

float sse_u4(const int8_t* a, int n) {
    __m128 acc[4];
    for (int i = 0; i < 4; i++) {
        acc[i] = _mm_setzero_ps();
    }

    for (int i = 0; i < n; i += 16) {
        for (int k = 0; k < 4; k++) {
            int j = i + k * 4;

            __m128 v = _mm_set_ps(
                (float)a[j+3],
                (float)a[j+2],
                (float)a[j+1],
                (float)a[j]
            );

            acc[k] = _mm_add_ps(acc[k], _mm_mul_ps(v, v));
        }
    }

    alignas(16) float t[4];
    float sum = 0;

    for (int k = 0; k < 4; k++) {
        _mm_store_ps(t, acc[k]);
        for (int i = 0; i < 4; i++) {
            sum += t[i];
        }
    }

    return std::sqrt(sum);
}

float sse_u8(const int8_t* a, int n) {
    __m128 acc[8];
    for (int i = 0; i < 8; i++) acc[i] = _mm_setzero_ps();

    for (int i = 0; i < n; i += 32) {
        for (int k = 0; k < 8; k++) {
            int j = i + k * 4;

            __m128 v = _mm_set_ps(
                (float)a[j+3],
                (float)a[j+2],
                (float)a[j+1],
                (float)a[j]
            );

            acc[k] = _mm_add_ps(acc[k], _mm_mul_ps(v, v));
        }
    }

    alignas(16) float t[4];
    float sum = 0;

    for (int k = 0; k < 8; k++) {
        _mm_store_ps(t, acc[k]);
        for (int i = 0; i < 4; i++) {
            sum += t[i];
        }
    }

    return std::sqrt(sum);
}

float sse2_u1(const int8_t* a, int n) {
    __m128i acc = _mm_setzero_si128();

    for (int i = 0; i < n; i += 16) {
        __m128i v = _mm_loadu_si128((const __m128i*)(a + i));

        __m128i zero = _mm_setzero_si128();
        __m128i sign = _mm_cmpgt_epi8(zero, v);

        __m128i lo = _mm_unpacklo_epi8(v, sign);
        __m128i hi = _mm_unpackhi_epi8(v, sign);

        __m128i sq_lo = _mm_madd_epi16(lo, lo);
        __m128i sq_hi = _mm_madd_epi16(hi, hi);

        acc = _mm_add_epi32(acc, _mm_add_epi32(sq_lo, sq_hi));
    }

    alignas(16) int tmp[4];
    _mm_store_si128((__m128i*)tmp, acc);

    int64_t sum = 0;
    for (int i = 0; i < 4; i++) {
        sum += tmp[i];
    }

    return std::sqrt((float)sum);
}

float sse2_u2(const int8_t* a, int n) {
    __m128i acc0 = _mm_setzero_si128();
    __m128i acc1 = _mm_setzero_si128();

    for (int i = 0; i < n; i += 32) {
        {
            __m128i v = _mm_loadu_si128((const __m128i*)(a + i));

            __m128i zero = _mm_setzero_si128();
            __m128i sign = _mm_cmpgt_epi8(zero, v);

            __m128i lo = _mm_unpacklo_epi8(v, sign);
            __m128i hi = _mm_unpackhi_epi8(v, sign);

            __m128i sq_lo = _mm_madd_epi16(lo, lo);
            __m128i sq_hi = _mm_madd_epi16(hi, hi);

            acc0 = _mm_add_epi32(acc0, _mm_add_epi32(sq_lo, sq_hi));
        }

        {
            __m128i v = _mm_loadu_si128((const __m128i*)(a + i + 16));

            __m128i zero = _mm_setzero_si128();
            __m128i sign = _mm_cmpgt_epi8(zero, v);

            __m128i lo = _mm_unpacklo_epi8(v, sign);
            __m128i hi = _mm_unpackhi_epi8(v, sign);

            __m128i sq_lo = _mm_madd_epi16(lo, lo);
            __m128i sq_hi = _mm_madd_epi16(hi, hi);

            acc1 = _mm_add_epi32(acc1, _mm_add_epi32(sq_lo, sq_hi));
        }
    }

    alignas(16) int t0[4], t1[4];
    _mm_store_si128((__m128i*)t0, acc0);
    _mm_store_si128((__m128i*)t1, acc1);

    int64_t sum = 0;
    for (int i = 0; i < 4; i++) {
        sum += t0[i] + t1[i];
    }

    return std::sqrt((float)sum);
}

float sse2_u4(const int8_t* a, int n) {
    __m128i acc[4];
    for (int i = 0; i < 4; i++) {
        acc[i] = _mm_setzero_si128();
    }

    for (int i = 0; i < n; i += 64) {
        for (int k = 0; k < 4; k++) {
            __m128i v = _mm_loadu_si128((const __m128i*)(a + i + 16 * k));

            __m128i zero = _mm_setzero_si128();
            __m128i sign = _mm_cmpgt_epi8(zero, v);

            __m128i lo = _mm_unpacklo_epi8(v, sign);
            __m128i hi = _mm_unpackhi_epi8(v, sign);

            __m128i sq_lo = _mm_madd_epi16(lo, lo);
            __m128i sq_hi = _mm_madd_epi16(hi, hi);

            acc[k] = _mm_add_epi32(acc[k], _mm_add_epi32(sq_lo, sq_hi));
        }
    }

    alignas(16) int tmp[4];
    int64_t sum = 0;

    for (int k = 0; k < 4; k++) {
        _mm_store_si128((__m128i*)tmp, acc[k]);
        for (int i = 0; i < 4; i++) {
            sum += tmp[i];
        }
    }

    return std::sqrt((float)sum);
}

float sse2_u8(const int8_t* a, int n) {
    __m128i acc[8];
    for (int i = 0; i < 8; i++) {
        acc[i] = _mm_setzero_si128();
    }

    for (int i = 0; i < n; i += 128) {
        for (int k = 0; k < 8; k++) {
            __m128i v = _mm_loadu_si128((const __m128i*)(a + i + 16 * k));

            __m128i zero = _mm_setzero_si128();
            __m128i sign = _mm_cmpgt_epi8(zero, v);

            __m128i lo = _mm_unpacklo_epi8(v, sign);
            __m128i hi = _mm_unpackhi_epi8(v, sign);

            __m128i sq_lo = _mm_madd_epi16(lo, lo);
            __m128i sq_hi = _mm_madd_epi16(hi, hi);

            acc[k] = _mm_add_epi32(acc[k], _mm_add_epi32(sq_lo, sq_hi));
        }
    }

    alignas(16) int tmp[4];
    int64_t sum = 0;

    for (int k = 0; k < 8; k++) {
        _mm_store_si128((__m128i*)tmp, acc[k]);
        for (int i = 0; i < 4; i++)
            sum += tmp[i];
    }

    return std::sqrt((float)sum);
}


float avx512_u1(const int8_t* a, int n) {
    __m512i acc = _mm512_setzero_si512();

    for (int i = 0; i < n; i += 64) {
        __m512i v = _mm512_loadu_si512((const void*)(a + i));

        __m256i v_lo = _mm512_castsi512_si256(v);
        __m256i v_hi = _mm512_extracti64x4_epi64(v, 1);

        __m512i lo = _mm512_cvtepi8_epi16(v_lo);
        __m512i hi = _mm512_cvtepi8_epi16(v_hi);

        __m512i sq_lo = _mm512_madd_epi16(lo, lo);
        __m512i sq_hi = _mm512_madd_epi16(hi, hi);

        acc = _mm512_add_epi32(acc, _mm512_add_epi32(sq_lo, sq_hi));
    }

    alignas(64) int tmp[16];
    _mm512_store_si512(tmp, acc);

    int64_t sum = 0;
    for (int i = 0; i < 16; i++) {
        sum += tmp[i];
    }

    return std::sqrt((float)sum);
}

float avx512_u2(const int8_t* a, int n) {
    __m512i acc0 = _mm512_setzero_si512();
    __m512i acc1 = _mm512_setzero_si512();

    for (int i = 0; i < n; i += 128) {
        {
            __m512i v = _mm512_loadu_si512((const void*)(a + i));

            __m256i v_lo = _mm512_castsi512_si256(v);
            __m256i v_hi = _mm512_extracti64x4_epi64(v, 1);

            __m512i lo = _mm512_cvtepi8_epi16(v_lo);
            __m512i hi = _mm512_cvtepi8_epi16(v_hi);

            __m512i sq_lo = _mm512_madd_epi16(lo, lo);
            __m512i sq_hi = _mm512_madd_epi16(hi, hi);

            acc0 = _mm512_add_epi32(acc0, _mm512_add_epi32(sq_lo, sq_hi));
        }

        {
            __m512i v = _mm512_loadu_si512((const void*)(a + i + 64));

            __m256i v_lo = _mm512_castsi512_si256(v);
            __m256i v_hi = _mm512_extracti64x4_epi64(v, 1);

            __m512i lo = _mm512_cvtepi8_epi16(v_lo);
            __m512i hi = _mm512_cvtepi8_epi16(v_hi);

            __m512i sq_lo = _mm512_madd_epi16(lo, lo);
            __m512i sq_hi = _mm512_madd_epi16(hi, hi);

            acc1 = _mm512_add_epi32(acc1, _mm512_add_epi32(sq_lo, sq_hi));
        }
    }

    alignas(64) int tmp[16];
    int64_t sum = 0;

    _mm512_store_si512(tmp, acc0);
    for (int i = 0; i < 16; i++) {
        sum += tmp[i];
    }

    _mm512_store_si512(tmp, acc1);
    for (int i = 0; i < 16; i++) {
        sum += tmp[i];
    }

    return std::sqrt((float)sum);
}

float avx512_u4(const int8_t* a, int n) {
    __m512i acc[4];
    for (int i = 0; i < 4; i++) acc[i] = _mm512_setzero_si512();

    for (int i = 0; i < n; i += 256) {
        for (int k = 0; k < 4; k++) {
            __m512i v = _mm512_loadu_si512((const void*)(a + i + 64 * k));

            __m256i v_lo = _mm512_castsi512_si256(v);
            __m256i v_hi = _mm512_extracti64x4_epi64(v, 1);

            __m512i lo = _mm512_cvtepi8_epi16(v_lo);
            __m512i hi = _mm512_cvtepi8_epi16(v_hi);

            __m512i sq_lo = _mm512_madd_epi16(lo, lo);
            __m512i sq_hi = _mm512_madd_epi16(hi, hi);

            acc[k] = _mm512_add_epi32(acc[k], _mm512_add_epi32(sq_lo, sq_hi));
        }
    }

    alignas(64) int tmp[16];
    int64_t sum = 0;

    for (int k = 0; k < 4; k++) {
        _mm512_store_si512(tmp, acc[k]);
        for (int i = 0; i < 16; i++) {
            sum += tmp[i];
        }
    }

    return std::sqrt((float)sum);
}

float avx512_u8(const int8_t* a, int n) {
    __m512i acc[8];
    for (int i = 0; i < 8; i++) {
        acc[i] = _mm512_setzero_si512();
    }

    for (int i = 0; i < n; i += 512) {
        for (int k = 0; k < 8; k++) {
            __m512i v = _mm512_loadu_si512((const void*)(a + i + 64 * k));

            __m256i v_lo = _mm512_castsi512_si256(v);
            __m256i v_hi = _mm512_extracti64x4_epi64(v, 1);

            __m512i lo = _mm512_cvtepi8_epi16(v_lo);
            __m512i hi = _mm512_cvtepi8_epi16(v_hi);

            __m512i sq_lo = _mm512_madd_epi16(lo, lo);
            __m512i sq_hi = _mm512_madd_epi16(hi, hi);

            acc[k] = _mm512_add_epi32(acc[k], _mm512_add_epi32(sq_lo, sq_hi));
        }
    }

    alignas(64) int tmp[16];
    int64_t sum = 0;

    for (int k = 0; k < 8; k++) {
        _mm512_store_si512(tmp, acc[k]);
        for (int i = 0; i < 16; i++) {
            sum += tmp[i];
        }
    }

    return std::sqrt((float)sum);
}


struct Result {
    const char* name;
    float value;
    double ms;
};

void check(const char* name, float val, float ref) {
    float err = std::abs(val - ref) / (ref + 1e-6f);
    if (err > 1e-3f) {
        std::cerr << "[fail] " << name << " val=" << val << " ref=" << ref << "\n";
    }
}

template<typename F>
Result run(const char* name, F fn, const int8_t* a, int n, float ref) {
    float val = 0;
    double t = MeasureTime([&] {
        val = fn(a, n);
    });

    return {name, val, t * 1000.0};
}

int main() {
    const int N_RAW = 1000000;
    const int N = ((N_RAW + 255) / 256) * 256;

    std::vector<int8_t> a(N);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-128, 127);

    for (int i = 0; i < N_RAW; i++) {
        a[i] = (int8_t)dist(rng);
    }

    for (int i = N_RAW; i < N; i++) {
        a[i] = 0;
    }

    float ref = scalar(a.data(), N);

    std::vector<Result> results;

    auto add = [&](const char* name, auto fn) {
        float val = fn(a.data(), N);
        check(name, val, ref);

        double t = MeasureTime([&] {
            volatile float sink = fn(a.data(), N);
            (void)sink;
        });

        results.push_back({name, val, t * 1000.0});
    };

    add("scalar", scalar);

    add("sse_scalar_u1", sse_scalar_u1);
    add("sse_scalar_u2", sse_scalar_u2);
    add("sse_scalar_u4", sse_scalar_u4);
    add("sse_scalar_u8", sse_scalar_u8);

    add("sse_u1", sse_u1);
    add("sse_u2", sse_u2);
    add("sse_u4", sse_u4);
    add("sse_u8", sse_u8);

    add("sse2_u1", sse2_u1);
    add("sse2_u2", sse2_u2);
    add("sse2_u4", sse2_u4);
    add("sse2_u8", sse2_u8);

    add("avx512_u1", avx512_u1);
    add("avx512_u2", avx512_u2);
    add("avx512_u4", avx512_u4);
    add("avx512_u8", avx512_u8);

    std::cout << "N = " << N_RAW << " (padded " << N << ")\n";
    std::cout << "Reference: " << ref << "\n";

    std::cout << std::left
            << std::setw(18) << "method"
            << std::setw(14) << "result"
            << std::setw(10) << "time_ms"
            << "\n";

    std::cout << "------------------ -------------- ----------\n";

    for (auto& r : results) {
        std::cout << std::left
                << std::setw(18) << r.name
                << std::setw(14) << std::fixed << std::setprecision(2) << r.value
                << std::setw(10) << std::fixed << std::setprecision(4) << r.ms
                << "\n";
    }
}