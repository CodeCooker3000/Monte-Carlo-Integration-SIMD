#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wpsabi"
#endif
#pragma once
#include "PhiloxSIMD.hpp"
#include <iostream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <vector>


/*-------------------------------------------------------------------------------------
Monte Carlo Integration Setup
-------------------------------------------------------------------------------------*/
double mc_integrate_scalar_eval(float a, float b, uint64_t total_samples, float (*f)(float)) {
    // each Philox loop iteration yields 32 samples (8 lanes * 4 words)
    uint64_t iterations = total_samples / 32;
    if (iterations == 0) iterations = 1;

    double acc = 0.0;
    
    // some sort of initial values
    __m256i v_c0 = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0); 
    __m256i v_c1 = _mm256_setzero_si256();
    __m256i v_c2 = _mm256_setzero_si256();
    __m256i v_c3 = _mm256_setzero_si256();

    // Constant keys also with some sort of initial values
    __m256i v_k0 = _mm256_set1_epi32(0x12345678);
    __m256i v_k1 = _mm256_set1_epi32(0xABCDEF90);

    // Scaling factors
    const float range_width = b - a;

    // Allocate an aligned buffer to extract vector floats to scalar space
    alignas(32) float buffer[32];

    for (uint64_t i = 0; i < iterations; ++i) {
        // generate 32 prns using the philox method
        Float8x4 rands = philox4x32_avx2(v_c0, v_c1, v_c2, v_c3, v_k0, v_k1);

        // store them in buffer for scalar extraction
        _mm256_store_ps(&buffer[0],  rands.x);
        _mm256_store_ps(&buffer[8],  rands.y);
        _mm256_store_ps(&buffer[16], rands.z);
        _mm256_store_ps(&buffer[24], rands.w);

        // scale each point to [a, b] and evaluate f(x)
        for (int j = 0; j < 32; ++j) {
            float x = a + buffer[j] * range_width;
            acc += static_cast<double>(f(x));
        }

        // increment counters for the next block of 32
        v_c0 = _mm256_add_epi32(v_c0, _mm256_set1_epi32(8));
    }

    // Expected Value Estimation: F_N = (b - a) * (sum(f(X_i)) / N)
    uint64_t actual_samples = iterations * 32;
    return (static_cast<double>(range_width) * acc) / static_cast<double>(actual_samples);
}

// Vector Version:
double mc_integrate_vector_eval(float a, float b, uint64_t total_samples, __m256 (*f_vec)(__m256)) {
    uint64_t iterations = total_samples / 32;
    if (iterations == 0) iterations = 1;

    // running vector accumulators
    __m256 v_sum0 = _mm256_setzero_ps();
    __m256 v_sum1 = _mm256_setzero_ps();
    __m256 v_sum2 = _mm256_setzero_ps();
    __m256 v_sum3 = _mm256_setzero_ps();

    // some sort of initial values to go off of
    __m256i v_c0 = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i v_c1 = _mm256_setzero_si256();
    __m256i v_c2 = _mm256_setzero_si256();
    __m256i v_c3 = _mm256_setzero_si256();

    __m256i v_k0 = _mm256_set1_epi32(0x12345678);
    __m256i v_k1 = _mm256_set1_epi32(0xABCDEF01);

    // vectorized scaling parameters
    const float range_width = b - a;
    __m256 v_width = _mm256_set1_ps(range_width);
    __m256 v_offset = _mm256_set1_ps(a);

    for (uint64_t i = 0; i < iterations; ++i) {
        // generate 32 uniform random floats in [0, 1)
        Float8x4 rands = philox4x32_avx2(v_c0, v_c1, v_c2, v_c3, v_k0, v_k1);

        // vectorized scaling to real coordinate: x = a + rand * (b - a)
        __m256 x0 = _mm256_add_ps(v_offset, _mm256_mul_ps(rands.x, v_width));
        __m256 x1 = _mm256_add_ps(v_offset, _mm256_mul_ps(rands.y, v_width));
        __m256 x2 = _mm256_add_ps(v_offset, _mm256_mul_ps(rands.z, v_width));
        __m256 x3 = _mm256_add_ps(v_offset, _mm256_mul_ps(rands.w, v_width));

        // vectorized math evaluation: f_vec(x)
        __m256 y0 = f_vec(x0);
        __m256 y1 = f_vec(x1);
        __m256 y2 = f_vec(x2);
        __m256 y3 = f_vec(x3);

        // vectorized accumulation
        v_sum0 = _mm256_add_ps(v_sum0, y0);
        v_sum1 = _mm256_add_ps(v_sum1, y1);
        v_sum2 = _mm256_add_ps(v_sum2, y2);
        v_sum3 = _mm256_add_ps(v_sum3, y3);

        // update counters
        v_c0 = _mm256_add_epi32(v_c0, _mm256_set1_epi32(8));
    }

    // combine the 4 vector accumulators into a single master vector sum
    __m256 v_total_sum = _mm256_add_ps(
        _mm256_add_ps(v_sum0, v_sum1),
        _mm256_add_ps(v_sum2, v_sum3)
    );

    // horizontal reduction of the final 8-lane float register to double
    alignas(32) float final_sums[8];
    _mm256_store_ps(final_sums, v_total_sum);

    double total_accumulated_sum = 0.0;
    for (int i = 0; i < 8; ++i) {
        total_accumulated_sum += static_cast<double>(final_sums[i]);
    }

    uint64_t actual_samples = iterations * 32;
    return (static_cast<double>(range_width) * total_accumulated_sum) / static_cast<double>(actual_samples);
}
