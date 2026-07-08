#pragma once

#include <immintrin.h>
#include <cstdint>
#include <array>

// Constants used in the Philox4x32 algorithm
namespace PhiloxConsts {
    constexpr uint32_t M0 = 0xD2511F53;
    constexpr uint32_t M1 = 0xCD9E8D57;
    constexpr uint32_t W0 = 0x9E3779B9;
    constexpr uint32_t W1 = 0xBB67AE85;
}

// Struct to hold 4 random 32-bit floats
struct Float4 {
    float x, y, z, w;
};

// Struct to hold 8 packed single-precision floats (AVX2 Vector)
struct Float8x4 {
    __m256 x;
    __m256 y;
    __m256 z;
    __m256 w;
};

/*-------------------------------------------------------------------------------------
Implementing the actual Philox4x32 algorithm using vectorization (8 parallel streams)
-------------------------------------------------------------------------------------*/

inline void avx2_multiply_32x32(const __m256i&a, const __m256i& b_const, __m256i& out_lo, __m256i& out_hi){
    __m256i even_mult = _mm256_mul_epu32(a, b_const);
    __m256i a_shift = _mm256_srli_epi64(a, 32);
    __m256i odd_mult = _mm256_mul_epu32(a_shift, b_const);

    // even_mult : [lo0, hi0, lo2, hi2, lo4, hi4, lo6, hi6]
    // odd_mult :  [lo1, hi1, lo3, hi3, lo5, hi5, lo7, hi7]

    // setting the bits of out_lo appropriately
    out_lo = _mm256_blend_epi32(even_mult, _mm256_slli_epi64(odd_mult, 32), 0xAA);
    // setting the bits of out_hi appropriately
    out_hi = _mm256_blend_epi32(_mm256_srli_epi64(even_mult, 32), odd_mult, 0xAA);
}

inline Float8x4 philox4x32_avx2(__m256i v_c0, __m256i v_c1, __m256i v_c2, __m256i v_c3, __m256i v_k0, __m256i v_k1) {
    
    const __m256i m0 = _mm256_set1_epi32(PhiloxConsts::M0);
    const __m256i m1 = _mm256_set1_epi32(PhiloxConsts::M1);
    const __m256i w0 = _mm256_set1_epi32(PhiloxConsts::W0);
    const __m256i w1 = _mm256_set1_epi32(PhiloxConsts::W1);

    for(int r = 0; r < 10; ++r){
        __m256i lo0, lo1, hi0, hi1;

        avx2_multiply_32x32(v_c0, m0, lo0, hi0);
        avx2_multiply_32x32(v_c2, m1, lo1, hi1);

        // Calculating the new values based on the rules of the Philox Algorithm
        __m256i new_c0 = _mm256_xor_si256(_mm256_xor_si256(hi1, v_c1), v_k0);
        __m256i new_c1 = lo1;
        __m256i new_c2 = _mm256_xor_si256(_mm256_xor_si256(hi0, v_c3), v_k1);
        __m256i new_c3 = lo0;
        
        // Updating the new values
        v_c0 = new_c0;
        v_c1 = new_c1;
        v_c2 = new_c2;
        v_c3 = new_c3;

        v_k0 = _mm256_add_epi32(v_k0, w0);
        v_k1 = _mm256_add_epi32(v_k1, w1);
    }

    // Convert 32-bit uint vectors to float vectors in range [0, 1)
    auto convert_to_floats = [](__m256i raw_ints) -> __m256 {
        __m256i mask = _mm256_set1_epi32(0x7FFFFF);
        __m256i ieee_ones = _mm256_set1_epi32(0x3F800000);
        
        // Extract 23 bits of entropy and bitwise OR into the mantissa space of 1.0f
        __m256i mantissa = _mm256_and_si256(raw_ints, mask);
        __m256i float_bits = _mm256_or_si256(mantissa, ieee_ones);
        
        // Cast bits to floats and subtract 1.0f
        __m256 floats = _mm256_castsi256_ps(float_bits);
        return _mm256_sub_ps(floats, _mm256_set1_ps(1.0f));
    };

    return Float8x4 {
        convert_to_floats(v_c0),
        convert_to_floats(v_c1),
        convert_to_floats(v_c2),
        convert_to_floats(v_c3)
    };
}

