#include <iostream>
#include "Quadrature.hpp"
#include "MonteCarloInt.hpp"

/*-------------------------------------------------------------------------------------
Using a scalar or a vectorized version of the function x^2 for example
-------------------------------------------------------------------------------------*/

inline float target_func_scalar(float x){
    return (x * x);
}

inline __m256 target_func_vector(__m256 v_x) {
    return _mm256_mul_ps(v_x, v_x);
}


int main() {
    const float a = 0.0f;
    const float b = 2.0f;
    const uint64_t SAMPLES = 512'000'000; // 512 Million Samples

    std::cout << "      SIMD MONTE CARLO INTEGRATION SIMULATION\n";
    std::cout << "-----------------------------------------------------\n";
    std::cout << "Integrating f(x) = x^2 from " << a << " to " << b << "\n";
    std::cout << "Target Analytical Solution: 8/3 ≈ 2.6666666667\n";
    std::cout << "Sample Count: " << SAMPLES << " (" << SAMPLES / 1000000 << "M points)\n\n";

    // scalar integration
    std::cout << "[1] Running SIMD PRNG with Scalar Evaluation...\n";
    auto start_scalar = std::chrono::high_resolution_clock::now();
    double result_scalar = mc_integrate_scalar_eval(a, b, SAMPLES, target_func_scalar);
    auto end_scalar = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_scalar = end_scalar - start_scalar;

    double error_scalar = std::abs(result_scalar - (8.0 / 3.0));
    std::cout << "\tResult : " << std::fixed << std::setprecision(10) << result_scalar << "\n";
    std::cout << "\tError  : " << error_scalar << "\n";
    std::cout << "\tTime   : " << duration_scalar.count() << " ms\n";
    std::cout << "\tSpeed  : " << (SAMPLES / (duration_scalar.count() / 1000.0)) / 1000000.0 << " M samples/sec\n\n";

    // vectorized Integration
    std::cout << "[2] Running SIMD PRNG with Vectorized Evaluation...\n";
    auto start_vector = std::chrono::high_resolution_clock::now();
    double result_vector = mc_integrate_vector_eval(a, b, SAMPLES, target_func_vector);
    auto end_vector = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_vector = end_vector - start_vector;

    double error_vector = std::abs(result_vector - (8.0 / 3.0));
    std::cout << "\tResult : " << result_vector << "\n";
    std::cout << "\tError  : " << error_vector << "\n";
    std::cout << "\tTime   : " << duration_vector.count() << " ms\n";
    std::cout << "\tSpeed  : " << (SAMPLES / (duration_vector.count() / 1000.0)) / 1000000.0 << " M samples/sec\n\n";

    // comparison 
    double speedup = duration_scalar.count() / duration_vector.count();
    std::cout << "--------------------------------------------------------\n";
    std::cout << "Vectorized Evaluation Speedup: " << std::fixed << std::setprecision(2) << speedup << "x\n";

    return 0;
}
