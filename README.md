## SIMD-Accelerated Monte Carlo Integrator

A fast, hardware-accelerated Monte Carlo numerical integrator written in modern C++. It supports both vectorized and scalar implementations of functions, which allows the user to choose between an easy scalar implementation versus a performant vectorized one.

By bypassing standard sequential execution models, this project implements a completely custom **AVX2-vectorized PRNG.** It leverages hardware resources and a pseudorandom number (PRN) generation scheme that is not state dependent, which makes it vectorizable as well. This performs significantly better than standard Monte Carlo simulations implemented using C++ methods like rand() or mersenne_twister_engine().

## Main Features:

* **Interleaved SIMD PRNG:** Implements a custom, vectorized version of the **Philox** algorithm from scratch. It processes 4 independent streams of randomness simultaneously across 256-bit AVX2 lanes, ensuring zero statistical overlap.
* **IEEE-754 Bit Hacking:** Achieves zero-cost integer-to-float conversions by directly injecting random bits into the mantissa of double-precision floating-point numbers, bypassing slow CPU division instructions.

## Build & Run

**Prerequisites:** * A CPU supporting the `AVX2` instruction set.
* GCC or Clang compiler.

**Compilation:**
To achieve maximum performance, compile with native architecture flags and high optimization:

```bash
g++ -O3 -mavx2 -mfma test.cpp -o test
