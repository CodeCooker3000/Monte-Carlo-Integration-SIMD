/* quadrature.hpp
     A small header-only library implementing three Newton-Cotes integration rules:
       - Simpson's 1/3 rule   (composite)
       - Simpson's 3/8 rule   (composite)
       - Boole's rule / "Bode's rule" (composite, 5-point Newton-Cotes)
*/

#ifndef QUADRATURE_HPP
#define QUADRATURE_HPP

#include <stdexcept>
#include <cmath>

namespace quadrature {

namespace detail {

inline int round_up_to_multiple(int n, int multiple) {
    if (n < multiple) n = multiple;
    int rem = n % multiple;
    if (rem != 0) n += (multiple - rem);
    return n;
}

inline void validate_inputs(double a, double b, double (*f)(double)) {
    if (f == nullptr) {
        throw std::invalid_argument("quadrature: function pointer is null");
    }
    if (!std::isfinite(a) || !std::isfinite(b)) {
        throw std::invalid_argument("quadrature: bounds must be finite");
    }
}

} 

inline double simpsons(double a, double b, double (*f)(double), int n = 100) {
    detail::validate_inputs(a, b, f);
    if (a == b) return 0.0;

    bool negate = false;
    if (a > b) { std::swap(a, b); negate = true; }

    n = detail::round_up_to_multiple(n, 2);

    const double h = (b - a) / n;
    double sum = f(a) + f(b);

    for (int i = 1; i < n; ++i) {
        double x = a + i * h;
        sum += (i % 2 == 0 ? 2.0 : 4.0) * f(x);
    }

    double result = (h / 3.0) * sum;
    return negate ? -result : result;
}

inline double simpsons38(double a, double b, double (*f)(double), int n = 99) {
    detail::validate_inputs(a, b, f);
    if (a == b) return 0.0;

    bool negate = false;
    if (a > b) { std::swap(a, b); negate = true; }

    n = detail::round_up_to_multiple(n, 3);

    const double h = (b - a) / n;
    double sum = f(a) + f(b);

    for (int i = 1; i < n; ++i) {
        double x = a + i * h;
        sum += (i % 3 == 0 ? 2.0 : 3.0) * f(x);
    }

    double result = (3.0 * h / 8.0) * sum;
    return negate ? -result : result;
}

inline double boole(double a, double b, double (*f)(double), int n = 100) {
    detail::validate_inputs(a, b, f);
    if (a == b) return 0.0;

    bool negate = false;
    if (a > b) { std::swap(a, b); negate = true; }

    n = detail::round_up_to_multiple(n, 4);

    const double h = (b - a) / n;
    double sum = 7.0 * (f(a) + f(b));

    for (int i = 1; i < n; ++i) {
        double x = a + i * h;
        int pos = i % 4;
        double w;
        if (pos == 0)      w = 14.0; 
        else if (pos == 2) w = 12.0;
        else               w = 32.0; 
        sum += w * f(x);
    }

    double result = (2.0 * h / 45.0) * sum;
    return negate ? -result : result;
}

// Since Bode and Boole refer to the same method, calling same functionality through both names
inline double bode(double a, double b, double (*f)(double), int n = 100) {
    return boole(a, b, f, n);
}

}

#endif // QUADRATURE_HPP
