#ifndef FILTER_HH
#define FILTER_HH

#include <sstream>
#include <stdexcept>
#include <vector>

#include "linear-algebra.hh"

template <class T, class Pred>
void filter(const std::vector<T>& input, std::vector<T>& result, Pred pred) {
    for (const T& x : input) {
        if (pred(x)) { result.push_back(x); }
    }
}

template <class T> void
verify_vector(const std::vector<T>& expected, const std::vector<T>& actual, T eps=T{1e-6}) {
    const auto n1 = expected.size(), n2 = actual.size();
    if (n1 != n2) {
        std::stringstream msg;
        msg << "Vector size does not match: expected.size=" << n1 << ", actual.size=" << n2;
        throw std::runtime_error(msg.str());
    }
    std::stringstream msg;
    for (size_t i=0; i<n1; ++i) {
        if (std::abs(expected[i] - actual[i]) > eps) {
            msg << "Bad vector value at " << i << ": "
                << "expected(i)=" << expected[i] << ", actual(i)=" << actual[i] << '\n';
        }
    }
    auto str = msg.str();
    if (!str.empty()) { str.pop_back(); throw std::runtime_error(msg.str()); }
}

template <class T> std::vector<T>
random_std_vector(int n) {
    auto prng = make_prng();
    std::vector<T> v(n);
    std::uniform_real_distribution<T> dist(T{-1}, T{1});
    for (int i=0; i<n; ++i) { v[i] = dist(prng); }
    return v;
}

#endif // vim:filetype=cpp
