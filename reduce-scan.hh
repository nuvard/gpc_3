#ifndef REDUCE_SCAN_HH
#define REDUCE_SCAN_HH

#include "linear-algebra.hh"

template <class T>
T fold(const Vector<T>& a) {
    const int n = a.size();
    T sum = 0;
    #pragma omp simd reduction(+:sum)
    for (int i=0; i<n; ++i) { sum += a(i); }
    return sum;
}

template <class T>
T reduce(const Vector<T>& a) {
    const int n = a.size();
    T sum = a(0);
    #pragma omp simd reduction(+:sum)
    for (int i=1; i<n; ++i) { sum += a(i); }
    return sum;
}

template <class T>
void scan_exclusive(Vector<T>& a) {
    const int n = a.size();
    T sum = 0;
    for (int i=0; i<n; ++i) {
        T a_i = a(i);
        a(i) = sum;
        sum += a_i;
    }
}

template <class T>
void scan_inclusive(Vector<T>& a) {
    const int n = a.size();
    T sum = 0;
    for (int i=0; i<n; ++i) {
        sum += a(i);
        a(i) = sum;
    }
}

#endif // vim:filetype=cpp
