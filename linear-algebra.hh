#ifndef LINEAR_ALGEBRA_HH
#define LINEAR_ALGEBRA_HH

#include <chrono>
#include <ostream>
#include <random>
#include <valarray>

template <class T>
class Vector: public std::valarray<T> {
public:
    using std::valarray<T>::valarray;
    using typename std::valarray<T>::value_type;
    inline value_type& operator()(int i) { return this->operator[](i); }
    inline value_type operator()(int i) const { return this->operator[](i); }
    inline friend std::ostream&
    operator<<(std::ostream& out, const Vector& rhs) {
        const int n = rhs.size();
        for (int i=0; i<n; ++i) { out << rhs(i) << ' '; }
        return out;
    }

};

template <class T>
class Matrix: public std::valarray<T> {
private:
    int _nrows = 0, _ncols = 0;
public:
    using std::valarray<T>::valarray;
    using typename std::valarray<T>::value_type;

    inline explicit Matrix(int m, int n):
    std::valarray<T>(m*n), _nrows(m), _ncols(n) {}

    inline value_type&
    operator()(int i, int j) {
        return this->operator[](i*cols() + j);
    }

    inline value_type
    operator()(int i, int j) const {
        return this->operator[](i*cols() + j);
    }

    inline int rows() const { return this->_nrows; }
    inline int cols() const { return this->_ncols; }

    inline friend std::ostream&
    operator<<(std::ostream& out, const Matrix& rhs) {
        const int n1 = rhs.rows(), n2 = rhs.cols();
        for (int i=0; i<n1; ++i) {
            for (int j=0; j<n2; ++j) {
                out << rhs(i,j) << ' ';
            }
            out << '\n';
        }
        return out;
    }

};

std::default_random_engine make_prng() {
    using clock_type = std::chrono::high_resolution_clock;
    std::random_device dev;
    std::default_random_engine prng;
    std::seed_seq seq{{
        clock_type::now().time_since_epoch().count(),
        clock_type::rep(dev())
    }};
    prng.seed(seq);
    return prng;
}

template <class T> Vector<T>
random_vector(int n) {
    auto prng = make_prng();
    Vector<T> v(n);
    std::uniform_real_distribution<T> dist(T{0}, T{1});
    for (int i=0; i<n; ++i) { v(i) = dist(prng); }
    return v;
}

template <class T> Matrix<T>
random_matrix(int n1, int n2) {
    auto prng = make_prng();
    std::uniform_real_distribution<T> dist(T{0}, T{1});
    Matrix<T> m(n1, n2);
    for (int i=0; i<n1; ++i) {
        for (int j=0; j<n1; ++j) {
            m(i,j) = dist(prng);
        }
    }
    return m;
}

template <class T>
void
vector_times_vector(const Vector<T>& a, const Vector<T>& b, Vector<T>& result) {
    const int n = result.size();
    #pragma omp simd
    for (int i=0; i<n; ++i) {
        result(i) = a(i) * b(i);
    }
}

template <class T>
void
matrix_times_vector(const Matrix<T>& a, const Vector<T>& b, Vector<T>& result) {
    const int nrows = a.rows(), ncols = a.cols();
    #pragma omp parallel for
    for (int i=0; i<nrows; ++i) {
        T sum = 0;
        #pragma omp simd reduction(+:sum)
        for (int j=0; j<ncols; ++j) {
            sum += a(i,j)*b(j);
        }
        result(i) = sum;
    }
}

template <class T>
void
matrix_times_matrix(const Matrix<T>& a, const Matrix<T>& b, Matrix<T>& result) {
    const int nrows = a.rows(), ncols = b.cols(), nmiddle = b.rows();
    #pragma omp parallel for collapse(2)
    for (int i=0; i<nrows; ++i) {
        for (int j=0; j<ncols; ++j) {
            T sum = 0;
            #pragma omp simd reduction(+:sum)
            for (int k=0; k<nmiddle; ++k) {
                sum += a(i,k)*b(k,j);
            }
            result(i,j) = sum;
        }
    }
}

template <class T>
void
matrix_transpose(Matrix<T>& a) {
    const int arows = a.rows(), acols = a.cols();
    #pragma omp parallel for
    for (int i=0; i<arows; ++i) {
        for (int j=i+1; j<acols; ++j) {
            std::swap(a(i,j), a(j,i));
        }
    }
}

template <class T> void
verify_vector(const Vector<T>& expected, const Vector<T>& actual, T eps=T{1e-6}) {
    const auto n1 = expected.size(), n2 = actual.size();
    if (n1 != n2) {
        std::stringstream msg;
        msg << "Vector size does not match: expected.size=" << n1 << ", actual.size=" << n2;
        throw std::runtime_error(msg.str());
    }
    std::stringstream msg;
    for (size_t i=0; i<n1; ++i) {
        if (std::abs(expected(i) - actual(i)) > eps) {
            msg << "Bad vector value at " << i << ": "
                << "expected(i)=" << expected(i) << ", actual(i)=" << actual(i) << '\n';
        }
    }
    auto str = msg.str();
    if (!str.empty()) { str.pop_back(); throw std::runtime_error(msg.str()); }
}

template <class T> void
verify_matrix(const Matrix<T>& expected, const Matrix<T>& actual, T eps=T{1e-6}) {
    const auto rows1 = expected.rows(), rows2 = actual.rows();
    const auto cols1 = expected.cols(), cols2 = actual.cols();
    if (rows1 != rows2 || cols1 != cols2) {
        std::stringstream msg;
        msg << "Matrix size does not match: expected.size=" << rows1 << "x" << cols1
            << ", actual.size=" << rows2 << "x" << cols2;
        throw std::runtime_error(msg.str());
    }
    std::stringstream msg;
    for (size_t i=0; i<rows1; ++i) {
        for (size_t j=0; j<cols1; ++j) {
            if (std::abs(expected(i,j) - actual(i,j)) > eps) {
                msg << "Bad matrix value at " << i << "," << j << ": "
                    << "expected(i,j)=" << expected(i,j) << ", actual(i,j)=" << actual(i,j)
                    << '\n';
            }
        }
    }
    auto str = msg.str();
    if (!str.empty()) { str.pop_back(); throw std::runtime_error(msg.str()); }
}

#endif // vim:filetype=cpp
