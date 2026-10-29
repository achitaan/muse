#include <pybind11/pybind11.h>
#include <cstdint>
int64_t cb_launch(int family, int fmt, int query, int m,
    int64_t wp, int64_t x, int64_t branch, int64_t esc,
    int64_t stream, int64_t alpha, int64_t beta, int64_t gain,
    int64_t new_stream, int64_t output, int64_t router,
    int64_t t01, int64_t u01, int64_t cuda_stream, float post_eps, float pre_eps);
