#pragma once
#include <cstdint>
#include <vector>
namespace torch {
struct Tensor {};
template<class R,class... A> auto wrap_pybind_function(R(*f)(A...)) {return f;}
}
struct Module {template<class R,class... A> void def(const char*,R(*)(A...)) {}};
#define PYBIND11_MODULE(name,m) void test_module(Module &m)
