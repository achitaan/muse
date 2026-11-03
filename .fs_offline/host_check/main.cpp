
#include <torch/extension.h>
void fs_run(torch::Tensor packed, std::vector<torch::Tensor> aux, std::vector<torch::Tensor> scratch,
            torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t grid);
void fs_set_pdl(bool x);
void okc_set_pdl(int64_t x);
std::vector<int64_t> fs_info();

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
m.def("fs_run", torch::wrap_pybind_function(fs_run));
m.def("fs_set_pdl", torch::wrap_pybind_function(fs_set_pdl));
m.def("okc_set_pdl", torch::wrap_pybind_function(okc_set_pdl));
m.def("fs_info", torch::wrap_pybind_function(fs_info));
}
