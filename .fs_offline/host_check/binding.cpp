#include <torch/extension.h>


void fsdiag_tiny_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,
             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid);
void fsdiag_tiny_pdl(int64_t on);

std::vector<int64_t> fsdiag_tiny_info();



void fsdiag_tiny2_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,
             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid);
void fsdiag_tiny2_pdl(int64_t on);

std::vector<int64_t> fsdiag_tiny2_info();



void fs_run(torch::Tensor packed, std::vector<torch::Tensor> aux, std::vector<torch::Tensor> scratch,
            torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t grid);
void fs_set_pdl(bool x);
void okc_set_pdl(int64_t x);
std::vector<int64_t> fs_info();

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
m.def("TINY_run", &fsdiag_tiny_run);
m.def("TINY_info", &fsdiag_tiny_info);
m.def("TINY2_run", &fsdiag_tiny2_run);
m.def("TINY2_info", &fsdiag_tiny2_info);
m.def("fs_run", &fs_run);
m.def("fs_info", &fs_info);
m.def("okc_set_pdl", &okc_set_pdl);
m.def("fs_set_pdl", &fs_set_pdl);
}
