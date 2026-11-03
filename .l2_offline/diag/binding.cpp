#include <torch/extension.h>


void md_MAIN_BASE_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,
             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid);
void md_MAIN_BASE_set_pdl(int64_t on);

std::vector<int64_t> md_MAIN_BASE_info();



void md_MAIN_L2_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,
             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid);
void md_MAIN_L2_set_pdl(int64_t on);

std::vector<int64_t> md_MAIN_L2_info();

void md_MAIN_stream(torch::Tensor buf, torch::Tensor out, int64_t grid);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("okc_run_BASE", &md_MAIN_BASE_run);
  m.def("okc_info_BASE", &md_MAIN_BASE_info);
  m.def("okc_run_L2", &md_MAIN_L2_run);
  m.def("okc_info_L2", &md_MAIN_L2_info);
  m.def("okc_diag_stream", &md_MAIN_stream);
}
