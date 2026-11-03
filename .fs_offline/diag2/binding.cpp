#include <torch/extension.h>


void fs2_tiny_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,
             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid);
void fs2_tiny_pdl(int64_t on);

std::vector<int64_t> fs2_tiny_info();


void fs2_tiny2_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,
             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid);
void fs2_tiny2_pdl(int64_t on);

std::vector<int64_t> fs2_tiny2_info();


void fs2_fs_fs_run(torch::Tensor packed, std::vector<torch::Tensor> aux, std::vector<torch::Tensor> scratch,
            torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t grid);
void fs2_fs_fs_set_pdl(bool x);
void fs2_fs_okc_set_pdl(int64_t x);
std::vector<int64_t> fs2_fs_fs_info();


void fs2_nob_fs_run(torch::Tensor packed, std::vector<torch::Tensor> aux, std::vector<torch::Tensor> scratch,
            torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t grid);
void fs2_nob_fs_set_pdl(bool x);
void fs2_nob_okc_set_pdl(int64_t x);
std::vector<int64_t> fs2_nob_fs_info();


void fs2_noflush_fs_run(torch::Tensor packed, std::vector<torch::Tensor> aux, std::vector<torch::Tensor> scratch,
            torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t grid);
void fs2_noflush_fs_set_pdl(bool x);
void fs2_noflush_okc_set_pdl(int64_t x);
std::vector<int64_t> fs2_noflush_fs_info();


void fs2_noa_fs_run(torch::Tensor packed, std::vector<torch::Tensor> aux, std::vector<torch::Tensor> scratch,
            torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t grid);
void fs2_noa_fs_set_pdl(bool x);
void fs2_noa_okc_set_pdl(int64_t x);
std::vector<int64_t> fs2_noa_fs_info();


void fs2_noacc_fs_run(torch::Tensor packed, std::vector<torch::Tensor> aux, std::vector<torch::Tensor> scratch,
            torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t grid);
void fs2_noacc_fs_set_pdl(bool x);
void fs2_noacc_okc_set_pdl(int64_t x);
std::vector<int64_t> fs2_noacc_fs_info();


void fs2_tl_fs_run(torch::Tensor packed, std::vector<torch::Tensor> aux, std::vector<torch::Tensor> scratch,
            torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t grid);
void fs2_tl_fs_set_pdl(bool x);
void fs2_tl_okc_set_pdl(int64_t x);
std::vector<int64_t> fs2_tl_fs_info();

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
m.def("TINY_run", &fs2_tiny_run);
m.def("TINY_info", &fs2_tiny_info);
m.def("TINY2_run", &fs2_tiny2_run);
m.def("TINY2_info", &fs2_tiny2_info);
m.def("FS_run", &fs2_fs_fs_run);
m.def("FS_info", &fs2_fs_fs_info);
m.def("NOB_run", &fs2_nob_fs_run);
m.def("NOB_info", &fs2_nob_fs_info);
m.def("NOFLUSH_run", &fs2_noflush_fs_run);
m.def("NOFLUSH_info", &fs2_noflush_fs_info);
m.def("NOA_run", &fs2_noa_fs_run);
m.def("NOA_info", &fs2_noa_fs_info);
m.def("NOACC_run", &fs2_noacc_fs_run);
m.def("NOACC_info", &fs2_noacc_fs_info);
m.def("TL_run", &fs2_tl_fs_run);
m.def("TL_info", &fs2_tl_fs_info);
}
