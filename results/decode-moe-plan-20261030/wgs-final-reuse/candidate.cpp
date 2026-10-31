
#include <torch/extension.h>
void wr_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,
             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid);
void wr_set_pdl(int64_t on);

std::vector<int64_t> wr_resources();
