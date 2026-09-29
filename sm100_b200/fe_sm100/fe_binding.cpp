#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "fable_frontend.h"

// SM100 port of the H20 Fable frontend: router logits + top-8 + softmax + activation quantisation into the
// MegaMoE input buffers. mode 2 = FP8/UE8M0-32, 3 = MXFP4/UE8M0-32, 4 = NVFP4/UE4M3-16 (B200 kernels); 0/1 = H20 formats.
static void fe_forward(const torch::Tensor& hidden, const torch::Tensor& router_weight,
                       const torch::Tensor& x, const torch::Tensor& x_sf,
                       const torch::Tensor& topk_idx, const torch::Tensor& topk_weights,
                       const torch::Tensor& workspace, int64_t mode, int64_t l2_persist) {
    TORCH_CHECK(hidden.dim() == 2 && hidden.scalar_type() == torch::kBFloat16 && hidden.is_contiguous());
    TORCH_CHECK(router_weight.dim() == 2 && router_weight.scalar_type() == torch::kBFloat16 && router_weight.is_contiguous());
    const int m = static_cast<int>(hidden.size(0)), h = static_cast<int>(hidden.size(1));
    const int e = static_cast<int>(router_weight.size(0));
    const int topk = static_cast<int>(topk_idx.size(1));
    TORCH_CHECK(router_weight.size(1) == h && m >= 1 && m <= 64);
    TORCH_CHECK(topk_idx.scalar_type() == torch::kInt64 && topk_weights.scalar_type() == torch::kFloat32);
    TORCH_CHECK(topk_idx.size(0) >= m && topk_weights.size(0) >= m);
    TORCH_CHECK(x.size(0) >= m && x_sf.size(0) >= m);
    TORCH_CHECK(workspace.nbytes() >= static_cast<size_t>(256 + 4 * 64 * e * 4));
    launch_router_quant_topk_frontend(
        hidden.data_ptr(), router_weight.data_ptr(), x.data_ptr(), x_sf.data_ptr(),
        topk_idx.data_ptr(), topk_weights.data_ptr(), workspace.data_ptr(), workspace.nbytes(),
        m, h, e, topk, static_cast<int>(mode), /*stamps*/ 0, static_cast<int>(l2_persist), /*wlayout*/ 0,
        /*select_in_mega*/ 0, /*zero_row_unrouted*/ 1, at::cuda::getCurrentCUDAStream().stream());
}

static int64_t fe_workspace_bytes(int64_t e) { return static_cast<int64_t>(router_quant_topk_frontend_workspace_bytes(static_cast<int>(e))); }
static int64_t fe_router_ctas(int64_t m, int64_t h, int64_t e, int64_t topk) { return router_quant_topk_frontend_router_ctas(m, h, e, topk); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
    mod.def("fe_forward", &fe_forward);
    mod.def("fe_workspace_bytes", &fe_workspace_bytes);
    mod.def("fe_router_ctas", &fe_router_ctas);
}
