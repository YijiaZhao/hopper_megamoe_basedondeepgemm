#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../heuristics/mega_moe.hpp"

namespace deep_gemm {

class SM100FP4FP4MegaMoERuntime final : public LaunchRuntime<SM100FP4FP4MegaMoERuntime> {
public:
    struct Args {
        // Templated arguments
        int num_max_tokens_per_rank;
        int hidden, intermediate_hidden;
        int num_experts, num_topk;
        int sf_vec_size;
        int num_ranks;
        float activation_clamp;
        bool fast_math;
        bool push_dispatch, no_clean_barrier;
        unsigned long long* phase_stamps;
        int num_combine_splits;
        bool static_slots;
        bool deterministic_slots;
        MegaMoEConfig config;

        // Runtime arguments
        void* y;
        int* cumulative_local_expert_recv_stats;
        int num_tokens;
        layout::SymBuffer<> sym_buffer_ptrs;

        // Tensormap
        CUtensorMap tensor_map_l1_acts;
        CUtensorMap tensor_map_l1_acts_sf;
        CUtensorMap tensor_map_l1_weights;
        CUtensorMap tensor_map_l1_weights_sf;
        CUtensorMap tensor_map_l1_output;
        CUtensorMap tensor_map_l2_acts;
        CUtensorMap tensor_map_l2_acts_sf;
        CUtensorMap tensor_map_l2_weights;
        CUtensorMap tensor_map_l2_weights_sf;

        // Launch configs
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_fp4_fp4_mega_moe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp4_fp4_mega_moe_impl<
        {},
        {}, {},
        {}, {},
        {},
        {},
        {}, {}, {},
        {},
        {}, {},
        {},
        {},
        {},
        {}, {}, {},
        {}, {},
        {},
        {},
        {}, {}, {},
        {}, {}, {}
    >);
}};
)", args.num_max_tokens_per_rank,
    args.hidden, args.intermediate_hidden,
    args.num_experts, args.num_topk,
    args.sf_vec_size,
    args.config.num_experts_per_wave,
    args.config.block_m, args.config.block_n, args.config.block_k,
    args.config.store_block_m,
    args.config.sf_block_m, args.config.sf_block_n,
    args.config.num_max_pool_tokens,
    args.config.num_padded_sf_pool_tokens,
    args.config.num_stages,
    args.config.num_dispatch_threads, args.config.num_non_epilogue_threads, args.config.num_epilogue_threads,
    args.launch_args.grid_dim.first, args.num_ranks,
    to_string(args.activation_clamp),
    args.fast_math ? "true" : "false",
    args.push_dispatch ? "true" : "false",
    args.no_clean_barrier ? "true" : "false",
    args.phase_stamps != nullptr ? "true" : "false",
    args.num_combine_splits,
    args.static_slots ? "true" : "false",
    args.deterministic_slots ? "true" : "false");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        // TODO: optimize `args` copy
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.y,
            args.cumulative_local_expert_recv_stats,
            args.num_tokens,
            args.sym_buffer_ptrs,
            args.tensor_map_l1_acts,
            args.tensor_map_l1_acts_sf,
            args.tensor_map_l1_weights,
            args.tensor_map_l1_weights_sf,
            args.tensor_map_l1_output,
            args.tensor_map_l2_acts,
            args.tensor_map_l2_acts_sf,
            args.tensor_map_l2_weights,
            args.tensor_map_l2_weights_sf,
            args.phase_stamps
        ));
    }
};

static void sm100_fp4_fp4_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts, const torch::Tensor& l1_acts_sf,
    const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const torch::Tensor& l1_weights_sf, const torch::Tensor& l2_weights_sf,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const bool& fast_math,
    const int& sf_vec_size
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_padded_sf_pool_tokens = static_cast<int>(l1_acts_sf.size(0));

    // Heuristics
    const auto config = get_mega_moe_config(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk, hidden, intermediate_hidden, num_padded_sf_pool_tokens,
        /* fp4_acts */ true, sf_vec_size);

    // Make tensormap
    // NOTES: all FP4 operand maps are byte-typed over packed E2M1 (inner dim = elements / 2), 64B swizzle;
    //        the L1 output map is byte-typed with no swizzle (row-major packed tile)
    const int kGranK = sf_vec_size;
    const auto l1_acts_u8 = l1_acts.view(torch::kUInt8);
    const auto l2_acts_u8 = l2_acts.view(torch::kUInt8);
    const auto l1_weights_u8 = l1_weights.view(torch::kUInt8);
    const auto l2_weights_u8 = l2_weights.view(torch::kUInt8);
    DG_HOST_ASSERT((config.block_n == 128 or config.block_n == 64) and (config.block_k == 128 or config.block_k == 256));
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts_u8,
                                                     hidden / 2, config.num_max_pool_tokens,
                                                     config.block_k / 2, config.load_block_m,
                                                     static_cast<int>(l1_acts_u8.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf,
                                                        config.num_padded_sf_pool_tokens, hidden,
                                                        config.sf_block_m, kGranK,
                                                        1, 0);
    const auto tensor_map_l1_weights = make_tma_2d_desc(l1_weights_u8,
                                                        hidden / 2, num_experts_per_rank * intermediate_hidden * 2,
                                                        config.block_k / 2, config.load_block_n,
                                                        static_cast<int>(l1_weights_u8.stride(-2)),
                                                        config.swizzle_weights_mode);
    const auto tensor_map_l1_weights_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_weights_sf,
                                                           intermediate_hidden * 2, hidden,
                                                           config.block_n, kGranK,
                                                           num_experts_per_rank, 0);
    // NOTES: L1 output and L2 activations are essentially the same tensor.
    // Post-SwiGLU output has half the N width (`BLOCK_N / 2` per input tile),
    // so the swizzle mode is also halved (128 -> 64).
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts_u8,
                                                       intermediate_hidden / 2, config.num_max_pool_tokens,
                                                       config.block_n / 4, config.store_block_m,
                                                       static_cast<int>(l2_acts_u8.stride(-2)),
                                                       0);
    const auto tensor_map_l2_acts = make_tma_2d_desc(l2_acts_u8,
                                                     intermediate_hidden / 2, config.num_max_pool_tokens,
                                                     config.block_k / 2, config.load_block_m,
                                                     static_cast<int>(l2_acts_u8.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf,
                                                        config.num_padded_sf_pool_tokens, intermediate_hidden,
                                                        config.sf_block_m, kGranK,
                                                        1, 0);
    const auto tensor_map_l2_weights = make_tma_2d_desc(l2_weights_u8,
                                                        intermediate_hidden / 2, num_experts_per_rank * hidden,
                                                        config.block_k / 2, config.load_block_n,
                                                        static_cast<int>(l2_weights_u8.stride(-2)),
                                                        config.swizzle_weights_mode);
    const auto tensor_map_l2_weights_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_weights_sf,
                                                           hidden, intermediate_hidden,
                                                           config.block_n, kGranK,
                                                           num_experts_per_rank, 0);

    // Counter-based synchronisation knobs (see the kernel):
    //   DG_SM100_PUSH_DISPATCH (default 1): push dispatch + DONE flags instead of NVLink barrier #1 + pull
    //   DG_SM100_NO_CLEAN_BARRIER (default 1, needs push): parity-rotated pool, no barrier #3
    //   DG_SM100_PHASE_STAMPS_PTR: device address (decimal) of 32 x u64 globaltimer stamp slots
    // The fixed-stride push pool must hold every routed row of an expert in the worst case (all ranks' tokens)
    const int num_pool_parity_slots = get_env<int>("DG_SM100_NO_CLEAN_BARRIER", 1) != 0 ? 2 : 1;
    const int push_blocks_per_expert = (config.num_max_pool_tokens / config.block_m) / (num_experts_per_rank * num_pool_parity_slots);
    bool push_dispatch = get_env<int>("DG_SM100_PUSH_DISPATCH", 1) != 0;
    if (push_dispatch and num_tokens * num_ranks > push_blocks_per_expert * config.block_m) {
        if (get_env<int>("DG_JIT_DEBUG"))
            printf("SM100 push dispatch disabled: %d x %d rows exceed the %d-row pool stride\n",
                   num_tokens, num_ranks, push_blocks_per_expert * config.block_m);
        push_dispatch = false;
    }
    const bool no_clean_barrier = push_dispatch and num_pool_parity_slots == 2;
    // Static slots (DG_SM100_STATIC_SLOTS, default 1): tiny M where every expert's rows fit one M block
    const bool static_slots = push_dispatch and get_env<int>("DG_SM100_STATIC_SLOTS", 1) != 0 and
                              num_tokens * num_ranks <= config.block_m;
    // Deterministic slots (DG_SM100_DETERMINISTIC_SLOTS, default 0: measured neutral, the ticket RTT was hidden behind the first-row prefetch): no remote ticket when every rank has at most
    // block_m / ranks tokens (all ranks must take the same decision: the harness uses uniform token counts)
    const bool deterministic_slots = static_slots and get_env<int>("DG_SM100_DETERMINISTIC_SLOTS", 0) != 0 and
                                     config.block_m % num_ranks == 0 and config.block_m <= 16 and
                                     num_tokens <= config.block_m / num_ranks;
    unsigned long long* phase_stamps = nullptr;
    if (const auto stamps_env = get_env<std::string>("DG_SM100_PHASE_STAMPS_PTR"); not stamps_env.empty())
        phase_stamps = reinterpret_cast<unsigned long long*>(std::stoull(stamps_env));
    // Combine hidden splits (DG_SM100_COMBINE_SPLITS, default auto): the largest divisor of 12 whose (token, chunk)
    // items still fit in one round of combine warps, with 512 B-aligned chunks; 0 = one warp per token
    int num_combine_splits = get_env<int>("DG_SM100_COMBINE_SPLITS", -1);
    if (num_combine_splits < 0) {
        num_combine_splits = 0;
        const int num_combine_warps = device_runtime->get_num_sms() * (config.num_epilogue_threads / 32);
        for (const int s: {12, 6, 4, 3, 2}) {
            if ((hidden * 2) % (s * 512) == 0 and std::max(num_tokens, 1) * s <= num_combine_warps) {
                num_combine_splits = s;
                break;
            }
        }
    }

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();

    // Launch
    const auto num_sms = device_runtime->get_num_sms();
    const SM100FP4FP4MegaMoERuntime::Args args = {
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .hidden = hidden, .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts, .num_topk = num_topk,
        .sf_vec_size = sf_vec_size,
        .num_ranks = num_ranks,
        .activation_clamp = activation_clamp,
        .fast_math = fast_math,
        .push_dispatch = push_dispatch,
        .no_clean_barrier = no_clean_barrier,
        .phase_stamps = phase_stamps,
        .num_combine_splits = num_combine_splits,
        .static_slots = static_slots,
        .deterministic_slots = deterministic_slots,
        .config = config,
        .y = y.data_ptr(),
        .cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats_ptr,
        .num_tokens = num_tokens,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .tensor_map_l1_acts = tensor_map_l1_acts,
        .tensor_map_l1_acts_sf = tensor_map_l1_acts_sf,
        .tensor_map_l1_weights = tensor_map_l1_weights,
        .tensor_map_l1_weights_sf = tensor_map_l1_weights_sf,
        .tensor_map_l1_output = tensor_map_l1_output,
        .tensor_map_l2_acts = tensor_map_l2_acts,
        .tensor_map_l2_acts_sf = tensor_map_l2_acts_sf,
        .tensor_map_l2_weights = tensor_map_l2_weights,
        .tensor_map_l2_weights_sf = tensor_map_l2_weights_sf,
        .launch_args = LaunchArgs(num_sms,
                                  config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads,
                                  config.smem_size, 2)
    };

    const auto code = SM100FP4FP4MegaMoERuntime::generate(args);
    const auto runtime = compiler->build(std::string("sm100_fp4_fp4_mega_moe") + (push_dispatch ? "_push" : "") +
                                         (no_clean_barrier ? "_rot" : "") + (phase_stamps != nullptr ? "_stamps" : "") +
                                         (num_combine_splits > 0 ? "_cs" + std::to_string(num_combine_splits) : "") + (static_slots ? "_ss" : "") + (deterministic_slots ? "_ds" : ""), code);
    SM100FP4FP4MegaMoERuntime::launch(runtime, args);
}

} // namespace deep_gemm
