#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/mega_moe_fused.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../heuristics/sm90_fp4_mega_moe_h20_fused.hpp"

namespace deep_gemm {

class SM90FP4H20FusedRuntime final
    : public LaunchRuntime<SM90FP4H20FusedRuntime> {
public:
    struct Args {
        int num_max_tokens_per_rank;
        float activation_clamp;
        bool fast_math;
        bool swap_ab;
        bool use_mode2_row_decoder;
        bool single_active_dispatch_warp;
        bool use_interleaved_scheduler;
        bool mxfp4;
        int prefetch_weight_k_blocks;
        bool swap_pipeline_decode;
        bool distributed_expert_bcast;
        bool qoq;
        bool dense_weight_tiles;
        bool half_tile_tasks;
        bool split_k_l1;
        bool l2_half_row_tasks;
        bool split_k_l2;
        bool stream_k;
        bool nvl_fast_epilogue;
        bool fine_combine;
        int k_blocks_per_stage;
        SM90FP4H20FusedConfig config;

        void* y;
        int* cumulative_local_expert_recv_stats;
        int num_tokens;
        layout::SymBuffer<> sym_buffer_ptrs;
        CUtensorMap tensor_map_l1_acts;
        CUtensorMap tensor_map_l1_acts_sf;
        CUtensorMap tensor_map_l1_weights;
        CUtensorMap tensor_map_l1_output;
        CUtensorMap tensor_map_l2_acts;
        CUtensorMap tensor_map_l2_acts_sf;
        CUtensorMap tensor_map_l2_weights;
        const void* l1_weights_ptr;
        const void* l2_weights_ptr;
        const float* l1_global_scales;
        const float* l2_global_scales;
        unsigned long long* phase_stamps;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        const char* kernel_symbol = args.qoq ? "sm90_qoq_mega_moe_h20_fused_impl" :
            (args.mxfp4 ? "sm90_mxfp4_mega_moe_h20_fused_impl" :
                          "sm90_nvfp4_mega_moe_h200_fused_impl");
        const std::string kernel_header = fmt::format(
            "#define DG_NVLINK_BARRIER_TRAP_ONLY_TIMEOUT 1\n"
            "#define sm90_nvfp4_mega_moe_h200_fused_impl {}\n"
            "#include <deep_gemm/impls/sm90_fp4_mega_moe_h20_fused.cuh>",
            kernel_symbol);
        const std::string policy_template_args = fmt::format(
            "/* kSwapABRequested */ {},\n"
            "        /* kSingleActiveDispatchWarp */ {},\n"
            "        /* kUseMode2RowDecoder */ {},\n"
            "        /* kUseInterleavedScheduler */ {},\n"
            "        /* kMXFP4 */ {},\n"
            "        /* kPrefetchWeightKBlocks */ {},\n"
            "        /* kSwapPipelineDecode */ {},\n"
            "        /* kDistributedExpertBcast */ {},\n"
            "        /* kQoQ */ {},\n"
            "        /* kDenseWeightTiles */ {},\n"
            "        /* kHalfTileTasksRequested */ {},\n"
            "        /* kSplitKL1Requested */ {},\n"
            "        /* kL2HalfRowTasksRequested */ {},\n"
            "        /* kSplitKL2Requested */ {},\n"
            "        /* kStreamKRequested */ {},\n"
            "        /* kNvlFastEpilogueRequested */ {},\n"
            "        /* kFineCombineRequested */ {},\n"
            "        /* kKBlocksPerStageRequested */ {}",
            args.swap_ab ? "true" : "false",
            args.single_active_dispatch_warp ? "true" : "false",
            args.use_mode2_row_decoder ? "true" : "false",
            args.use_interleaved_scheduler ? "true" : "false",
            args.mxfp4 ? "true" : "false",
            args.prefetch_weight_k_blocks,
            args.swap_pipeline_decode ? "true" : "false",
            args.distributed_expert_bcast ? "true" : "false",
            args.qoq ? "true" : "false",
            args.dense_weight_tiles ? "true" : "false",
            args.half_tile_tasks ? "true" : "false",
            args.split_k_l1 ? "true" : "false",
            args.l2_half_row_tasks ? "true" : "false",
            args.split_k_l2 ? "true" : "false",
            args.stream_k ? "true" : "false",
            args.nvl_fast_epilogue ? "true" : "false",
            args.fine_combine ? "true" : "false",
            args.k_blocks_per_stage);
        return fmt::format(R"(
{}

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&{}<
        /* kNumMaxTokensPerRank */ {},
        /* kNumExpertsPerWave */ {},
        /* BLOCK_M */ {},
        /* BLOCK_N */ {},
        /* kNumMaxPoolTokens */ {},
        /* kNumPaddedSFPoolTokens */ {},
        /* kNumStages */ {},
        /* kActivationClamp */ {},
        /* kFastMath */ {},
        {}
    >);
}};
)",
            kernel_header,
            kernel_symbol,
            args.num_max_tokens_per_rank,
            args.config.num_experts_per_wave,
            args.config.block_m,
            args.config.block_n,
            args.config.num_max_pool_tokens,
            args.config.num_padded_sf_pool_tokens,
            args.config.num_stages,
            to_string(args.activation_clamp),
            args.fast_math ? "true" : "false",
            policy_template_args);
    }

    static void launch_impl(
            const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(
            kernel, config,
            args.y,
            args.cumulative_local_expert_recv_stats,
            args.num_tokens,
            args.sym_buffer_ptrs,
            args.tensor_map_l1_acts,
            args.tensor_map_l1_acts_sf,
            args.tensor_map_l1_weights,
            args.tensor_map_l1_output,
            args.tensor_map_l2_acts,
            args.tensor_map_l2_acts_sf,
            args.tensor_map_l2_weights,
            args.l1_weights_ptr,
            args.l2_weights_ptr,
            args.l1_global_scales,
            args.l2_global_scales,
            args.phase_stamps));
    }
};

static void sm90_fp4_h20_fused_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts, const torch::Tensor& l1_acts_sf,
    const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::optional<torch::Tensor> l1_global_scales,
    const std::optional<torch::Tensor> l2_global_scales,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const bool& fast_math,
    const bool& mxfp4 = false,
    const std::optional<torch::Tensor>& phase_stamps = std::nullopt,
    const bool& qoq = false
) {
    const int num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const int num_experts = num_experts_per_rank * num_ranks;
    const int num_sms = device_runtime->get_num_sms();
    const int num_padded_sf_pool_tokens = static_cast<int>(l1_acts_sf.size(0));
    const SM90FP4H20FusedInput heuristic_input {
        num_sms,
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk,
        hidden, intermediate_hidden, num_padded_sf_pool_tokens,
        /* rf_decode */ mxfp4 || qoq,
    };
    const auto plan = select_sm90_nvfp4_h200_fused(heuristic_input);
    const auto& config = plan.config;
    using KernelConfig = SM90FP4H20FusedConfig;
    DG_HOST_ASSERT(num_experts_per_rank % config.num_experts_per_wave == 0);
    DG_HOST_ASSERT((config.block_m == 8 || config.block_m == 16 ||
                    config.block_m == 24 || config.block_m == 64 ||
                    config.block_m == 128));
    DG_HOST_ASSERT(config.block_n == 128 || config.block_n == 256);
    DG_HOST_ASSERT(plan.swap_ab == (num_tokens <= 64));
    // MXFP4/QoQ fused weights are packed as dense 256 x BK128 tiles (80 B rows,
    // kSM90FusedPackedTileN) and loaded with one 1D bulk copy per stage. A
    // BLOCK_N=128 kernel tile is one contiguous half of a packed tile.
    const bool dense_weight_tiles = mxfp4 || qoq;
    if (dense_weight_tiles) {
        DG_HOST_ASSERT(256 % config.block_n == 0);
        DG_HOST_ASSERT(l1_weights.is_contiguous() && l2_weights.is_contiguous());
        DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(l1_weights.data_ptr()) % 16 == 0);
        DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(l2_weights.data_ptr()) % 16 == 0);
        DG_HOST_ASSERT(l1_weights.size(2) == (hidden / KernelConfig::kBlockK) * kSM90NVFP4BStoragePerKBlock);
        DG_HOST_ASSERT(l2_weights.size(2) == (intermediate_hidden / KernelConfig::kBlockK) * kSM90NVFP4BStoragePerKBlock);
    }
    // QoQ W4A8 is implemented for the swapAB tiers only (<= 64 tokens per rank).
    DG_HOST_ASSERT(!qoq || plan.swap_ab);

    // Half-tile tasks (kernel `kHalfTileTasks`, master gate
    // fused_layout::kSM90FusedHalfTileTasks): the BM8 MXFP4 RF swapAB tier
    // schedules 128-row tasks (intra-CTA K-split), so the L1 output store box and
    // the L2 activation-scale granularity follow the 128-row task N.
    // H20 A/B (2026-09-09): wins only when the L1 tasks fit one wave (M=2 global
    // tokens: 51 -> 44.5 us); at M=8/16 the per-K-block RF loop cost does not halve
    // with the task, so it loses (65 -> 68.6, 90 -> 101 us). Default OFF; the token
    // count per rank cannot tell M=2 from M=8 (both 1 token/rank), so this is an
    // explicit knob rather than a tier rule.
    const bool half_tile_tasks = fused_layout::kSM90FusedHalfTileTasks &&
        mxfp4 && plan.swap_ab && config.block_m == 8 &&
        get_env<int>("DG_FP4_HALF_TILE", 0) != 0;
    // L1 split-K tasks (kernel `kSplitKL1`): the BM8 MXFP4 RF swapAB tier claims
    // the L1 tasks of the last partial L1 wave as two K halves on two SMs (cross-CTA fp32
    // reduction through fused_layout::Workspace scratch, 5 MB, bounded to
    // kSM90SplitKL1MaxPoolBlocks pool blocks; larger launches fall back in-kernel).
    // The per-WG math shape is unchanged, so unlike half-tile tasks the per-task
    // latency really halves. H20 A/B (2026-09-09, phase stamps, kernel end us,
    // ON vs same-day OFF): M=2 47.4-48.9 vs 50.7; M=8 62.8-64.9 vs 65.3; M=16
    // 86.7-88.6 vs 89.2 (L1 phase -4.5 / -12 us at M=8 / 16, partly given back to
    // a longer L2 tail). Default ON for that tier; DG_FP4_SPLITK_L1=0 disables.
    // Exclusive with half-tile tasks. No effect on TMA boxes / SF granularity.
    const bool split_k_l1 = (mxfp4 || qoq) && plan.swap_ab && config.block_m == 8 &&
        !half_tile_tasks && plan.use_interleaved_scheduler &&
        get_env<int>("DG_FP4_SPLITK_L1", 1) != 0;
    // L2 half-row tasks (kernel `kL2HalfRowTasks`): the BM8 MXFP4 RF swapAB tier
    // schedules 128-row L2 tasks (24 per M block instead of 12) and the two math
    // WGs each own 64 of those rows over the full K (own accumulators + epilogue,
    // no reduction), so the per-WG per-stage work and the L2 task latency halve.
    // Targets the post-split-K critical tail (L2 tasks quantising on 78 SMs).
    // L1 tasks / TMA boxes / SF granularity are untouched. Exclusive with
    // half-tile tasks.
    // H20 A/B (2026-09-09, 8 ranks, phase stamps, back-to-back ON vs OFF): the L2
    // task does NOT get shorter with half the rows per WG (per-task probe 6.5 us
    // both ways: the RF stage chain is latency-bound, as the half-tile experiment
    // already found per K-block), so doubling the task count doubles the L2 tail
    // (M16 last L2 - last L1: 26 vs 10.6 us) and the ON build also slows the L1
    // task (29 vs 22.7 us; one ON cubin spills, STACK 56). Kernel end M8 73.3-73.9
    // vs 62.9-63.8, M16 115.8-116.0 vs 87.1-99.9. Default OFF; DG_FP4_L2_HALFROW=1
    // enables it (numerics verified: T=2/8/16/128/512 + QoQ pass).
    const bool l2_half_row_tasks = mxfp4 && plan.swap_ab && config.block_m == 8 &&
        !half_tile_tasks && plan.use_interleaved_scheduler && dense_weight_tiles &&
        get_env<int>("DG_FP4_L2_HALFROW", 0) != 0;
    // L2 split-K tasks (kernel `kSplitKL2`): the L2 tasks of the last L2 claim
    // batch (the batch that runs after the last L1 wave: (num_l2 - first batch)
    // % 78, M=8: 22 of 96, M=2: all 24; M=16: 44 -> 88 halves do not fit, unsplit)
    // run as two K-range halves (K-blocks [0, 4) publisher / [4, 10) finisher) with
    // the kSplitKL1 protocol on separate scratch slots (workspace +6 MB).
    // H20 A/B (2026-09-09, 3 interleaved reps, rank-0 stamps): NO gain. The L2 tail
    // (last L2 - last L1) is a dependency-latency chain after the last L1 notify
    // (poll + TMA + last stage + epilogue + NVLink scatter, ~5 us at M=8) plus, at
    // M=16, two full L2 waves gated on the last L1 wave (122 tasks / 78 SMs), and
    // a 2-way K split shortens neither: M=8 tail 4.7 vs 4.9 us (kernel end 63.9 vs
    // 64.4), M=16 11.6 vs 10.7 (88.9 vs 87.3), M=2 7.8 vs 6.6 (50.8 vs 48.2).
    // Default OFF; DG_FP4_SPLITK_L2=1 enables (numerics verified: T=2/8/16/128/512
    // + QoQ pass). Exclusive with L2 half-row tasks.
    const bool split_k_l2 = mxfp4 && plan.swap_ab && config.block_m == 8 &&
        !half_tile_tasks && !l2_half_row_tasks && plan.use_interleaved_scheduler &&
        dense_weight_tiles && get_env<int>("DG_FP4_SPLITK_L2", 0) != 0;
    // Stream-K (kernel `kStreamK`): for tiny M the (task, K128 block) units of the
    // L1 and of the L2 phase are split into 78 contiguous near-equal unit ranges (one
    // per SM, task-major) with an n-way cross-CTA fp32 reduction per tile through
    // the split-K scratch, replacing the wave scheduler and both tail splits. Only
    // when the launch's global token count (num_tokens per rank x ranks, an upper
    // bound: 1 token/rank == M <= 8 global) is <= DG_FP4_STREAMK_MAX_M (default 16),
    // so M=128/512 launches are untouched. DG_FP4_STREAMK=1 enables (default 0).
    const int num_global_tokens_upper = num_tokens * num_ranks;
    const bool stream_k = (mxfp4 || qoq) && plan.swap_ab && config.block_m == 8 &&
        !half_tile_tasks && !l2_half_row_tasks && plan.use_interleaved_scheduler &&
        dense_weight_tiles && get_env<int>("DG_FP4_STREAMK", 0) != 0 &&
        num_global_tokens_upper <= get_env<int>("DG_FP4_STREAMK_MAX_M", 16);
    // Fast NVLink-barrier epilogue (kernel `kNvlFastEpilogue`, needs the
    // distributed expert bcast so the first barrier has a prologue grid sync):
    // the two barriers with an epilogue (before dispatch pull, before combine)
    // replace their second grid-wide sync with one SM0-written completion word.
    // H20 A/B (2026-09-09, 3 interleaved reps): within noise (the epilogue grid
    // sync is cheap when SM0 arrives last: combine-barrier segment 5.8 vs 6.3 us at
    // M=8, 5.5 vs 4.8 at M=16; kernel end M=8 64.0 vs 64.4, M=16 87.3 vs 87.3,
    // M=2 47.3 vs 48.2). Default OFF; DG_FP4_NVL_FAST_EPI=1 enables (numerics
    // verified: T=2/8/16/128/512 + QoQ pass).
    const bool nvl_fast_epilogue = get_env<int>("DG_FP4_NVL_FAST_EPI", 0) != 0;
    // Fine-grained combine (kernel `kFineCombine`): the combine NVLink barrier is
    // replaced by per-token arrival counters (each L2 task red.release.sys-adds 1
    // per scattered token row into the destination rank's counter; the combine
    // warp of a token spins on its own counter), so a rank's combine overlaps the
    // other ranks' L2 tails. The sys-scope release fence is issued by dispatch warp
    // 0 through a per-CTA mailbox (a fence.sys in the math warps cost ~1.5-2 us per
    // L2 task on H20 and pushed the last L2 task end out by 2-4 us at M=8/16).
    // H20 A/B (2026-09-09, phase stamps rank 0, median us, ON x2 vs OFF, same
    // session): kernel end M=2 49.9/47.3 vs 47.8, M=8 58.3/58.1 vs 61.6, M=16
    // 82.9/82.4 vs 88.4; last L2 task end -> kernel end M=8 2.4/2.0 vs 7.8, M=16
    // 3.9/3.9 vs 8.1; CUDA-event wall M=2 58.7/56.8 vs 56.9, M=8 72.7/73.9 vs 73.2,
    // M=16 96.0/95.2 vs 97.0. Default ON; DG_FP4_FINE_COMBINE=0 restores the
    // barrier path (numerics identical: T=2/8/16/128/512 + QoQ, 200-iter graph
    // replay stress clean).
    const bool fine_combine = get_env<int>("DG_FP4_FINE_COMBINE", 1) != 0;
    const int task_block_n = half_tile_tasks ? config.block_n / 2 : config.block_n;
    constexpr int kL1ScaleGranK = 128;
    const int l2_scale_gran_k = task_block_n / 2;
    const auto tensor_map_l1_acts = make_tma_2d_desc(
        l1_acts, hidden, config.num_max_pool_tokens,
        KernelConfig::kBlockK, config.block_m,
        static_cast<int>(l1_acts.stride(-2)), KernelConfig::kSwizzleActsMode);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(
        cute::UMMA::Major::MN, l1_acts_sf,
        config.num_padded_sf_pool_tokens, hidden,
        config.block_m, kL1ScaleGranK, 1, 0);
    const auto tensor_map_l1_weights = make_tma_2d_desc(
        l1_weights, static_cast<int>(l1_weights.size(2)),
        num_experts_per_rank * intermediate_hidden * 2,
        kSM90NVFP4BStoragePerKBlock, config.block_n,
        static_cast<int>(l1_weights.stride(-2)), 0);

    const int l1_output_store_block_n = task_block_n / 2;
    const auto tensor_map_l1_output = make_tma_2d_desc(
        l2_acts, intermediate_hidden, config.num_max_pool_tokens,
        l1_output_store_block_n, config.block_m,
        static_cast<int>(l2_acts.stride(-2)), 0);
    const auto tensor_map_l2_acts = make_tma_2d_desc(
        l2_acts, intermediate_hidden, config.num_max_pool_tokens,
        KernelConfig::kBlockK, config.block_m,
        static_cast<int>(l2_acts.stride(-2)), KernelConfig::kSwizzleActsMode);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(
        cute::UMMA::Major::MN, l2_acts_sf,
        config.num_padded_sf_pool_tokens, intermediate_hidden,
        config.block_m, l2_scale_gran_k, 1, 0);
    const auto tensor_map_l2_weights = make_tma_2d_desc(
        l2_weights, static_cast<int>(l2_weights.size(2)),
        num_experts_per_rank * hidden,
        kSM90NVFP4BStoragePerKBlock, config.block_n,
        static_cast<int>(l2_weights.stride(-2)), 0);
    int* cumulative_stats_ptr = cumulative_local_expert_recv_stats.has_value() ?
        cumulative_local_expert_recv_stats->data_ptr<int>() : nullptr;
    const float* l1_global_scales_ptr = l1_global_scales.has_value() ?
        l1_global_scales->data_ptr<float>() : nullptr;
    const float* l2_global_scales_ptr = l2_global_scales.has_value() ?
        l2_global_scales->data_ptr<float>() : nullptr;

    const SM90FP4H20FusedRuntime::Args args = {
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .activation_clamp = activation_clamp,
        .fast_math = fast_math,
        .swap_ab = plan.swap_ab,
        .use_mode2_row_decoder = plan.use_mode2_row_decoder,
        .single_active_dispatch_warp = plan.single_active_dispatch_warp,
        .use_interleaved_scheduler = plan.use_interleaved_scheduler,
        .mxfp4 = mxfp4,
        // Env knob (default 8): number of BK128 weight rows per task prefetched
        // into L2 cache while waiting for activations; 0 disables.
        // H20 A/B (2026-09-09, phase stamps): prefetching 8 K-blocks per task is
        // ~neutral at M<=8 but pollutes L2 / steals HBM at M=16 (-6us L1 phase
        // when disabled), so default it off from 16 tokens up.
        .prefetch_weight_k_blocks = get_env<int>("DG_FP4_PREFETCH_KBLOCKS",
                                                 num_tokens >= 16 ? 0 : 8),
        // DG_FP4_SWAP_PIPE stays OFF (no gain on H200 09-02 nor H20 09-09).
        // DG_FP4_DIST_BCAST defaults ON since the H20 09-09 A/B.
        //   DG_FP4_SWAP_PIPE=1   overlap decode(k+1) with WGMMA(k) in swapAB tiles
        //   DG_FP4_DIST_BCAST=1  spread the dispatch expert-count broadcast over all SMs
        // MXFP4/QoQ swapAB tiers use the RF-decode serial loop (kRFDecode).
        .swap_pipeline_decode = !qoq && !mxfp4 && get_env<int>("DG_FP4_SWAP_PIPE", 0) != 0,
        // H20 A/B (2026-09-09): spreading the expert-count broadcast over all
        // SMs removes ~10us of SM0-serial sys-scope atomics (M=2: 62->52us).
        .distributed_expert_bcast = get_env<int>("DG_FP4_DIST_BCAST", 1) != 0,
        .qoq = qoq,
        // MXFP4/QoQ hosts pack dense (E, N/256, K/128, 256, 80 B) weight tiles
        // (deep_gemm/quantization_{mxfp4,qoq}_fused.py); the B loader then does
        // one 1D bulk copy per stage instead of a TMA-issue-bound 2D box.
        // NVFP4 keeps the row-major fused layout + 2D TMA.
        .dense_weight_tiles = dense_weight_tiles,
        .half_tile_tasks = half_tile_tasks,
        .split_k_l1 = split_k_l1,
        .l2_half_row_tasks = l2_half_row_tasks,
        .split_k_l2 = split_k_l2,
        .stream_k = stream_k,
        .nvl_fast_epilogue = nvl_fast_epilogue,
        .fine_combine = fine_combine,
        // K128 blocks per pipeline stage on the BM8 RF swapAB tiers (MXFP4 and QoQ,
        // kernel `kKBlocksPerStage`; other tiers ignore it). DG_FP4_KBLOCKS_PER_STAGE in
        // {2, 4}: 2 blocks x 4 stages (default) or 4 blocks x 2 stages (same 173 KB
        // of stages, the L2 K loop (10 blocks) ends with a 2-block partial stage).
        // H20 A/B (2026-09-09, phase stamps, kernel end us, 4 vs 2 blocks, same
        // session): M=2 53.5-54.4 vs 46.9; M=8 73.9-75.0 vs 64.9; M=16 101-115 vs
        // 88.3. The per-K128 stage cost barely moves (slot 22: 2655/4 = 664 ns vs
        // 1386/2 = 693 ns at M=8; the exposed drain per block doubles, 330 vs 132 ns)
        // while the task head must land an 80 KB first stage and only ONE stage can
        // be in flight while the other is consumed (vs 3 x 40 KB), so the L1 and L2
        // phases lose 5-6 us / 2-5 us. Default 2 (the 4-block kernel also carries
        // 16 B of ptxas spill at 168 regs; the 2-block one has none).
        .k_blocks_per_stage = (mxfp4 || qoq) ? get_sm90_fp4_h20_bm8_k_blocks_per_stage() : 2,
        .config = config,
        .y = y.data_ptr(),
        .cumulative_local_expert_recv_stats = cumulative_stats_ptr,
        .num_tokens = num_tokens,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .tensor_map_l1_acts = tensor_map_l1_acts,
        .tensor_map_l1_acts_sf = tensor_map_l1_acts_sf,
        .tensor_map_l1_weights = tensor_map_l1_weights,
        .tensor_map_l1_output = tensor_map_l1_output,
        .tensor_map_l2_acts = tensor_map_l2_acts,
        .tensor_map_l2_acts_sf = tensor_map_l2_acts_sf,
        .tensor_map_l2_weights = tensor_map_l2_weights,
        .l1_weights_ptr = l1_weights.data_ptr(),
        .l2_weights_ptr = l2_weights.data_ptr(),
        .l1_global_scales = l1_global_scales_ptr,
        .l2_global_scales = l2_global_scales_ptr,
        .phase_stamps = phase_stamps.has_value() ?
            reinterpret_cast<unsigned long long*>(phase_stamps->data_ptr()) : nullptr,
        .launch_args = LaunchArgs(
            num_sms,
            KernelConfig::kNumThreads,
            config.smem_size, 1)
    };

    const auto code = SM90FP4H20FusedRuntime::generate(args);
    if (phase_stamps.has_value()) {
        DG_HOST_ASSERT(phase_stamps->scalar_type() == torch::kInt64 && phase_stamps->numel() >= 8);
    }
    const std::string kernel_name = std::string(qoq ? "sm90_qoq" : (mxfp4 ? "sm90_mxfp4" : "sm90_nvfp4")) +
        (plan.use_interleaved_scheduler ?
            "_h200_fused_interleaved" :
            (plan.use_mode2_row_decoder ?
                "_h200_fused_mode2_row" :
                "_h200_fused_lut_window"));
    const auto runtime = compiler->build(kernel_name, code);
    SM90FP4H20FusedRuntime::launch(runtime, args);
}

}  // namespace deep_gemm
