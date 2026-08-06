#pragma once

#include <unordered_set>

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

// ============================================================================
// SM90 (Hopper) MXFP4 MegaMoE host runtime. This is intentionally independent
// from the FP8 runtime so adding packed-B dequant cannot change FP8 codegen.
// ============================================================================

class SM90MXFP4MegaMoERuntime final : public LaunchRuntime<SM90MXFP4MegaMoERuntime> {
public:
    enum class KernelPhase {
        Linear1,
        Linear2
    };

    struct Args {
        // Templated arguments
        int num_max_tokens_per_rank;
        int hidden, intermediate_hidden;
        int num_experts, num_topk;
        int num_ranks;
        float activation_clamp;
        bool fast_math;
        bool direct_l2_scatter;
        bool phase_profile;
        bool l2_nmajor_schedule;
        bool one_warp_cleanup;
        // Small-M 2-CTAs/SM slim variant (DG_W4A8_INT_SMALLM_OCC2): doubled
        // grid, shallow pipeline, halved per-CTA register budget.
        bool occ2;
        // Diagnostic bisect (DG_W4A8_INT_SMALLM_OCC2_NO2X=1): slim config
        // (stages/regs/pad) but single grid -- isolates the slim pipeline
        // from the doubled-grid protocol.
        bool occ2_doubled;
        // Number of contiguous attention-TP ranks. Router weights are
        // replicated, but every physical rank owns a distinct DP token shard.
        int attn_tp_size;
        bool fuse_router;
        bool router_renormalize;
        bool router_logits_input_mode;
        bool tp_distributed_combine;
        bool fuse_input_quant;
        KernelPhase kernel_phase;
        MegaMoESM90Config config;

        // Runtime arguments
        void* y;
        int* cumulative_local_expert_recv_stats;
        int num_tokens;
        const void* hidden_input;
        const void* router_input;
        const void* router_weight;
        const void* router_logits_input;
        layout::SymBuffer<> sym_buffer_ptrs;

        // Packed E2M1 values and UE4M3 scales share the weight TMA descriptor.
        // Optional global scales remain raw per-expert pointers.
        CUtensorMap tensor_map_l1_acts;
        CUtensorMap tensor_map_l1_acts_sf;
        CUtensorMap tensor_map_l1_weights;
        CUtensorMap tensor_map_l1_weights_sf;
        const float* l1_global_scales;
        CUtensorMap tensor_map_l1_output;
        CUtensorMap tensor_map_l2_acts;
        CUtensorMap tensor_map_l2_acts_sf;
        CUtensorMap tensor_map_l2_weights;
        CUtensorMap tensor_map_l2_weights_sf;
        const float* l2_global_scales;

        // Launch configs
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        const char* kernel_symbol = args.kernel_phase == KernelPhase::Linear1 ? "sm90_mxfp4_mega_moe_l1_impl" :
            "sm90_mxfp4_mega_moe_l2_impl";
        // iter17 relative-LUT fold: auto-on at mid M where the per-K32
        // promotion tax dominates (measured crossovers on H20: flash ~M128,
        // pro ~M256). DG_MXFP4_REL_LUT=0/1 forces; the define lands in the
        // generated source so it participates in the JIT cache key.
        // K32P study (2026-07-16, 10-rep paired A/B, MXFP4_K32P_STATUS.md):
        // the mid-M crossovers re-confirmed (flash M128 tie, M256 rel-LUT
        // -34%; pro M128 K32 -3.5%, M256 rel-LUT -8%), and the plain per-K32
        // path stays the winner at flash M2-64 / pro M1-128. The one flip:
        // flash M1 (protocol-bound) prefers the single-accumulator rel-LUT
        // (median -25%; the per-K32 arm was bimodal with a 270-350us slow
        // mode in 5/10 reps, rel-LUT slow-mode-free) -> engage rel-LUT at
        // M <= 1 for the small-IH family (pro M1 measured the other way).
        const int rel_lut_env = get_env<int>("DG_MXFP4_REL_LUT", -1);
        const int rel_lut_min_m = get_env<int>("DG_MXFP4_REL_LUT_MIN_M",
            args.intermediate_hidden <= 2048 ? 128 : 256);
        const int rel_lut_smallm_max = get_env<int>("DG_MXFP4_REL_LUT_SMALLM_MAX",
            args.intermediate_hidden <= 2048 ? 1 : 0);
        const int rel_lut = rel_lut_env >= 0 ? (rel_lut_env != 0 ? 1 : 0) :
            ((args.num_tokens >= rel_lut_min_m or
              args.num_tokens <= rel_lut_smallm_max) ? 1 : 0);
        const int abs_scale256 =
            get_env<int>("DG_MXFP4_ABS_SCALE256", 0) != 0 ? 1 : 0;
        // INT4 has one canonical format only: QoQ + asymmetric zero point,
        // full INT4 L1/L2, decoded with SHIFTXOR. Legacy GPTQ/fp32-scale,
        // hybrid L2, PRE, SHADOW and PRELUT modes have been retired.
        const int w4a8_int = get_env<int>("DG_W4A8_INT", 0) != 0 ? 1 : 0;
        const int w4a8_int_l2 = w4a8_int;
        const int w4a8_int_pre = 0;
        const int w4a8_int_shadow = 0;
        const int w4a8_int_qoq = w4a8_int;
        const int w4a8_int_qoq_zp = w4a8_int;
        const int w4a8_int_qoq_zp_shiftxor = w4a8_int;
        const int w4a8_int_direct_nibble =
            (w4a8_int and get_env<int>("DG_W4A8_INT_DIRECT_NIBBLE", 0) != 0) ? 1 : 0;
        const int w4a8_int_zsub_xor =
            (w4a8_int and get_env<int>("DG_W4A8_INT_ZSUB_XOR", 0) != 0) ? 1 : 0;
        const int w4a8_int_qoq_zp_prelut = 0;
        const int w4a8_int_qoq_zp_prelut_const = 0;
        // Diagnostic-only: compile the comm-barrier timeout printf path in
        // (degrades scheduling; never enable for perf runs).
        const int comm_barrier_printf = get_env<int>("DG_COMM_BARRIER_TIMEOUT_PRINTF", 0) != 0 ? 1 : 0;
        const int occ2_spin_trap = get_env<int>("DG_OCC2_SPIN_TRAP", 0) != 0 ? 1 : 0;
        const int occ2_spin_trap_l1full =
            (occ2_spin_trap or get_env<int>("DG_OCC2_SPIN_TRAP_L1FULL", 0) != 0) ? 1 : 0;
        const int occ2_spin_trap_sched =
            (occ2_spin_trap or get_env<int>("DG_OCC2_SPIN_TRAP_SCHED", 0) != 0) ? 1 : 0;
        const int occ2_baseline_regs = get_env<int>("DG_OCC2_BASELINE_REGS", 0) != 0 ? 1 : 0;
        const int occ2_lb1 = get_env<int>("DG_OCC2_LB1", 0) != 0 ? 1 : 0;
        const int occ2_no_pad = get_env<int>("DG_OCC2_NO_PAD", 0) != 0 ? 1 : 0;
        return fmt::format(R"(
#define DG_OCC2_LB1 {}
#define DG_OCC2_NO_PAD {}
#define DG_COMM_BARRIER_TIMEOUT_PRINTF {}
#define DG_OCC2_SPIN_TRAP_L1FULL {}
#define DG_OCC2_SPIN_TRAP_SCHED {}
#define DG_OCC2_BASELINE_REGS {}
#define DG_MXFP4_REL_LUT {}
#define DG_MXFP4_ABS_SCALE256 {}
#define DG_W4A8_INT {}
#define DG_W4A8_INT_L2 {}
#define DG_W4A8_INT_PRE {}
#define DG_W4A8_INT_SHADOW {}
#define DG_W4A8_INT_QOQ {}
#define DG_W4A8_INT_QOQ_ZP {}
#define DG_W4A8_INT_QOQ_ZP_PRELUT {}
#define DG_W4A8_INT_QOQ_ZP_PRELUT_CONST {}
#define DG_W4A8_INT_QOQ_ZP_SHIFTXOR {}
#define DG_W4A8_INT_DIRECT_NIBBLE {}
#define DG_W4A8_INT_ZSUB_XOR {}
#include <deep_gemm/impls/sm90_mxfp4_mega_moe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&{}<
        {},
        {}, {},
        {}, {},
        {},
        {}, {}, {},
        {},
        {}, {},
        {},
        {}, {}, {},
        {},
        {}, {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {}
    >);
}};
)",
    occ2_lb1,
    occ2_no_pad,
    comm_barrier_printf,
    occ2_spin_trap_l1full,
    occ2_spin_trap_sched,
    occ2_baseline_regs,
    rel_lut,
    abs_scale256,
    w4a8_int,
    w4a8_int_l2,
    w4a8_int_pre,
    w4a8_int_shadow,
    w4a8_int_qoq,
    w4a8_int_qoq_zp,
    w4a8_int_qoq_zp_prelut,
    w4a8_int_qoq_zp_prelut_const,
    w4a8_int_qoq_zp_shiftxor,
    w4a8_int_direct_nibble,
    w4a8_int_zsub_xor,
    kernel_symbol,
    args.num_max_tokens_per_rank,
    args.hidden, args.intermediate_hidden,
    args.num_experts, args.num_topk,
    args.config.num_experts_per_wave,
    args.config.block_m, args.config.block_n, args.config.block_k,
    args.config.num_ring_tokens,
    args.config.num_sf_ring_tokens,
    args.config.sf_ring_stride_tokens,
    args.config.num_stages,
    args.config.num_dispatch_threads, args.config.num_non_epilogue_threads, args.config.num_epilogue_threads,
    args.config.cluster_size,
    args.launch_args.grid_dim.first, args.num_ranks,
    to_string(args.activation_clamp),
    args.fast_math ? "true" : "false",
    args.direct_l2_scatter ? "true" : "false",
    args.phase_profile ? "true" : "false",
    args.l2_nmajor_schedule ? "true" : "false",
    args.one_warp_cleanup ? "true" : "false",
    args.config.swap_ab ? "true" : "false",
    args.occ2 ? "true" : "false",
    args.attn_tp_size,
    args.fuse_router ? "true" : "false",
    args.router_renormalize ? "true" : "false",
    args.router_logits_input_mode ? "true" : "false",
    args.tp_distributed_combine ? "true" : "false",
    args.fuse_input_quant ? "true" : "false");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.y,
            args.cumulative_local_expert_recv_stats,
            args.num_tokens,
            args.hidden_input,
            args.router_input,
            args.router_weight,
            args.router_logits_input,
            args.sym_buffer_ptrs,
            args.tensor_map_l1_acts,
            args.tensor_map_l1_acts_sf,
            args.tensor_map_l1_weights,
            args.tensor_map_l1_weights_sf,
            args.l1_global_scales,
            args.tensor_map_l1_output,
            args.tensor_map_l2_acts,
            args.tensor_map_l2_acts_sf,
            args.tensor_map_l2_weights,
            args.tensor_map_l2_weights_sf,
            args.l2_global_scales
        ));
    }
};

static void sm90_mxfp4_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts, const torch::Tensor& l1_acts_sf,
    const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const torch::Tensor& l1_weights_sf, const torch::Tensor& l2_weights_sf,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::optional<torch::Tensor> l1_global_scales,
    const std::optional<torch::Tensor> l2_global_scales,
    const std::optional<torch::Tensor> router_input,
    const std::optional<torch::Tensor> router_weight,
    const std::optional<torch::Tensor> router_logits,
    const bool& router_renormalize,
    const std::optional<torch::Tensor> tp_combine_output,
    const std::optional<torch::Tensor> hidden_input,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& deployment_block_n,
    const int& attn_tp_size,
    const float& activation_clamp,
    const bool& fast_math
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    DG_HOST_ASSERT(attn_tp_size >= 1 and num_ranks % attn_tp_size == 0);
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_ring_tokens = static_cast<int>(l1_acts.size(0));
    const auto num_sf_ring_tokens = static_cast<int>(l1_acts_sf.size(0));
    const bool router_logits_input_mode = router_logits.has_value();
    const bool fuse_router = router_input.has_value() or router_weight.has_value() or router_logits_input_mode;
    DG_HOST_ASSERT(router_input.has_value() == router_weight.has_value());
    DG_HOST_ASSERT(not (router_logits_input_mode and router_input.has_value()));
    if (router_input.has_value()) {
        DG_HOST_ASSERT(router_input->scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(router_weight->scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(router_input->dim() == 2 and router_weight->dim() == 2);
        DG_HOST_ASSERT(router_input->size(0) == num_tokens and router_input->size(1) == hidden);
        DG_HOST_ASSERT(router_weight->size(0) == num_experts and router_weight->size(1) == hidden);
        DG_HOST_ASSERT(router_input->is_contiguous() and router_weight->is_contiguous());
        DG_HOST_ASSERT(router_input->device() == y.device() and router_weight->device() == y.device());
    }
    if (router_logits_input_mode) {
        DG_HOST_ASSERT(router_logits->scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(router_logits->dim() == 2);
        DG_HOST_ASSERT(router_logits->size(0) == num_tokens and router_logits->size(1) == num_experts);
        DG_HOST_ASSERT(router_logits->is_contiguous() and router_logits->device() == y.device());
    }
    const bool tp_distributed_combine = tp_combine_output.has_value();
    if (tp_distributed_combine) {
        DG_HOST_ASSERT(attn_tp_size > 1 and num_tokens % attn_tp_size == 0);
        DG_HOST_ASSERT(tp_combine_output->scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(tp_combine_output->dim() == 2);
        DG_HOST_ASSERT(tp_combine_output->size(0) == num_tokens / attn_tp_size);
        DG_HOST_ASSERT(tp_combine_output->size(1) == hidden);
        DG_HOST_ASSERT(tp_combine_output->is_contiguous() and tp_combine_output->device() == y.device());
    }
    const bool fuse_input_quant = hidden_input.has_value();
    if (fuse_input_quant) {
        DG_HOST_ASSERT(get_env<int>("DG_W4A8_INT", 0) != 0);
        DG_HOST_ASSERT(hidden_input->scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(hidden_input->dim() == 2);
        DG_HOST_ASSERT(hidden_input->size(0) == num_tokens and hidden_input->size(1) == hidden);
        DG_HOST_ASSERT(hidden_input->is_contiguous() and hidden_input->device() == y.device());
    }

    // Start from the retained FP8 policy, then constrain only the dimensions
    // and resources changed by in-kernel MXFP4 dequantization.
    auto config = get_mega_moe_config_sm90(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk,
        hidden, intermediate_hidden, num_ring_tokens, num_sf_ring_tokens);

    DG_HOST_ASSERT(deployment_block_n == 128 or deployment_block_n == 256);
    DG_HOST_ASSERT(static_cast<int>(l1_weights_sf.size(3)) == deployment_block_n);
    DG_HOST_ASSERT(static_cast<int>(l2_weights_sf.size(3)) == deployment_block_n);

    config.block_m = 64;
    config.block_n = deployment_block_n;
    config.block_k = 128;
    config.cluster_size = 1;
    config.swap_ab = deployment_block_n == 128 and config.swap_ab;
    // MXFP4 RF decode made the swapAB path much cheaper than it was for FP8;
    // allow extending its M coverage past the inherited FP8 policy threshold.
    // Measured on H20: Pro M256 -40% vs the default path, other points flat.
    // iter17: when the relative-LUT fold engages it frees the accumulator
    // registers that capped swapAB, and inline REL_LUT beats the prologue
    // tier up to ~M1024 (measured) -> extend the default cap accordingly.
    // Keep this mirror of generate_impl's rel_lut heuristic in sync (incl.
    // the K32P-study small-M term; it only fires at M <= 1 where both
    // swap_ab caps are >= M, so it cannot change the swap_ab decision).
    const int rel_lut_env = get_env<int>("DG_MXFP4_REL_LUT", -1);
    const int rel_lut_min_m = get_env<int>("DG_MXFP4_REL_LUT_MIN_M",
        intermediate_hidden <= 2048 ? 128 : 256);
    const int rel_lut_smallm_max = get_env<int>("DG_MXFP4_REL_LUT_SMALLM_MAX",
        intermediate_hidden <= 2048 ? 1 : 0);
    const bool rel_lut_engaged = rel_lut_env >= 0 ? rel_lut_env != 0 :
        (num_tokens >= rel_lut_min_m or num_tokens <= rel_lut_smallm_max);
    const int default_swap_ab_max_m = rel_lut_engaged ? 1024 : 256;
    if (deployment_block_n == 128 and not config.swap_ab and num_tokens > 0 and
        num_tokens <= get_env<int>("DG_MXFP4_SWAP_AB_MAX_M", default_swap_ab_max_m))
        config.swap_ab = true;
    // Legacy PRE/PRELUT modes were removed; canonical QoQ always decodes the
    // packed uint4 stream in-kernel with SHIFTXOR.
    const bool pre_decoded_b = false;
    const bool zp_prelut = false;
    const bool zp_prelut_const = false;
    const int zp_prelut_smem_size = 0;
    config.num_epilogue_threads = deployment_block_n == 256 ? 256 :
        (config.swap_ab ? 256 : 128);
    const bool compact_frontend = deployment_block_n == 256 or config.swap_ab;
    config.num_dispatch_threads = compact_frontend ? 64 : 128;
    config.num_non_epilogue_threads = compact_frontend ? 64 : 128;

    const auto policy = get_sm90_moe_heuristic_policy(
        num_experts_per_rank, num_tokens, num_topk,
        intermediate_hidden, config.block_m, config.block_n);
    config.direct_l2_scatter = not config.swap_ab and policy.direct_l2_scatter();
    config.l2_nmajor_schedule = policy.l2_nmajor_schedule(
        get_env<int>("DG_SM90_MOE_EPLB_HINT", 0) != 0,
        get_env<int>("DG_SM90_MOE_SKEW_HINT", 0) != 0);
    config.one_warp_cleanup = policy.one_warp_cleanup(
        get_env<int>("DG_SM90_MOE_MASKED_HINT", 0) != 0);
    config.num_experts_per_wave = get_num_experts_per_wave_for_mega_moe_sm90(
        num_experts_per_rank, num_tokens, num_topk,
        intermediate_hidden, config.block_m, config.block_n,
        device_runtime->get_num_sms());
    config.sf_ring_stride_tokens = layout::get_num_sf_ring_tokens(
        config.num_ring_tokens, config.block_m);

    const bool stage5 = policy.stage5_pipeline(
        config.direct_l2_scatter,
        get_env<int>("DG_SM90_MOE_EPLB_HINT", 0) != 0,
        get_env<int>("DG_SM90_MOE_SKEW_HINT", 0) != 0,
        get_env<int>("DG_SM90_MOE_MASKED_HINT", 0) != 0);
    // On the RF path (swapAB) the decoded-FP8 B buffer (block_n*block_k per
    // stage) is not allocated in the kernel; grant the helper those bytes so
    // it selects a deeper pipeline, then re-validate with the true footprint.
    const int rf_reclaim_per_stage = config.swap_ab ? deployment_block_n * 128 : 0;
    // PRE mode frees the packed-B scratch; grant the budget back and request
    // a deeper pipeline (B rows are 2x bytes, more stages hide the TMA).
    const int default_rf_stages = num_tokens <= 32 ? 4 : 6;
    int requested_stages = config.swap_ab ?
        get_env<int>("DG_MXFP4_RF_STAGES", default_rf_stages) :
        (pre_decoded_b ? get_env<int>("DG_W4A8_PRE_STAGES", 6) : (stage5 ? 5 : 4));
    const int packed_scratch_per_stage = pre_decoded_b ? 0 : deployment_block_n * 80;
    while (true) {
        const auto [num_stages, fp8_smem_size] = get_pipeline_config_for_mega_moe_sm90(
            SM90ArchSpec::smem_capacity +
                (config.swap_ab ? requested_stages * rf_reclaim_per_stage : 0),
            num_experts, hidden,
            config.block_m, config.block_n, config.block_k,
            config.num_dispatch_threads / 32, config.num_epilogue_threads / 32,
            config.direct_l2_scatter, requested_stages, config.swap_ab);
        // No LUT region; per-stage E8M0 coefficient array (4 bytes per weight
        // row) plus the dequant barrier per stage; minus the reclaimed B buffer.
        const int mxfp4_smem_size = fp8_smem_size -
            num_stages * rf_reclaim_per_stage +
            num_stages * (packed_scratch_per_stage + deployment_block_n * 4 +
                          static_cast<int>(sizeof(uint64_t))) +
            zp_prelut_smem_size;
        if (mxfp4_smem_size <= SM90ArchSpec::smem_capacity) {
            config.num_stages = num_stages;
            config.smem_size = mxfp4_smem_size;
            break;
        }
        DG_HOST_ASSERT(num_stages > 2);
        requested_stages = num_stages - 1;
    }

    // ------------------------------------------------------------------
    // Small-M OCC2 slim variant (env-gated, default off): run 2 CTAs/SM on
    // the compact swapAB RF path by shrinking the pipeline so the per-CTA
    // footprint fits under half the SM SMEM (H20/H100: 228KB carveout, 1KB
    // per-CTA reserve -> 115712B budget) and halving the register budget in
    // the kernel (kOcc2). The grid doubles; every kNumSMs-templated protocol
    // (grid sync, NVLink barrier, scheduler striding, dispatch pull) scales
    // with the doubled CTA count consistently.
    // The SMEM accounting below mirrors the kernel layout EXACTLY for the
    // swapAB + packed-scratch + (optional) ZP-PRELUT configuration.
    // ------------------------------------------------------------------
    // Default M gate is empirical (H20 paired medians, 2026-07-13): M2-8 win
    // on both models (flash M2 -14%, M4/M8 -3%; pro -4..-5%), pro M1 wins
    // (-4.7%) but flash M1 regresses (+12%, protocol-bound at 6 active
    // experts) -> M1 only engages for the larger FFN (IH >= 3072).
    const int occ2_min_m_default = intermediate_hidden >= 3072 ? 1 : 2;
    bool occ2 = ((get_env<int>("DG_W4A8_INT_SMALLM_OCC2", 0) != 0 and get_env<int>("DG_W4A8_INT", 0) != 0) or get_env<int>("DG_MXFP4_OCC2", 0) != 0) and
                not pre_decoded_b and config.swap_ab and
                num_tokens >= get_env<int>("DG_W4A8_INT_SMALLM_OCC2_MIN_M", occ2_min_m_default) and
                num_tokens <= get_env<int>("DG_W4A8_INT_SMALLM_OCC2_MAX_M", 8);
    if (occ2) {
        constexpr int kOcc2SmemBudget = 115712;
        const int num_epilogue_warps_o = config.num_epilogue_threads / 32;      // 8
        const int num_epilogue_wgs_o = config.num_epilogue_threads / 128;       // 2
        const int wg_block_n_o = config.block_n / num_epilogue_wgs_o;           // 64 (split-N)
        const int smem_expert = align(num_experts * 4, 1024);
        const int smem_send = align(hidden * (config.num_dispatch_threads / 32), 1024);
        const int cd_l1 = num_epilogue_wgs_o * 64 * (wg_block_n_o / 2);          // FP8
        const int cd_l2 = num_epilogue_wgs_o * 64 * wg_block_n_o * 2;            // BF16
        const int cd_swap = 64 * (config.block_n / 2) * (4 + 1);                 // FP32 + FP8
        const int cd_base = align(std::max(std::max(cd_l1, cd_l2), cd_swap), 1024);
        // swapAB RF decode: no decoded-B buffer; packed scratch is 64B/row.
        const int per_stage_data = 64 * config.block_k /* A (FP8) */ +
                                   config.block_n * 64 /* packed B  */ +
                                   config.block_n * 4  /* B coeff   */;
        const int sfa_per_stage = 2 * align(64 * 4, 128);
        const int two_chunk_alias = 3 * num_epilogue_warps_o * hidden;
        int stages = std::min(get_env<int>("DG_W4A8_INT_SMALLM_OCC2_STAGES", 4),
                              config.num_stages);
        bool fitted = false;
        for (; stages >= 2; -- stages) {
            const int pre_barrier_base = smem_expert + smem_send + cd_base +
                                         stages * per_stage_data;
            // Mirror the kernel's SMEM_CD_PAD_SIZE (keep 2-chunk combine).
            int cd_pad = 0;
            if (get_env<int>("DG_OCC2_NO_PAD", 0) == 0 and
                hidden % 2 == 0 and hidden <= 2 * 32 * 128 and
                pre_barrier_base < two_chunk_alias and
                two_chunk_alias - pre_barrier_base <= 4096)
                cd_pad = align(two_chunk_alias - pre_barrier_base, 1024);
            const int smem_barriers = (config.num_dispatch_threads / 32 +
                                       3 * stages + 2 * num_epilogue_warps_o) * 8;
            const int total = pre_barrier_base + cd_pad + stages * sfa_per_stage +
                              smem_barriers + zp_prelut_smem_size;
            if (total <= kOcc2SmemBudget) {
                config.num_stages = stages;
                config.smem_size = total;
                fitted = true;
                break;
            }
        }
        if (not fitted)
            occ2 = false;
        if (get_env<int>("DG_JIT_DEBUG") or get_env<int>("DG_PRINT_CONFIGS"))
            printf("W4A8-int small-M OCC2: %s (num_tokens=%d, stages=%d, smem_size=%d)\n",
                   occ2 ? "on" : "off (no fit)", num_tokens, config.num_stages, config.smem_size);
    }

    // Activation descriptors stay FP8. Packed weights use one unswizzled 80B
    // row per logical BK128 tile and are restaged to swizzled FP8 in the CTA.
    constexpr int kGranK = 128;
    constexpr int kL2ActsSFGranK = 64;
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     hidden, config.num_ring_tokens,
                                                     config.block_k, config.block_m,
                                                     static_cast<int>(l1_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf,
                                                        config.sf_ring_stride_tokens, hidden,
                                                        config.block_m, kGranK,
                                                        1, 0);
    const auto tensor_map_l1_weights = make_tma_2d_desc(l1_weights,
                                                        static_cast<int>(l1_weights.size(2)),
                                                        num_experts_per_rank * intermediate_hidden * 2,
                                                        pre_decoded_b ? config.block_k : 64, config.block_n,
                                                        static_cast<int>(l1_weights.stride(-2)),
                                                        pre_decoded_b ? config.swizzle_acts_mode : 0);
    // E8M0 scale planes: tile-major (E, N/BN, K/128, BN, 4) viewed as 2D
    // [tiles, BN*4]; one 512B row per (expert, n_block, k_block) tile.
    // TMA boxDim is capped at 256 per dimension: expose the 512B tile rows as
    // two 256B half-rows and copy them with a (256, 2) box.
    // Fixed 256B rows (TMA boxDim cap); a BN*4-byte tile spans rows_per_tile.
    const int sf_rows_per_tile = config.block_n * 4 / 256;
    const int l1_sf_rows = num_experts_per_rank *
        (intermediate_hidden * 2 / config.block_n) * (hidden / 128) * sf_rows_per_tile;
    const auto tensor_map_l1_weights_sf = make_tma_2d_desc(
        l1_weights_sf, 256, l1_sf_rows,
        256, sf_rows_per_tile, 256, 0);
    const int l2_sf_rows = num_experts_per_rank *
        (hidden / config.block_n) * (intermediate_hidden / 128) * sf_rows_per_tile;
    const auto tensor_map_l2_weights_sf = make_tma_2d_desc(
        l2_weights_sf, 256, l2_sf_rows,
        256, sf_rows_per_tile, 256, 0);

    // L1 output (post-SwiGLU FP8): N is halved. The SM90 epilogue writes this
    // staging tile to SMEM as plain row-major bytes, so the TMA store descriptor
    // must use no shared-memory swizzle. Later L2 TMA loads may still swizzle
    // from this row-major global buffer into their own SMEM tile.
    // The default TMA store is issued per warpgroup, each writing a WG_BLOCK_M
    // row tile. In split-N mode, two WGs produce different N halves of the same
    // M rows, then one TMA store writes the full 64x128 post-SwiGLU tile.
    const int num_epilogue_warpgroups_h = config.num_epilogue_threads / 128;
    const bool split_n_warpgroups_h =
        config.block_m == 64 and num_epilogue_warpgroups_h > 1 and
        config.block_n % num_epilogue_warpgroups_h == 0 and
        (config.block_n / num_epilogue_warpgroups_h == 64 or
         config.block_n / num_epilogue_warpgroups_h == 128);
    const bool split_mn_warpgroups_h =
        config.block_m == 128 and config.block_n == 256 and num_epilogue_warpgroups_h == 4;
    const int wg_split_m = split_n_warpgroups_h ? 1 : (split_mn_warpgroups_h ? 2 : num_epilogue_warpgroups_h);
    const int wg_split_n = split_n_warpgroups_h ? num_epilogue_warpgroups_h : (split_mn_warpgroups_h ? 2 : 1);
    const int wg_block_m = config.block_m / wg_split_m;
    const int wg_block_n = config.block_n / wg_split_n;
    const int wg_l1_out_block_n = wg_block_n / 2;
    const int l1_output_box_n = split_n_warpgroups_h ? config.block_n / 2 : wg_l1_out_block_n;
    const int l1_output_box_m = split_n_warpgroups_h ? config.block_m : wg_block_m;
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts,
                                                       intermediate_hidden, config.num_ring_tokens,
                                                       l1_output_box_n, l1_output_box_m,
                                                       static_cast<int>(l2_acts.stride(-2)),
                                                       0);
    const auto tensor_map_l2_acts = make_tma_2d_desc(l2_acts,
                                                     intermediate_hidden, config.num_ring_tokens,
                                                     config.block_k, config.block_m,
                                                     static_cast<int>(l2_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf,
                                                        config.sf_ring_stride_tokens, intermediate_hidden,
                                                        config.block_m, kL2ActsSFGranK,
                                                        1, 0);
    const auto tensor_map_l2_weights = make_tma_2d_desc(l2_weights,
                                                        static_cast<int>(l2_weights.size(2)),
                                                        num_experts_per_rank * hidden,
                                                        pre_decoded_b ? config.block_k : 64, config.block_n,
                                                        static_cast<int>(l2_weights.stride(-2)),
                                                        pre_decoded_b ? config.swizzle_acts_mode : 0);

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();
    const float* l1_global_scales_ptr = l1_global_scales.has_value() ?
        l1_global_scales->data_ptr<float>() : nullptr;
    const float* l2_global_scales_ptr = l2_global_scales.has_value() ?
        l2_global_scales->data_ptr<float>() : nullptr;

    // Launch
    const auto num_sms = device_runtime->get_num_sms();
    const SM90MXFP4MegaMoERuntime::Args args = {
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .hidden = hidden, .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts, .num_topk = num_topk,
        .num_ranks = num_ranks,
        .activation_clamp = activation_clamp,
        .fast_math = fast_math,
        .direct_l2_scatter = config.direct_l2_scatter,
        .phase_profile = get_env<int>("DG_SM90_MOE_PHASE_PROFILE", 0) != 0,
        .l2_nmajor_schedule = config.l2_nmajor_schedule,
        .one_warp_cleanup = config.one_warp_cleanup,
        .occ2 = occ2,
        .occ2_doubled = occ2 and get_env<int>("DG_W4A8_INT_SMALLM_OCC2_NO2X", 0) == 0,
        .attn_tp_size = attn_tp_size,
        .fuse_router = fuse_router,
        .router_renormalize = router_renormalize,
        .router_logits_input_mode = router_logits_input_mode,
        .tp_distributed_combine = tp_distributed_combine,
        .fuse_input_quant = fuse_input_quant,
        .kernel_phase = SM90MXFP4MegaMoERuntime::KernelPhase::Linear1,
        .config = config,
        .y = tp_distributed_combine ? tp_combine_output->data_ptr() : y.data_ptr(),
        .cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats_ptr,
        .num_tokens = num_tokens,
        .hidden_input = fuse_input_quant ? hidden_input->data_ptr() : nullptr,
        .router_input = router_input.has_value() ? router_input->data_ptr() : nullptr,
        .router_weight = router_weight.has_value() ? router_weight->data_ptr() : nullptr,
        .router_logits_input = router_logits_input_mode ? router_logits->data_ptr() : nullptr,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .tensor_map_l1_acts = tensor_map_l1_acts,
        .tensor_map_l1_acts_sf = tensor_map_l1_acts_sf,
        .tensor_map_l1_weights = tensor_map_l1_weights,
        .tensor_map_l1_weights_sf = tensor_map_l1_weights_sf,
        .l1_global_scales = l1_global_scales_ptr,
        .tensor_map_l1_output = tensor_map_l1_output,
        .tensor_map_l2_acts = tensor_map_l2_acts,
        .tensor_map_l2_acts_sf = tensor_map_l2_acts_sf,
        .tensor_map_l2_weights = tensor_map_l2_weights,
        .tensor_map_l2_weights_sf = tensor_map_l2_weights_sf,
        .l2_global_scales = l2_global_scales_ptr,
        .launch_args = LaunchArgs(((occ2 and get_env<int>("DG_W4A8_INT_SMALLM_OCC2_NO2X", 0) == 0) ? 2 : 1) * num_sms,
                                  config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads,
                                  config.smem_size, config.cluster_size)
    };
    const auto launch_with_phase = [&](const SM90MXFP4MegaMoERuntime::KernelPhase kernel_phase,
                                       const char* kernel_name) {
        auto split_args = args;
        split_args.kernel_phase = kernel_phase;
        const auto code = SM90MXFP4MegaMoERuntime::generate(split_args);
        const auto runtime = compiler->build(kernel_name, code);
        if (split_args.occ2 and split_args.occ2_doubled) {
            // The doubled grid relies on 2 CTAs/SM co-residency: at 1 CTA/SM
            // the second half of the persistent grid never launches and the
            // grid sync deadlocks. Verify occupancy once per loaded kernel.
            static std::unordered_set<void*> verified_kernels;
            const auto handle_key = reinterpret_cast<void*>(runtime->kernel);
            if (verified_kernels.count(handle_key) == 0) {
                const int max_blocks = get_max_active_blocks_per_sm(
                    runtime->kernel, split_args.launch_args.num_threads,
                    split_args.launch_args.smem_size);
                DG_HOST_ASSERT(max_blocks >= 2 and
                               "W4A8-int small-M OCC2 kernel failed 2-CTAs/SM occupancy");
                verified_kernels.insert(handle_key);
            }
        }
        // Whole-grid PDL is disabled for the persistent MegaMoE phases.
        // Allowing L1/L2 to start before the predecessor grid retires caused
        // SM starvation and millisecond slow modes on H20.
        const bool global_pdl = device_runtime->get_pdl();
        if (global_pdl)
            device_runtime->set_pdl(false);
        SM90MXFP4MegaMoERuntime::launch(runtime, split_args);
        if (global_pdl)
            device_runtime->set_pdl(true);
    };

    launch_with_phase(SM90MXFP4MegaMoERuntime::KernelPhase::Linear1, "sm90_mxfp4_mega_moe_l1_impl");
    launch_with_phase(SM90MXFP4MegaMoERuntime::KernelPhase::Linear2, "sm90_mxfp4_mega_moe_l2_impl");
}

} // namespace deep_gemm
