#pragma once

#include <cute/numeric/math.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::fused_layout {

static constexpr int kNumCandidateBlockMs = 7;
static constexpr int kCandidateBlockM[kNumCandidateBlockMs] = {8, 16, 32, 64, 96, 128, 192};
static constexpr int kMaxCandidateBlockM = 192;
static constexpr int kMinCandidateBlockM = 8;
static constexpr int kLCMCandidateBlockM = 384;
static constexpr int kSM90InterleavedSchedulerSMEMBytes = 96;
// SM90 fused MXFP4 BM8 (RF swapAB, 2 K128 blocks per stage): HALF-tile tasks
// (128 of the 256 packed weight rows per task, intra-CTA K-split between the two
// math WGs). Compile-time master gate shared by host (TMA boxes / SF granularity)
// and device; the host additionally selects the mode per launch (env
// DG_FP4_HALF_TILE, default off) through the `kHalfTileTasksRequested` policy.
// -DDG_FUSED_HALF_TILE_TASKS=0 removes the code path entirely.
#ifndef DG_FUSED_HALF_TILE_TASKS
#define DG_FUSED_HALF_TILE_TASKS 1
#endif
static constexpr bool kSM90FusedHalfTileTasks = DG_FUSED_HALF_TILE_TASKS != 0;

// SM90 fused MXFP4 BM8 (RF swapAB, 2 K128 blocks per stage): L1 split-K tasks
// (`kSplitKL1` in the kernel body; host env DG_FP4_SPLITK_L1, default ON for that
// tier). The L1 (expert, n_block) tasks of the last partial L1 wave are claimed as
// two K halves that run on two SMs; K half 0 publishes its fp32 partial sums through a workspace scratch slot
// indexed by (pool_block, n_block) and releases a per-(pool_block, n_block) flag
// that K half 1 acquires before running the epilogue.
// The scratch is bounded: the scheduler only splits when the launch's total pool
// block count is <= kSM90SplitKL1MaxPoolBlocks (M<=16 per rank fits easily; larger
// launches silently fall back to unsplit tasks). Sizes below are fixed by the
// tier: 10 BN256 L1 N-blocks, BM8 tokens, 256 weight rows per tile.
static constexpr uint32_t kSM90SplitKL1MaxPoolBlocks = 64;
static constexpr uint32_t kSM90SplitKL1NumL1BlockNs = 10;
static constexpr uint32_t kSM90SplitKL1NumKSplits = 2;
// One partial per (weight row, token): 256 rows x 8 tokens x fp32.
static constexpr uint32_t kSM90SplitKL1PartialBytes = 256u * 8u * 4u;
static constexpr uint32_t kSM90SplitKL1NumSlots =
    kSM90SplitKL1MaxPoolBlocks * kSM90SplitKL1NumL1BlockNs;
// L2 split-K (`kSplitKL2` in the kernel body; host env DG_FP4_SPLITK_L2): the same
// protocol for the L2 (expert, n_block) tasks of the last partial L2 wave. The L2
// accumulator is also 256 hidden rows x 8 tokens (same slot size); the L2 slots
// live in their own index space (pool_block x 12 BN256 L2 N-blocks) after the L1
// slots, so the L1 and L2 handshakes of one pool block never share a slot.
static constexpr uint32_t kSM90SplitKL2NumL2BlockNs = 12;
static constexpr uint32_t kSM90SplitKL2NumSlots =
    kSM90SplitKL1MaxPoolBlocks * kSM90SplitKL2NumL2BlockNs;
// Total (L1 + L2) split-K slots: flags and fp32 partial scratch are sized by this.
static constexpr uint32_t kSM90SplitKNumSlots = kSM90SplitKL1NumSlots + kSM90SplitKL2NumSlots;
// Fine-grained combine (kernel `kFineCombine`): per-CTA epilogue -> dispatch mailbox
// (see `Workspace::get_combine_mailbox_ptr`).
static constexpr uint32_t kSM90FineCombineMaxSMs = 160;
static constexpr uint32_t kSM90FineCombineMailboxBytes = 32;
static constexpr uint32_t kSM90FineCombineRingSize = 4;
static constexpr uint32_t kSM90FineCombineDoneEntry = 0xffffffffu;
static constexpr uint64_t kSM90SplitKL1ScratchBytes =
    static_cast<uint64_t>(kSM90SplitKNumSlots) * kSM90SplitKL1PartialBytes;  // 11 MB (L1 5 MB + L2 6 MB)

// Pool capacity for shared expert token pool: worst-case total tokens + per-expert BLOCK_M alignment padding, among all possible BLOCK_M
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_max_pool_tokens(T num_ranks, T num_max_tokens_per_rank, T num_topk,
                                                        T num_experts_per_rank) {
    const auto num_max_recv_tokens = num_ranks * num_max_tokens_per_rank;
    const auto num_max_experts_per_token = math::constexpr_min(num_topk, num_experts_per_rank);
    return math::constexpr_align(
        num_max_recv_tokens * num_max_experts_per_token + num_experts_per_rank * (static_cast<T>(kMaxCandidateBlockM) - 1),
        static_cast<T>(kLCMCandidateBlockM));
}

// SF pool capacity: all experts share a contiguous SF region, sized by pool blocks × SF_BLOCK_M
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_padded_sf_pool_tokens(T num_max_pool_tokens, T block_m) {
    return (num_max_pool_tokens / block_m) * math::constexpr_align(block_m, static_cast<T>(128));
}

// Per-token source metadata for combine write-back
struct TokenSrcMetadata {
    uint32_t rank_idx;
    uint32_t token_idx;
    uint32_t topk_idx;
};

struct Workspace {
    void* base;
    uint32_t num_ranks, num_experts;
    uint32_t num_experts_per_rank;
    uint32_t num_max_tokens_per_rank;
    uint32_t num_max_recv_tokens_per_expert;

    // Pool capacity: all local experts share a contiguous token pool
    uint32_t num_max_pool_tokens;
    uint32_t num_max_pool_blocks;

    // Keep grid/NVLink/schedule counters isolated from expert counters.  The
    // interleaved SM90 scheduler places its two hot task counters in separate
    // 32-byte L2 sectors; expert counters start at the next 128-byte cache line.
    static constexpr uint64_t kNumBarrierSignalBytes = 128;

    CUTLASS_HOST_DEVICE
    Workspace(void* base,
              const uint32_t& num_ranks,
              const uint32_t& num_experts,
              const uint32_t& num_max_tokens_per_rank,
              const uint32_t& num_topk):
        base(base),
        num_ranks(num_ranks), num_experts(num_experts),
        num_max_tokens_per_rank(num_max_tokens_per_rank) {
        num_experts_per_rank = num_experts / num_ranks;
        num_max_recv_tokens_per_expert = num_ranks * num_max_tokens_per_rank;
        num_max_pool_tokens = get_num_max_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
        num_max_pool_blocks = num_max_pool_tokens / kMinCandidateBlockM;
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        uint64_t num_bytes = 0;

        // Barrier and in-kernel scheduling counters
        num_bytes += kNumBarrierSignalBytes;

        // Expert send/recv count
        num_bytes += num_experts * sizeof(uint64_t) * 2;

        // Expert recv count sum
        num_bytes += num_experts_per_rank * sizeof(uint64_t);

        // L1 arrival count (padded to even entry count for `uint64_t` alignment of L2 mask)
        num_bytes += math::align(num_max_pool_blocks, 2u) * sizeof(uint32_t);

        // L2 block arrival mask
        num_bytes += num_max_pool_blocks * sizeof(uint64_t);

        // Split-K L1 + L2 ready flags (padded to keep `uint64_t` alignment)
        num_bytes += math::align<uint64_t>(kSM90SplitKNumSlots * sizeof(uint32_t), 8);

        // Fine-grained combine arrival counters, one per local token (padded)
        num_bytes += math::align<uint64_t>(num_max_tokens_per_rank * sizeof(int), 8);

        // Fine-grained combine per-CTA mailboxes (epilogue -> dispatch warp)
        num_bytes += kSM90FineCombineMaxSMs * kSM90FineCombineMailboxBytes;

        // Dispatch pulling source token-topk
        num_bytes += num_experts_per_rank * num_ranks * num_max_recv_tokens_per_expert * sizeof(int);

        // Combine push source indices
        num_bytes += num_max_pool_tokens * sizeof(TokenSrcMetadata);

        // Split-K L1 + L2 fp32 partial-sum scratch (128 B aligned start)
        num_bytes = math::align<uint64_t>(num_bytes, 128);
        num_bytes += kSM90SplitKL1ScratchBytes;

        // Align to TMA descriptor requirements
        num_bytes = math::align<uint64_t>(num_bytes, 16);
        return num_bytes;
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    // Grid sync counters: `kNumBarrierSignalBytes` layout
    // [ 0..15]: 4 x `uint32_t` grid sync counters
    // [16..20]: `uint32_t` NVLink barrier counter
    // [20..27]: 2 x `int` NVLink barrier signals (phase 0 and 1)
    // [28..31]: `uint32_t` L1 schedule task counter
    // [32..35]: `uint32_t` L2 schedule task counter
    // [36..39]: `uint32_t` NVLink barrier completion count (fast epilogue; SM0-written,
    //           never reset: read at kernel start by every CTA as the launch base)
    // [40..127]: padding to isolate hot schedule and expert counters
    static constexpr uint32_t kNumMaxGridSyncCounters = 4;

    template <uint32_t kIndex = 0>
    CUTLASS_DEVICE
    uint32_t* get_grid_sync_count_ptr() const {
        DG_STATIC_ASSERT(kIndex < kNumMaxGridSyncCounters, "Grid sync index out of bounds");
        return static_cast<uint32_t*>(base) + kIndex;
    }

    CUTLASS_DEVICE
    uint32_t* get_nvl_barrier_counter_ptr() const {
        return static_cast<uint32_t*>(base) + kNumMaxGridSyncCounters;
    }

    CUTLASS_DEVICE
    int* get_nvl_barrier_signal_ptr(const uint32_t& phase) const {
        // NOTES: the signal is signed, as we may minus
        return math::advance_ptr<int>(base, (kNumMaxGridSyncCounters + 1) * sizeof(uint32_t) + phase * sizeof(int));
    }

    CUTLASS_DEVICE
    uint32_t* get_l1_task_count_ptr() const {
        return math::advance_ptr<uint32_t>(base, 28u);
    }

    CUTLASS_DEVICE
    uint32_t* get_l2_task_count_ptr() const {
        return math::advance_ptr<uint32_t>(base, 32u);
    }

    CUTLASS_DEVICE
    uint32_t* get_nvl_done_count_ptr() const {
        return math::advance_ptr<uint32_t>(base, 36u);
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_send_count_ptr(const uint32_t& expert_idx = 0) const {
        return math::advance_ptr<uint64_t>(base, kNumBarrierSignalBytes) + expert_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_recv_count_ptr(
        const uint32_t& rank_idx = 0, const uint32_t& expert_idx = 0) const {
        return get_expert_send_count_ptr(num_experts) + rank_idx * num_experts_per_rank + expert_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_recv_count_sum_ptr(const uint32_t& expert_idx = 0) const {
        return get_expert_send_count_ptr(num_experts * 2) + expert_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l1_arrival_count_ptr(const uint32_t& pool_block_idx = 0) const {
        const auto base = get_expert_recv_count_sum_ptr(num_experts_per_rank);
        return reinterpret_cast<uint32_t*>(base) + pool_block_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_l2_arrival_mask_ptr(const uint32_t& pool_block_idx = 0) const {
        // Pad L1 entry count to even so that the `l2_arrival_mask` is 8-byte aligned
        const auto base = get_l1_arrival_count_ptr(math::align(num_max_pool_blocks, 2u));
        return reinterpret_cast<uint64_t*>(base) + pool_block_idx;
    }

    // Split-K L1: ready flag per (pool_block, n_block) (K half 0 -> K half 1).
    // Flag slot index space: [0, kSM90SplitKL1NumSlots) L1, then kSM90SplitKL2NumSlots L2.
    CUTLASS_DEVICE
    uint32_t* get_splitk_l1_flag_ptr(const uint32_t& pool_block_idx = 0, const uint32_t& n_block_idx = 0) const {
        const auto base = get_l2_arrival_mask_ptr(num_max_pool_blocks);
        return reinterpret_cast<uint32_t*>(base) + pool_block_idx * kSM90SplitKL1NumL1BlockNs + n_block_idx;
    }

    // Split-K L2: ready flag per (pool_block, L2 n_block), after the L1 flags
    CUTLASS_DEVICE
    uint32_t* get_splitk_l2_flag_ptr(const uint32_t& pool_block_idx = 0, const uint32_t& n_block_idx = 0) const {
        return get_splitk_l1_flag_ptr(0, 0) + kSM90SplitKL1NumSlots +
            pool_block_idx * kSM90SplitKL2NumL2BlockNs + n_block_idx;
    }

    // Fine-grained combine (kernel `kFineCombine`): per-local-token arrival counter.
    // Every L2 task that scatters a (token, topk-slot) hidden slice into this rank's
    // combine pool does one `red.release.sys.add 1` on the token's counter (over
    // NVLink for remote writers); the combine warp spins until the counter reaches
    // popc(valid topk slots) * (L2 N-blocks per token) and then resets it to 0
    // (the reader owns the reset: all writers of this launch have already arrived,
    // and the next launch's writers only start after its first NVLink barrier,
    // which every CTA of this rank reaches only after this launch has exited).
    CUTLASS_DEVICE
    int* get_combine_arrival_count_ptr(const uint32_t& token_idx = 0) const {
        const auto base = get_splitk_l1_flag_ptr(0, 0) +
            math::align<uint64_t>(kSM90SplitKNumSlots * sizeof(uint32_t), 8) / sizeof(uint32_t);
        return reinterpret_cast<int*>(base) + token_idx;
    }

    // Fine-grained combine per-CTA mailbox (32 B per SM, never reset): the epilogue
    // warps of a CTA post one entry per finished L2 task (packed pool block + valid
    // rows) and the CTA's dispatch warp 0 turns each entry into the sys-scope
    // arrival signals, so the sys-scope fence never stalls the math warps.
    //   [0] producer sequence (st.release.gpu), [1] consumer sequence,
    //   [4..7] 4-entry ring indexed by sequence & 3.
    CUTLASS_DEVICE
    uint32_t* get_combine_mailbox_ptr(const uint32_t& sm_idx) const {
        const auto base = get_combine_arrival_count_ptr(0) +
            math::align<uint64_t>(num_max_tokens_per_rank * sizeof(int), 8) / sizeof(int);
        return reinterpret_cast<uint32_t*>(base) + sm_idx * (kSM90FineCombineMailboxBytes / sizeof(uint32_t));
    }

    // For dispatch pulling
    CUTLASS_DEVICE
    uint32_t* get_src_token_topk_idx_ptr(
        const uint32_t& expert_idx = 0, const uint32_t& rank_idx = 0, const uint32_t& token_idx = 0) const {
        // By value: binding the namespace-scope constexpr to the `const uint32_t&` parameter
        // odr-uses it, which nvcc rejects in device code ("undefined in device code").
        const auto base = get_combine_mailbox_ptr(static_cast<uint32_t>(kSM90FineCombineMaxSMs));
        return reinterpret_cast<uint32_t*>(base) +
            expert_idx * (num_ranks * num_max_recv_tokens_per_expert) +
            rank_idx * num_max_recv_tokens_per_expert + token_idx;
    }

    // For combine usages
    CUTLASS_DEVICE
    TokenSrcMetadata* get_token_src_metadata_ptr(const uint32_t& pool_token_idx = 0) const {
        const auto base = reinterpret_cast<TokenSrcMetadata*>(get_src_token_topk_idx_ptr(num_experts_per_rank));
        return base + pool_token_idx;
    }

    // Split-K L1: fp32 partial scratch [(pool_block, n_block)][kSM90SplitKL1PartialBytes]
    CUTLASS_DEVICE
    float* get_splitk_l1_scratch_ptr(const uint32_t& pool_block_idx, const uint32_t& n_block_idx) const {
        const auto end = reinterpret_cast<uint8_t*>(get_token_src_metadata_ptr(num_max_pool_tokens));
        const auto base = reinterpret_cast<uint8_t*>(
            math::align<uint64_t>(reinterpret_cast<uint64_t>(end) - reinterpret_cast<uint64_t>(this->base), 128) +
            reinterpret_cast<uint64_t>(this->base));
        return reinterpret_cast<float*>(base +
            static_cast<uint64_t>(pool_block_idx * kSM90SplitKL1NumL1BlockNs + n_block_idx) *
                kSM90SplitKL1PartialBytes);
    }

    // Split-K L2: fp32 partial scratch slot (pool_block, L2 n_block), after the L1 slots
    CUTLASS_DEVICE
    float* get_splitk_l2_scratch_ptr(const uint32_t& pool_block_idx, const uint32_t& n_block_idx) const {
        return get_splitk_l1_scratch_ptr(0, 0) +
            (static_cast<uint64_t>(kSM90SplitKL1NumSlots) +
             pool_block_idx * kSM90SplitKL2NumL2BlockNs + n_block_idx) *
                (kSM90SplitKL1PartialBytes / sizeof(float));
    }
};

struct Data {
    uint32_t num_bytes;
    bool require_tma_alignment;
    void* base;

    CUTLASS_HOST_DEVICE
    constexpr explicit Data(
        const uint32_t& num_bytes,
        const bool& require_tma_alignment = true,
        void* base = nullptr) :
        num_bytes(num_bytes), require_tma_alignment(require_tma_alignment), base(base) {
        DG_UNIFIED_ASSERT(num_bytes % 16 == 0 or not require_tma_alignment);
    }

    template <typename dtype_t = uint32_t>
    CUTLASS_HOST_DEVICE constexpr dtype_t get_num_bytes() const {
        return static_cast<dtype_t>(num_bytes);
    }

    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_base_ptr() const {
        return static_cast<dtype_t*>(base);
    }

    CUTLASS_HOST_DEVICE void set_base_ptr(void* ptr) {
        base = ptr;
    }
};

struct Buffer {
    Data data_layout;
    uint32_t num_ranks;
    uint32_t num_max_tokens_per_rank;

    void* base;

    CUTLASS_HOST_DEVICE
    Buffer(const Data& data_layout,
           const uint32_t& num_ranks,
           const uint32_t& max_num_tokens_per_rank,
           void* base = nullptr) :
        data_layout(data_layout),
        num_ranks(num_ranks), num_max_tokens_per_rank(max_num_tokens_per_rank),
        base(base) {}

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes_per_rank() const {
        return num_max_tokens_per_rank * data_layout.get_num_bytes<uint64_t>();
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        return get_num_bytes_per_rank() * num_ranks;
    }

    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_base_ptr() const {
        return static_cast<dtype_t*>(base);
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    CUTLASS_HOST_DEVICE
    Buffer get_rank_buffer(const uint32_t& rank_idx) const {
        return {
            data_layout,
            1, num_max_tokens_per_rank,
            math::advance_ptr(base, get_num_bytes_per_rank() * rank_idx)
        };
    }

    CUTLASS_HOST_DEVICE
    Data get_data_buffer(const uint32_t& token_idx, const bool& global = false) const {
        DG_DEVICE_ASSERT(num_ranks == 1 or global);
        return Data(
            data_layout.num_bytes,
            data_layout.require_tma_alignment,
            math::advance_ptr(base, data_layout.get_num_bytes<uint64_t>() * token_idx)
        );
    }
};

} // namespace deep_gemm::layout
