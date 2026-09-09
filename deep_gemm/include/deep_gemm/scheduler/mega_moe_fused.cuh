#pragma once

#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/layout/mega_moe_fused.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::fused_sched {

// Computation phase for the current block
enum class BlockPhase : uint32_t {
    None = 0,
    Linear1 = 1,
    Linear2 = 2
};

// Get the minimum number of all-L1 waves that must be issued before L1/L2
// interleaving starts.  An L2 task may only be claimed after every L1 N task
// for its M block has been claimed; this warm-up prevents the scheduler from
// forming a cycle with the data-readiness wait in the L2 activation loader.
CUTLASS_HOST_DEVICE constexpr
int get_num_l1_warmup_waves(
        const int& num_total_m_blocks,
        const int& num_workers,
        const int& num_l1_n_blocks,
        const int& num_l2_n_blocks) {
    if (num_total_m_blocks <= 0)
        return 0;
    const int num_first_l2_wave_m_blocks =
        math::constexpr_ceil_div(num_workers, num_l2_n_blocks);
    const int num_l1_warmup_waves_for_first_l2_wave =
        math::constexpr_ceil_div(
            num_first_l2_wave_m_blocks * num_l1_n_blocks, num_workers);

    const int num_interleave_task_diff_per_m_block =
        num_l1_n_blocks > num_l2_n_blocks ?
            num_l1_n_blocks - num_l2_n_blocks : 0;
    const int num_warmup_waves_for_interleave_schedule =
        math::constexpr_ceil_div(
            num_l1_n_blocks +
                (num_total_m_blocks - 1) * num_interleave_task_diff_per_m_block,
            num_workers) + 1;

    return cute::max(
        num_l1_warmup_waves_for_first_l2_wave,
        num_warmup_waves_for_interleave_schedule);
}

// Dynamic task payload shared by the producer and all consumers in one CTA.
// Keep the layout identical to the current upstream MegaMoE task payload so
// that the SM90 path follows the same scheduling contract even though it uses
// one CTA per task instead of a two-CTA cluster.
struct alignas(16) TaskInfo {
    BlockPhase block_phase;
    uint32_t local_expert_idx;
    uint32_t m_block_idx;
    uint32_t n_block_idx;
    uint32_t pool_block_idx;
    uint32_t valid_m;
    uint32_t shape_n;
    // bits [0, 16): K extent of the whole GEMM; bits [16, 24): K-split index;
    // bits [24, 32): number of K splits of this task (1 == unsplit). Consumers
    // derive their K-block range from `get_k_split_idx()` / `get_num_k_splits()`.
    uint32_t shape_k;

    CUTLASS_HOST_DEVICE
    TaskInfo(): TaskInfo(BlockPhase::None, 0, 0, 0, 0, 0, 0, 0) {}

    CUTLASS_HOST_DEVICE
    TaskInfo(const BlockPhase& block_phase,
             const uint32_t& local_expert_idx,
             const uint32_t& m_block_idx,
             const uint32_t& n_block_idx,
             const uint32_t& pool_block_idx,
             const uint32_t& valid_m,
             const uint32_t& shape_n,
             const uint32_t& shape_k):
        block_phase(block_phase),
        local_expert_idx(local_expert_idx),
        m_block_idx(m_block_idx), n_block_idx(n_block_idx),
        pool_block_idx(pool_block_idx), valid_m(valid_m),
        shape_n(shape_n), shape_k(shape_k | (1u << 24)) {}

    CUTLASS_HOST_DEVICE bool is_valid() const {
        return block_phase != BlockPhase::None;
    }

    CUTLASS_HOST_DEVICE void set_k_split(const uint32_t& k_split_idx, const uint32_t& num_k_splits) {
        shape_k = (shape_k & 0xffffu) | (k_split_idx << 16) | (num_k_splits << 24);
    }
    CUTLASS_HOST_DEVICE uint32_t get_shape_k() const { return shape_k & 0xffffu; }
    CUTLASS_HOST_DEVICE uint32_t get_k_split_idx() const { return (shape_k >> 16) & 0xffu; }
    CUTLASS_HOST_DEVICE uint32_t get_num_k_splits() const { return shape_k >> 24; }

    // Stream-K segments (no consumer reads `shape_n` / the K extent, both are
    // compile-time per phase): `shape_n` carries the K-block range
    // [k_block_begin, k_block_end) and bits [0, 16) of `shape_k` the worker index
    // of the tile's FIRST contributor (this segment's worker == first + split idx).
    CUTLASS_HOST_DEVICE void set_streamk(const uint32_t& k_block_begin, const uint32_t& k_block_end,
                                         const uint32_t& first_worker_idx,
                                         const uint32_t& k_split_idx, const uint32_t& num_k_splits) {
        shape_n = k_block_begin | (k_block_end << 16);
        shape_k = (first_worker_idx & 0xffffu) | (k_split_idx << 16) | (num_k_splits << 24);
    }
    CUTLASS_HOST_DEVICE uint32_t get_k_block_begin() const { return shape_n & 0xffffu; }
    CUTLASS_HOST_DEVICE uint32_t get_k_block_end() const { return shape_n >> 16; }
    CUTLASS_HOST_DEVICE uint32_t get_first_worker_idx() const { return shape_k & 0xffffu; }
};

DG_STATIC_ASSERT(sizeof(TaskInfo) == 32, "Invalid task payload layout");

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t L1_SHAPE_N, uint32_t L1_SHAPE_K,
          uint32_t L2_SHAPE_N, uint32_t L2_SHAPE_K,
          uint32_t kNumExpertsPerRank,
          uint32_t kNumExpertsPerWave,
          uint32_t kNumSMs, uint32_t kNumRanks,
          uint32_t kClusterSize = 2,
          uint32_t kNumExpertsPerLane = math::constexpr_ceil_div(kNumExpertsPerRank, 32u),
          uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N,
          uint32_t kNumL2BlockNs = L2_SHAPE_N / BLOCK_N,
          uint32_t kNumL1BlockKs = L1_SHAPE_K / BLOCK_K,
          uint32_t kNumL2BlockKs = L2_SHAPE_K / BLOCK_K>
struct MegaMoEScheduler {
    DG_STATIC_ASSERT(L1_SHAPE_N % BLOCK_N == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_N % BLOCK_N == 0, "Invalid shape");
    DG_STATIC_ASSERT(L1_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(kNumExpertsPerRank % kNumExpertsPerWave == 0, "Invalid wave config");

    // For 2-CTA clusters, neighbour SMs share the same m_block_idx with adjacent
    // n_block_idx; the asserts below guarantee that pairing is always possible.
    // SM90 / single-CTA paths set kClusterSize = 1 and do not need this.
    DG_STATIC_ASSERT(kClusterSize == 1 or kClusterSize == 2, "Invalid cluster size");
    DG_STATIC_ASSERT(kClusterSize == 1 or kNumSMs % 2 == 0, "Number of SMs must be even for 2-CTA cluster");
    DG_STATIC_ASSERT(kClusterSize == 1 or kNumL1BlockNs % 2 == 0, "L1 N block count must be even for 2-CTA cluster");
    DG_STATIC_ASSERT(kClusterSize == 1 or kNumL2BlockNs % 2 == 0, "L2 N block count must be even for 2-CTA cluster");

    // Arrival counts
    const fused_layout::Workspace& workspace;

    // Scheduler state
    BlockPhase next_phase = BlockPhase::Linear1;

    // Current expert and block indices
    uint32_t current_local_expert_idx = 0;
    uint32_t current_num_tokens = 0;
    uint32_t current_pool_block_offset = 0;
    uint32_t block_idx = 0;
    uint32_t m_block_idx = 0;
    uint32_t n_block_idx = 0;

    // Pre-cached per-expert token counts (filled by `fetch_expert_recv_count`)
    // Layout: `stored_num_tokens_per_expert[i]` holds expert (i * 32 + lane_idx)'s count
    uint32_t stored_num_tokens_per_expert[kNumExpertsPerLane] = {};

    CUTLASS_DEVICE explicit MegaMoEScheduler(const fused_layout::Workspace& workspace): workspace(workspace) {
        block_idx = blockIdx.x;
    }

    CUTLASS_DEVICE uint32_t get_wave_expert_end_idx() const {
        return math::align(current_local_expert_idx + 1, kNumExpertsPerWave);
    }

    CUTLASS_DEVICE uint32_t get_num_tokens(const uint32_t& expert_idx) const {
        uint32_t valid_value;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            valid_value = (expert_idx == i * 32 + ptx::get_lane_idx()) ?
                stored_num_tokens_per_expert[i] : valid_value;
        }
        return ptx::exchange(valid_value, expert_idx % 32);
    }

    // Get pool block offset for a given expert index from a per-lane token count array
    CUTLASS_DEVICE uint32_t get_pool_block_offset(const uint32_t& expert_idx) {
        uint32_t num_blocks = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            if (i * 32 + ptx::get_lane_idx() < expert_idx)
                num_blocks += math::ceil_div(stored_num_tokens_per_expert[i], BLOCK_M);
        }
        return __reduce_add_sync(0xffffffff, num_blocks);
    }

    CUTLASS_DEVICE void advance_expert_idx() {
        current_pool_block_offset += get_current_num_m_blocks();
        current_local_expert_idx += 1;
        current_num_tokens = get_num_tokens(current_local_expert_idx);
    }

    CUTLASS_DEVICE void set_expert_idx(const uint32_t& expert_idx) {
        current_local_expert_idx = expert_idx;
        current_num_tokens = get_num_tokens(expert_idx);
        current_pool_block_offset = get_pool_block_offset(expert_idx);
    }

    CUTLASS_DEVICE uint32_t get_current_pool_block_offset() const {
        return current_pool_block_offset;
    }

    CUTLASS_DEVICE uint32_t get_current_num_m_blocks() const {
        return math::ceil_div(current_num_tokens, BLOCK_M);
    }

    template <bool kDoUMMAAligned = false>
    CUTLASS_DEVICE uint32_t get_valid_m() const {
        const auto m = cute::min(current_num_tokens - m_block_idx * BLOCK_M, BLOCK_M);
        return kDoUMMAAligned ? math::align(m, 16u) : m;
    }

    CUTLASS_DEVICE bool fetch_next_l1_block() {
        const auto wave_end_expert_idx = get_wave_expert_end_idx();
        while (current_local_expert_idx < wave_end_expert_idx) {
            const auto num_m_blocks = get_current_num_m_blocks();
            m_block_idx = block_idx / kNumL1BlockNs;
            if (m_block_idx < num_m_blocks)
                return true;

            // Current expert is fully assigned, move to the next
            block_idx -= num_m_blocks * kNumL1BlockNs;
            advance_expert_idx();
        }
        return false;
    }

    CUTLASS_DEVICE bool fetch_next_l2_block() {
        const auto wave_end_expert_idx = get_wave_expert_end_idx();
        while (current_local_expert_idx < wave_end_expert_idx) {
            const auto num_m_blocks = get_current_num_m_blocks();
            if (block_idx < num_m_blocks * kNumL2BlockNs) {
                m_block_idx = block_idx / kNumL2BlockNs;
                return true;
            }

            // Current expert is fully assigned, move to the next
            block_idx -= num_m_blocks * kNumL2BlockNs;
            advance_expert_idx();
        }
        return false;
    }

    // Core state machine: assigns the next block
    CUTLASS_DEVICE cute::tuple<BlockPhase, uint32_t, uint32_t, uint32_t> get_next_block() {
        while (true) {
            if (current_local_expert_idx >= kNumExpertsPerRank)
                break;

            if (next_phase == BlockPhase::Linear1) {
                if (fetch_next_l1_block()) {
                    // Found a new L1 block
                    n_block_idx = block_idx - m_block_idx * kNumL1BlockNs;
                    // Jump to next block
                    block_idx += kNumSMs;
                    return {BlockPhase::Linear1, current_local_expert_idx, m_block_idx, n_block_idx};
                } else {
                    // L1 for the current wave is complete, transition to L2
                    next_phase = BlockPhase::Linear2;
                    set_expert_idx(math::align<uint32_t, false>(current_local_expert_idx - 1, kNumExpertsPerWave));
                }
            } else {
                if (fetch_next_l2_block()) {
                    // Found a new L2 block
                    n_block_idx = block_idx - m_block_idx * kNumL2BlockNs;
                    // Jump to next block
                    block_idx += kNumSMs;
                    return {BlockPhase::Linear2, current_local_expert_idx, m_block_idx, n_block_idx};
                } else {
                    // Move to L1 of the next wave
                    next_phase = BlockPhase::Linear1;
                }
            }
        }

        // All waves and experts are fully processed
        return {BlockPhase::None, 0, 0, 0};
    }

    // `push_done_ptr` (kPushDoneFlags): wait for the push DONE count to reach
    // `push_done_target` (acquire.sys: every source rank's rows and tickets of this
    // launch are then visible) and take the final low words directly, instead of
    // polling the per-expert completeness high word published by SM e.
    CUTLASS_DEVICE void fetch_expert_recv_count(const int* push_done_ptr = nullptr,
                                                const int& push_done_target = 0) {
        // NOTES: each lane caches experts at indices (i * 32 + lane_idx)
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            const auto expert_idx = i * 32 + ptx::get_lane_idx();
            uint64_t value = 0;
            if (expert_idx < kNumExpertsPerRank) {
                if (push_done_ptr != nullptr) {
                    DG_SPIN_WHILE(ptx::ld_acq_sys(push_done_ptr) - push_done_target < 0, 90001 + 10);
                    value = ptx::ld_volatile(workspace.get_expert_recv_count_sum_ptr(expert_idx));
                } else {
                    DG_SPIN_WHILE(static_cast<uint32_t>(
                        (value = ptx::ld_acq_gpu(workspace.get_expert_recv_count_sum_ptr(expert_idx))) >> 32) !=
                        kNumSMs * kNumRanks, 90001);
                }
            }
            stored_num_tokens_per_expert[i] = static_cast<uint32_t>(value);
        }
        __syncwarp();
    }

    template <typename Func>
    CUTLASS_DEVICE void for_each_block(Func&& func) {
        // Wait for all expert counters to be finalized
        fetch_expert_recv_count();

        // Initialize current expert with 0
        set_expert_idx(0);

        // Iterate over all blocks
        // TODO: add swizzle within expert waves for better L2 cache utilization
        while (true) {
            CUTE_TIE_DECL(get_next_block(), block_phase, current_local_expert_idx, m_block_idx, n_block_idx);
            if (block_phase == BlockPhase::None)
                break;

            func(block_phase, current_local_expert_idx,
                 block_phase == BlockPhase::Linear2 ? kNumL2BlockKs : kNumL1BlockKs,
                 m_block_idx, n_block_idx);
        }
    }

};

// SM90 one-CTA dynamic scheduler.
//
// The weight-loader warp is the producer: it claims globally indexed L1/L2
// tasks and publishes them through a two-stage shared-memory mailbox.  The A
// loader and math warps consume the exact same payload, so task assignment is
// dynamic across SMs without requiring every role to race on a global atomic.
// After a minimal L1-only warm-up, each producer alternates L2 and L1 claims.
template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t L1_SHAPE_N, uint32_t L1_SHAPE_K,
          uint32_t L2_SHAPE_N, uint32_t L2_SHAPE_K,
          uint32_t kNumExpertsPerRank,
          uint32_t kNumSMs, uint32_t kNumRanks,
          // L1 split-K: the L1 tasks of the last, partial L1 wave (the
          // `num_l1_tasks % kNumSMs` highest task indices, i.e. the stragglers
          // that would otherwise serialise a whole extra task length) are each
          // claimed as `kNumL1KSplits` K-range tasks (adjacent task indices, so
          // the halves run concurrently on different SMs). Splitting is only
          // applied when all the halves fit one wave and the launch's pool block
          // count fits the partial-sum scratch (`kMaxSplitKPoolBlocks`);
          // otherwise the launch runs unsplit tasks.
          uint32_t kNumL1KSplits = 1,
          uint32_t kMaxSplitKPoolBlocks = 0xffffffffu,
          uint32_t kNumExpertsPerLane = math::constexpr_ceil_div(kNumExpertsPerRank, 32u),
          uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N,
          uint32_t kNumL2BlockNs = L2_SHAPE_N / BLOCK_N,
          // L2 split-K: same tail rule for the L2 tasks of the last partial L2 wave
          // (`num_l2_tasks % kNumSMs` highest L2 task indices), each claimed as
          // kNumL2KSplits adjacent K-range task indices. Same fit conditions.
          uint32_t kNumL2KSplits = 1,
          // Stream-K (tiny M): see `claim_next_streamk_task`. Replaces the wave
          // scheduler and both tail splits whenever the pool block count fits the
          // scratch (kMaxSplitKPoolBlocks); otherwise the launch falls back to the
          // wave scheduler above.
          bool kStreamK = false,
          // Stream-K unit = one pipeline stage (kKBlocksPerStage K128 blocks): the RF
          // math loop consumes whole stages, so segments must be stage-aligned.
          uint32_t kStreamKKBlocksPerUnit = 2,
          uint32_t kNumL1BlockKs = L1_SHAPE_K / BLOCK_K,
          uint32_t kNumL2BlockKs = L2_SHAPE_K / BLOCK_K,
          // Push dispatch (kernel `kPushDispatch`): the token pool is addressed with
          // a fixed stride of kPushBlocksPerExpert blocks per local expert (the
          // sender picks the row before the destination knows any total), so a
          // task's `pool_block_idx` is expert * stride + m_block instead of the
          // dense prefix-sum block index. Task indices / counts stay dense.
          // 0 == packed (pull) layout.
          uint32_t kPushBlocksPerExpert = 0>
struct InterleavedMegaMoEScheduler {
    DG_STATIC_ASSERT(!kStreamK || (kNumL1BlockKs % kStreamKKBlocksPerUnit == 0 &&
                                   kNumL2BlockKs % kStreamKKBlocksPerUnit == 0),
                     "Stream-K units (stages) must tile both K extents");
    static constexpr uint32_t kNumL1StreamKUnitsPerTask = kNumL1BlockKs / kStreamKKBlocksPerUnit;
    static constexpr uint32_t kNumL2StreamKUnitsPerTask = kNumL2BlockKs / kStreamKKBlocksPerUnit;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using task_info_t = TaskInfo;

    static constexpr uint32_t kNumScheduleStages = 2;
    static constexpr uint32_t kNumL1WavesDone = 0xffffffffu;
    // The split tail adds < 1 wave of L1 task indices; one extra L1-first wave
    // keeps them ahead of the L2 claims (extra L1 waves are deadlock-free).
    static constexpr uint32_t kNumSplitKExtraWarmupWaves = 1;

    DG_STATIC_ASSERT(L1_SHAPE_N % BLOCK_N == 0, "Invalid L1 shape");
    DG_STATIC_ASSERT(L2_SHAPE_N % BLOCK_N == 0, "Invalid L2 shape");
    DG_STATIC_ASSERT(L1_SHAPE_K % BLOCK_K == 0, "Invalid L1 K shape");
    DG_STATIC_ASSERT(L2_SHAPE_K % BLOCK_K == 0, "Invalid L2 K shape");
    DG_STATIC_ASSERT(kNumL1BlockNs <= 64, "L1 readiness mask is too small");
    DG_STATIC_ASSERT(kNumL2BlockNs >= kNumL1BlockNs,
                     "Alternating scheduler requires at least as many L2 tasks as L1 tasks");
    // With split-K the L1 task count per M block (kNumL1BlockNs * kNumL1KSplits)
    // exceeds the L2 count; `get_num_l1_warmup_waves` accounts for the surplus
    // through its per-M-block task difference term.
    DG_STATIC_ASSERT(kNumL1KSplits >= 1 && kNumL1KSplits <= 255, "Invalid L1 K-split count");
    DG_STATIC_ASSERT(kNumL2KSplits >= 1 && kNumL2KSplits <= 255, "Invalid L2 K-split count");
    DG_STATIC_ASSERT(L2_SHAPE_K <= 0xffffu, "L2 K extent must fit the TaskInfo shape_k field");
    DG_STATIC_ASSERT(L1_SHAPE_K <= 0xffffu, "L1 K extent must fit the TaskInfo shape_k field");

    const fused_layout::Workspace& workspace;
    Barrier* task_info_full_barriers;
    Barrier* task_info_empty_barriers;
    task_info_t* task_infos;

    uint32_t sched_stage_idx = 0;
    uint32_t sched_phase = 0;
    uint32_t stored_num_tokens_per_expert[kNumExpertsPerLane] = {};
    uint32_t num_total_m_blocks = 0;
    uint32_t num_l1_warmup_waves = 0;
    // Effective L1 K-split factor for this launch (1 or kNumL1KSplits)
    uint32_t num_l1_k_splits = 1;
    // L1 task indices [0, num_l1_split_base) are unsplit tasks; indices from
    // `num_l1_split_base` on are the K splits of the tail tasks.
    uint32_t num_l1_split_base = 0;
    uint32_t num_total_l1_task_indices = 0;
    // L2 counterpart: effective split factor, first split L2 task, total indices
    uint32_t num_l2_k_splits = 1;
    uint32_t num_l2_split_base = 0;
    uint32_t num_total_l2_task_indices = 0;

    // Stream-K state (kStreamK launches that fit the scratch). Units are (task,
    // K128 block) pairs numbered task-major; worker w owns the contiguous unit
    // range [begin(w), end(w)) of a phase (near-equal: total / kNumSMs each, the
    // first total % kNumSMs workers one more). Each CTA claims exactly one L1
    // worker index (L1 task counter ticket) and then one L2 worker index (L2
    // counter ticket) and publishes its range segment by segment (one segment ==
    // one task with a K-block sub-range). All L1 units are owned by the first
    // ceil(U1 / units-per-worker) tickets, which never wait on L2 progress, so an
    // L2 segment's L1-readiness waits are deadlock-free by induction.
    bool streamk_active = false;
    uint32_t sk_state = 0;  // 0 claim L1 range, 1 in L1 range, 2 claim L2 range, 3 in L2 range, 4 done
    uint32_t sk_worker_idx = 0;
    uint32_t sk_unit = 0, sk_unit_end = 0;
    uint32_t sk_num_l1_units = 0, sk_num_l2_units = 0;

    CUTLASS_DEVICE
    InterleavedMegaMoEScheduler(
            const fused_layout::Workspace& workspace,
            Barrier* task_info_full_barriers,
            Barrier* task_info_empty_barriers,
            task_info_t* task_infos):
        workspace(workspace),
        task_info_full_barriers(task_info_full_barriers),
        task_info_empty_barriers(task_info_empty_barriers),
        task_infos(task_infos) {}

    CUTLASS_DEVICE void advance_schedule_pipeline() {
        sched_stage_idx ^= 1u;
        sched_phase ^= sched_stage_idx == 0;
    }

    CUTLASS_DEVICE uint32_t get_num_tokens(const uint32_t& expert_idx) const {
        uint32_t valid_value = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            valid_value = (expert_idx == i * 32 + ptx::get_lane_idx()) ?
                stored_num_tokens_per_expert[i] : valid_value;
        }
        return ptx::exchange(valid_value, expert_idx % 32);
    }

    CUTLASS_DEVICE uint32_t get_pool_block_offset(const uint32_t& expert_idx) const {
        uint32_t num_blocks = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            if (i * 32 + ptx::get_lane_idx() < expert_idx)
                num_blocks += math::ceil_div(stored_num_tokens_per_expert[i], BLOCK_M);
        }
        return __reduce_add_sync(0xffffffff, num_blocks);
    }

    // `push_done_ptr` (kPushDoneFlags): wait for the push DONE count to reach
    // `push_done_target` (acquire.sys: every source rank's rows and tickets of this
    // launch are then visible) and take the final low words directly, instead of
    // polling the per-expert completeness high word published by SM e.
    CUTLASS_DEVICE void fetch_expert_recv_count(const int* push_done_ptr = nullptr,
                                                const int& push_done_target = 0) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            const auto expert_idx = i * 32 + ptx::get_lane_idx();
            uint64_t value = 0;
            if (expert_idx < kNumExpertsPerRank) {
                if (push_done_ptr != nullptr) {
                    DG_SPIN_WHILE(ptx::ld_acq_sys(push_done_ptr) - push_done_target < 0, 90002 + 10);
                    value = ptx::ld_volatile(workspace.get_expert_recv_count_sum_ptr(expert_idx));
                } else {
                    DG_SPIN_WHILE(static_cast<uint32_t>(
                        (value = ptx::ld_acq_gpu(workspace.get_expert_recv_count_sum_ptr(expert_idx))) >> 32) !=
                        kNumSMs * kNumRanks, 90002);
                }
            }
            stored_num_tokens_per_expert[i] = static_cast<uint32_t>(value);
        }
        __syncwarp();

        num_total_m_blocks = get_pool_block_offset(kNumExpertsPerRank);
        // Stream-K only pays when L1 tasks leave SMs idle (fewer L1 tasks than SMs:
        // H20 M=2, 16-20 tasks). With >= 1 full L1 wave the per-SM stage chain, not
        // idle SMs, bounds the phase, and serialising L1 before L2 (a CTA runs its
        // whole L1 range first) loses the wave scheduler's L1/L2 overlap: H20
        // 2026-09-09 M=8 +10 us, M=16 +20 us kernel end (see the host).
        streamk_active = kStreamK && num_total_m_blocks <= kMaxSplitKPoolBlocks &&
            num_total_m_blocks * kNumL1BlockNs < kNumSMs;
        if (streamk_active) {
            sk_num_l1_units = num_total_m_blocks * kNumL1BlockNs * kNumL1StreamKUnitsPerTask;
            sk_num_l2_units = num_total_m_blocks * kNumL2BlockNs * kNumL2StreamKUnitsPerTask;
        }
        const uint32_t num_l1_full_tasks = num_total_m_blocks * kNumL1BlockNs;
        const uint32_t num_l1_tail_tasks = num_l1_full_tasks % kNumSMs;
        // Split only when the tail's splits fit one wave (otherwise they would
        // form another full wave and merely add per-task fixed cost).
        const bool split_tail = !streamk_active && kNumL1KSplits > 1 &&
            num_total_m_blocks <= kMaxSplitKPoolBlocks &&
            num_l1_tail_tasks > 0 && num_l1_tail_tasks * kNumL1KSplits <= kNumSMs;
        num_l1_k_splits = split_tail ? kNumL1KSplits : 1u;
        num_l1_split_base = split_tail ? num_l1_full_tasks - num_l1_tail_tasks : num_l1_full_tasks;
        num_total_l1_task_indices =
            num_l1_split_base + (num_l1_full_tasks - num_l1_split_base) * num_l1_k_splits;
        const uint32_t num_total_l1_waves =
            math::ceil_div(num_total_l1_task_indices, kNumSMs);
        uint32_t min_l1_warmup_waves = get_num_l1_warmup_waves(
            num_total_m_blocks, kNumSMs, kNumL1BlockNs, kNumL2BlockNs);
        // Split-K tail: claim the (short) halves in the L1 warm-up rather than
        // behind an L2 task on the alternating schedule (M=16: 156 + 8 indices
        // on 78 SMs -> 3 warm-up waves claim them all).
        if (split_tail)
            min_l1_warmup_waves += kNumSplitKExtraWarmupWaves;
        num_l1_warmup_waves =
            cute::min(min_l1_warmup_waves, num_total_l1_waves);

        // L2 split-K tail. L2 indices are claimed in batches: the producers that
        // run out of L1 indices in the last L1 wave (kNumSMs - L1 indices % kNumSMs
        // of them) take the first batch, then ~kNumSMs per batch as SMs free up
        // (the tasks of a batch are dependency-gated on the same L1 wave, so they
        // finish together). The last batch, (num_l2 - first_batch) % kNumSMs tasks
        // (H20 M=8: 22 of 96, not 96 % 78 = 18; M=16: 44 -> 88 halves do not fit,
        // unsplit), is the L2 tail that runs after the last L1 wave; only it is
        // split, and only if its halves fit one wave. The halves are adjacent
        // indices so the finisher is always claimed after its publisher.
        const uint32_t num_l2_full_tasks = num_total_m_blocks * kNumL2BlockNs;
        const uint32_t num_l1_last_wave = num_total_l1_task_indices % kNumSMs;
        const uint32_t num_l2_first_batch = num_l1_last_wave == 0 ? 0u : kNumSMs - num_l1_last_wave;
        const uint32_t num_l2_tail_tasks = num_l2_full_tasks <= num_l2_first_batch ?
            num_l2_full_tasks : (num_l2_full_tasks - num_l2_first_batch) % kNumSMs;
        const bool split_l2_tail = !streamk_active && kNumL2KSplits > 1 &&
            num_total_m_blocks <= kMaxSplitKPoolBlocks &&
            num_l2_tail_tasks > 0 && num_l2_tail_tasks * kNumL2KSplits <= kNumSMs;
        num_l2_k_splits = split_l2_tail ? kNumL2KSplits : 1u;
        num_l2_split_base = split_l2_tail ? num_l2_full_tasks - num_l2_tail_tasks : num_l2_full_tasks;
        num_total_l2_task_indices =
            num_l2_split_base + (num_l2_full_tasks - num_l2_split_base) * num_l2_k_splits;
    }

    // Number of L1 task indices covering the first `num_full_tasks` L1 tasks
    CUTLASS_DEVICE uint32_t get_num_l1_task_indices(const uint32_t& num_full_tasks) const {
        return num_full_tasks <= num_l1_split_base ?
            num_full_tasks :
            num_l1_split_base + (num_full_tasks - num_l1_split_base) * num_l1_k_splits;
    }

    CUTLASS_DEVICE task_info_t create_task(
            const BlockPhase& block_phase,
            const uint32_t& task_idx,
            const uint32_t& num_n_blocks,
            const uint32_t& shape_n,
            const uint32_t& shape_k) const {
        const uint32_t lane_idx = ptx::get_lane_idx();
        const uint32_t pool_block_idx = task_idx / num_n_blocks;
        const uint32_t n_block_idx = task_idx % num_n_blocks;

        task_info_t result(
            block_phase, 0, 0, n_block_idx, pool_block_idx, 0,
            shape_n, shape_k);
        uint32_t block_offset = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            const uint32_t expert_idx = i * 32 + lane_idx;
            const uint32_t num_tokens = stored_num_tokens_per_expert[i];
            const uint32_t num_m_blocks = math::ceil_div(num_tokens, BLOCK_M);
            const uint32_t inclusive_num_m_blocks =
                math::warp_inclusive_sum(num_m_blocks, lane_idx);
            const uint32_t lane_pool_block_offset =
                block_offset + inclusive_num_m_blocks - num_m_blocks;
            const bool is_owner = expert_idx < kNumExpertsPerRank &&
                pool_block_idx >= lane_pool_block_offset &&
                pool_block_idx < lane_pool_block_offset + num_m_blocks;
            const uint32_t owner_mask = __ballot_sync(0xffffffff, is_owner);

            if (owner_mask) {
                const uint32_t owner_lane_idx =
                    static_cast<uint32_t>(__ffs(owner_mask) - 1);
                const uint32_t owner_m_block_idx =
                    pool_block_idx - lane_pool_block_offset;
                const uint32_t owner_valid_m =
                    cute::min(num_tokens - owner_m_block_idx * BLOCK_M, BLOCK_M);
                result.local_expert_idx = ptx::exchange(expert_idx, owner_lane_idx);
                result.m_block_idx = ptx::exchange(owner_m_block_idx, owner_lane_idx);
                result.valid_m = ptx::exchange(owner_valid_m, owner_lane_idx);
                if constexpr (kPushBlocksPerExpert != 0)
                    result.pool_block_idx = result.local_expert_idx * kPushBlocksPerExpert + result.m_block_idx;
            }
            block_offset += ptx::exchange(inclusive_num_m_blocks, 31);
        }
        return result;
    }

    static CUTLASS_DEVICE uint32_t get_next_task_idx(uint32_t* task_count_ptr) {
        uint32_t result = 0;
        if (cute::elect_one_sync())
            result = ptx::atomic_add(task_count_ptr, 1u);
        return ptx::exchange(result, 0);
    }

    // Stream-K partition of `num_units` units over kNumSMs workers.
    static CUTLASS_HOST_DEVICE void get_streamk_range(
            const uint32_t& num_units, const uint32_t& worker_idx,
            uint32_t& begin, uint32_t& end) {
        const uint32_t base = num_units / kNumSMs, rem = num_units % kNumSMs;
        begin = worker_idx * base + cute::min(worker_idx, rem);
        end = begin + base + (worker_idx < rem ? 1u : 0u);
    }

    // Inverse: the worker owning unit `unit_idx` (< num_units)
    static CUTLASS_HOST_DEVICE uint32_t get_streamk_worker_of_unit(
            const uint32_t& num_units, const uint32_t& unit_idx) {
        const uint32_t base = num_units / kNumSMs, rem = num_units % kNumSMs;
        const uint32_t num_long_units = rem * (base + 1);
        return unit_idx < num_long_units ?
            unit_idx / (base + 1) :
            rem + (unit_idx - num_long_units) / base;
    }

    // Next segment of the current range: the task owning unit `sk_unit` and the
    // stage-aligned K-block range of the range's units inside that task.
    template <bool kIsL2>
    CUTLASS_DEVICE task_info_t make_streamk_segment() {
        constexpr uint32_t units_per_task = kIsL2 ? kNumL2StreamKUnitsPerTask : kNumL1StreamKUnitsPerTask;
        const uint32_t num_units = kIsL2 ? sk_num_l2_units : sk_num_l1_units;
        const uint32_t task_idx = sk_unit / units_per_task;
        const uint32_t unit_begin = sk_unit % units_per_task;
        const uint32_t unit_end = cute::min(units_per_task, unit_begin + (sk_unit_end - sk_unit));
        sk_unit += unit_end - unit_begin;
        const uint32_t k_block_begin = unit_begin * kStreamKKBlocksPerUnit;
        const uint32_t k_block_end = unit_end * kStreamKKBlocksPerUnit;
        // Contributors of this tile: the workers owning its first and last unit
        const uint32_t first_worker_idx =
            get_streamk_worker_of_unit(num_units, task_idx * units_per_task);
        const uint32_t last_worker_idx =
            get_streamk_worker_of_unit(num_units, task_idx * units_per_task + units_per_task - 1);
        auto task_info = kIsL2 ?
            create_task(BlockPhase::Linear2, task_idx, kNumL2BlockNs, L2_SHAPE_N, L2_SHAPE_K) :
            create_task(BlockPhase::Linear1, task_idx, kNumL1BlockNs, L1_SHAPE_N, L1_SHAPE_K);
        task_info.set_streamk(k_block_begin, k_block_end, first_worker_idx,
                              sk_worker_idx - first_worker_idx,
                              last_worker_idx - first_worker_idx + 1);
        return task_info;
    }

    CUTLASS_DEVICE task_info_t claim_next_streamk_task() {
        while (true) {
            if (sk_state == 0 || sk_state == 2) {
                const bool is_l2 = sk_state == 2;
                sk_worker_idx = get_next_task_idx(
                    is_l2 ? workspace.get_l2_task_count_ptr() : workspace.get_l1_task_count_ptr());
                if (sk_worker_idx < kNumSMs) {
                    get_streamk_range(is_l2 ? sk_num_l2_units : sk_num_l1_units,
                                      sk_worker_idx, sk_unit, sk_unit_end);
                    sk_state += 1;
                } else {
                    sk_state += 2;
                }
            } else if (sk_state == 1) {
                if (sk_unit < sk_unit_end)
                    return make_streamk_segment<false>();
                sk_state = 2;
            } else if (sk_state == 3) {
                if (sk_unit < sk_unit_end)
                    return make_streamk_segment<true>();
                sk_state = 4;
            } else {
                return task_info_t();
            }
        }
    }

    // Producer-side dynamic claim.  Task counters describe issued work, while
    // the activation loader's acquire waits below the scheduler protect actual
    // data completion.
    CUTLASS_DEVICE task_info_t claim_next_task() {
        if constexpr (kStreamK) {
            if (streamk_active)
                return claim_next_streamk_task();
        }
        while (true) {
            if (num_l1_warmup_waves != kNumL1WavesDone &&
                num_l1_warmup_waves > 0) {
                -- num_l1_warmup_waves;
                const uint32_t task_idx =
                    get_next_task_idx(workspace.get_l1_task_count_ptr());
                if (task_idx >= num_total_l1_task_indices) {
                    num_l1_warmup_waves = kNumL1WavesDone;
                    continue;
                }
                // Split-K tail: index = base + (full_idx - base) * splits + k_split
                if (task_idx < num_l1_split_base)
                    return create_task(
                        BlockPhase::Linear1, task_idx, kNumL1BlockNs,
                        L1_SHAPE_N, L1_SHAPE_K);
                const uint32_t tail_idx = task_idx - num_l1_split_base;
                auto task_info = create_task(
                    BlockPhase::Linear1, num_l1_split_base + tail_idx / num_l1_k_splits,
                    kNumL1BlockNs, L1_SHAPE_N, L1_SHAPE_K);
                task_info.set_k_split(tail_idx % num_l1_k_splits, num_l1_k_splits);
                return task_info;
            }

            const uint32_t task_idx =
                get_next_task_idx(workspace.get_l2_task_count_ptr());
            if (task_idx >= num_total_l2_task_indices)
                break;

            if (num_l1_warmup_waves != kNumL1WavesDone)
                num_l1_warmup_waves = 1;

            // Split-K tail: index = base + (full_idx - base) * splits + k_split
            const bool is_l2_split = task_idx >= num_l2_split_base;
            const uint32_t l2_tail_idx = is_l2_split ? task_idx - num_l2_split_base : 0u;
            const uint32_t l2_full_task_idx =
                is_l2_split ? num_l2_split_base + l2_tail_idx / num_l2_k_splits : task_idx;
            auto task_info = create_task(
                BlockPhase::Linear2, l2_full_task_idx,
                kNumL2BlockNs, L2_SHAPE_N, L2_SHAPE_K);
            if (is_l2_split)
                task_info.set_k_split(l2_tail_idx % num_l2_k_splits, num_l2_k_splits);
            // Dependency gate on the DENSE block index (task_info.pool_block_idx is
            // the strided pool index under push dispatch).
            const uint32_t dense_pool_block_idx = l2_full_task_idx / kNumL2BlockNs;
            const uint32_t num_required_l1_tasks =
                get_num_l1_task_indices((dense_pool_block_idx + 1) * kNumL1BlockNs);
            DG_SPIN_WHILE(ptx::ld_volatile(workspace.get_l1_task_count_ptr()) <
                          num_required_l1_tasks, 90003);
            return task_info;
        }
        return task_info_t();
    }

    CUTLASS_DEVICE void wait_task_slot_empty() const {
        task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1u);
    }

    CUTLASS_DEVICE void publish_task(const task_info_t& task_info) {
        if (cute::elect_one_sync()) {
            task_infos[sched_stage_idx] = task_info;
            __threadfence_block();
            task_info_full_barriers[sched_stage_idx].arrive();
        }
        __syncwarp();
        advance_schedule_pipeline();
    }

    // Consumer-side mailbox read.  Every role has an independent mailbox
    // cursor but observes the same two-stage sequence.
    CUTLASS_DEVICE bool get_published_task(task_info_t& task_info) {
        task_info_full_barriers[sched_stage_idx].wait(sched_phase);
        asm volatile("" ::: "memory");
        task_info = task_infos[sched_stage_idx];
        advance_schedule_pipeline();
        return task_info.is_valid();
    }

    // Called once by each math warp after the first GEMM stage is ready.  At
    // that point the activation loader has necessarily consumed the payload,
    // so the producer may safely recycle the mailbox slot.
    CUTLASS_DEVICE void release_task_info(const uint32_t& lane_idx) const {
        if (lane_idx == 0)
            task_info_empty_barriers[sched_stage_idx ^ 1u].arrive();
    }
};

} // namespace deep_gemm::sched
