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
        shape_n(shape_n), shape_k(shape_k) {}

    CUTLASS_HOST_DEVICE bool is_valid() const {
        return block_phase != BlockPhase::None;
    }
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

    CUTLASS_DEVICE void fetch_expert_recv_count() {
        // NOTES: each lane caches experts at indices (i * 32 + lane_idx)
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            const auto expert_idx = i * 32 + ptx::get_lane_idx();
            uint64_t value = 0;
            if (expert_idx < kNumExpertsPerRank) {
                do {
                    value = ptx::ld_volatile(workspace.get_expert_recv_count_sum_ptr(expert_idx));
                } while (static_cast<uint32_t>(value >> 32) != kNumSMs * kNumRanks);
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
          uint32_t kNumExpertsPerLane = math::constexpr_ceil_div(kNumExpertsPerRank, 32u),
          uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N,
          uint32_t kNumL2BlockNs = L2_SHAPE_N / BLOCK_N>
struct InterleavedMegaMoEScheduler {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using task_info_t = TaskInfo;

    static constexpr uint32_t kNumScheduleStages = 2;
    static constexpr uint32_t kNumL1WavesDone = 0xffffffffu;

    DG_STATIC_ASSERT(L1_SHAPE_N % BLOCK_N == 0, "Invalid L1 shape");
    DG_STATIC_ASSERT(L2_SHAPE_N % BLOCK_N == 0, "Invalid L2 shape");
    DG_STATIC_ASSERT(L1_SHAPE_K % BLOCK_K == 0, "Invalid L1 K shape");
    DG_STATIC_ASSERT(L2_SHAPE_K % BLOCK_K == 0, "Invalid L2 K shape");
    DG_STATIC_ASSERT(kNumL1BlockNs <= 64, "L1 readiness mask is too small");
    DG_STATIC_ASSERT(kNumL2BlockNs >= kNumL1BlockNs,
                     "Alternating scheduler requires at least as many L2 tasks as L1 tasks");

    const fused_layout::Workspace& workspace;
    Barrier* task_info_full_barriers;
    Barrier* task_info_empty_barriers;
    task_info_t* task_infos;

    uint32_t sched_stage_idx = 0;
    uint32_t sched_phase = 0;
    uint32_t stored_num_tokens_per_expert[kNumExpertsPerLane] = {};
    uint32_t num_total_m_blocks = 0;
    uint32_t num_l1_warmup_waves = 0;

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

    CUTLASS_DEVICE void fetch_expert_recv_count() {
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            const auto expert_idx = i * 32 + ptx::get_lane_idx();
            uint64_t value = 0;
            if (expert_idx < kNumExpertsPerRank) {
                do {
                    value = ptx::ld_volatile(
                        workspace.get_expert_recv_count_sum_ptr(expert_idx));
                } while (static_cast<uint32_t>(value >> 32) != kNumSMs * kNumRanks);
            }
            stored_num_tokens_per_expert[i] = static_cast<uint32_t>(value);
        }
        __syncwarp();

        num_total_m_blocks = get_pool_block_offset(kNumExpertsPerRank);
        const uint32_t num_total_l1_tasks = num_total_m_blocks * kNumL1BlockNs;
        const uint32_t num_total_l1_waves =
            math::ceil_div(num_total_l1_tasks, kNumSMs);
        const uint32_t min_l1_warmup_waves = get_num_l1_warmup_waves(
            num_total_m_blocks, kNumSMs, kNumL1BlockNs, kNumL2BlockNs);
        num_l1_warmup_waves =
            cute::min(min_l1_warmup_waves, num_total_l1_waves);
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

    // Producer-side dynamic claim.  Task counters describe issued work, while
    // the activation loader's acquire waits below the scheduler protect actual
    // data completion.
    CUTLASS_DEVICE task_info_t claim_next_task() {
        while (true) {
            if (num_l1_warmup_waves != kNumL1WavesDone &&
                num_l1_warmup_waves > 0) {
                -- num_l1_warmup_waves;
                const uint32_t task_idx =
                    get_next_task_idx(workspace.get_l1_task_count_ptr());
                if (task_idx >= num_total_m_blocks * kNumL1BlockNs) {
                    num_l1_warmup_waves = kNumL1WavesDone;
                    continue;
                }
                return create_task(
                    BlockPhase::Linear1, task_idx, kNumL1BlockNs,
                    L1_SHAPE_N, L1_SHAPE_K);
            }

            const uint32_t task_idx =
                get_next_task_idx(workspace.get_l2_task_count_ptr());
            if (task_idx >= num_total_m_blocks * kNumL2BlockNs)
                break;

            if (num_l1_warmup_waves != kNumL1WavesDone)
                num_l1_warmup_waves = 1;

            auto task_info = create_task(
                BlockPhase::Linear2, task_idx, kNumL2BlockNs,
                L2_SHAPE_N, L2_SHAPE_K);
            const uint32_t num_required_l1_tasks =
                (task_info.pool_block_idx + 1) * kNumL1BlockNs;
            while (ptx::ld_volatile(workspace.get_l1_task_count_ptr()) <
                   num_required_l1_tasks) {}
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
