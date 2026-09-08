#pragma once

#include <cutlass/arch/barrier.h>

#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe_fused.cuh>

namespace deep_gemm::fused_comm {

CUTLASS_DEVICE void cluster_sync_with_relaxed_arrive() {
    // Perform cluster_sync with `barrier.cluster.arrive.relaxed`
    // This is slightly faster than `cute::cluster_sync` but has weaker memory ordering guarantee
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
}

template <uint32_t kNumSMs, uint32_t kGridSyncIndex = 0, typename sync_scope_t>
CUTLASS_DEVICE void grid_sync(const fused_layout::Workspace& workspace,
                              const uint32_t& sm_idx, const uint32_t& thread_idx,
                              const sync_scope_t& sync_scope) {
    // NOTES: the implementation idea is from `cooperative_groups::this_grid().sync()`
    static constexpr uint32_t kFinishSumTag = 0x80000000u;
    sync_scope();
    if (thread_idx == 0) {
        const auto count_ptr = workspace.get_grid_sync_count_ptr<kGridSyncIndex>();
        const auto old_value = ptx::atomic_add_rel(
            count_ptr, sm_idx == 0 ? (kFinishSumTag - (kNumSMs - 1)) : 1);
        uint32_t new_value;
        do {
            new_value = ptx::ld_acq(count_ptr);
        } while (((new_value ^ old_value) & kFinishSumTag) == 0);
    }
    sync_scope();
}

template <uint32_t kNumRanks, uint32_t kNumSMs, uint32_t kNumThreads, uint32_t kGridSyncIndex, uint32_t kTag, typename sync_scope_t>
CUTLASS_DEVICE void nvlink_barrier(const fused_layout::Workspace& workspace,
                                   const layout::SymBuffer<kNumRanks>& sym_buffer,
                                   const uint32_t& sm_idx, const uint32_t& thread_idx,
                                   const sync_scope_t& sync_scope,
                                   const bool& sync_prologue = true,
                                   const bool& sync_epilogue = true,
                                   // Fast epilogue (requires prologue + epilogue): instead of a
                                   // second grid-wide sync (78 CTAs atomically arrive on one word
                                   // and spin), SM0 publishes completion by storing
                                   // `done_base + done_ordinal` (release.gpu) to the NVLink done
                                   // count after its cross-rank wait, and every other CTA's
                                   // thread 0 spins on that word (acquire.gpu) -> same completion
                                   // condition ("SM0 observed all ranks' signals"), one writer.
                                   // `done_base` is the done count snapshotted by every thread of
                                   // every CTA at kernel start, BEFORE the kernel-start
                                   // __syncthreads that precedes the first barrier's prologue grid
                                   // sync; SM0 only writes the word after that prologue completes
                                   // (all CTAs arrived => all snapshots taken), and the previous
                                   // launch's writes are ordered by the kernel boundary, so all
                                   // CTAs hold the same base. `done_ordinal` is the fixed 1-based
                                   // ordinal of this barrier within the launch (the call sites
                                   // execute in a fixed order on every rank; the existing
                                   // counter/phase scheme already requires that).
                                   const bool& fast_epilogue = false,
                                   const uint32_t& done_base = 0,
                                   const uint32_t& done_ordinal = 0) {
    DG_STATIC_ASSERT(kNumRanks <= kNumThreads, "Insufficient threads");
    DG_DEVICE_ASSERT(!fast_epilogue || (sync_prologue && sync_epilogue));

    // Grid sync before NVLink signaling
    if (sync_prologue)
        grid_sync<kNumSMs, kGridSyncIndex>(workspace, sm_idx, thread_idx, sync_scope);

    // NVLink cross-rank barrier, only SM 0 participates
    if (sm_idx == 0) {
        auto* counter_ptr = workspace.get_nvl_barrier_counter_ptr();
        const auto status = (*counter_ptr) & 3;
        const auto signal_phase = status & 1, signal_sign = status >> 1;
        auto* signal_ptr = workspace.get_nvl_barrier_signal_ptr(signal_phase);

        // Send signals to remote ranks
        if (thread_idx < kNumRanks)
            ptx::red_add_rel_sys(sym_buffer.map(signal_ptr, thread_idx), signal_sign ? -1 : 1);
        sync_scope();

        // Update status and wait arrival (with 30s timeout, at 2 GHz)
        constexpr int64_t kNumTimeoutCycles = 30ll * 2000000000ll;
        if (thread_idx == 0) {
            ptx::red_add(counter_ptr, 1);
            const int target = signal_sign ? 0 : static_cast<int>(kNumRanks);
            const auto start_clock = clock64();
            while (ptx::ld_acq_sys(signal_ptr) != target) {
                if (clock64() - start_clock >= kNumTimeoutCycles) {
#if defined(DG_NVLINK_BARRIER_TRAP_ONLY_TIMEOUT)
                    DG_TRAP_ONLY_DEVICE_ASSERT(false and "NVLink barrier timeout");
#else
                    printf("DeepGEMM NVLink barrier timeout (30s): rank=%d, counter=%d, signal=%d, target=%d, phase=%d, sign=%d, tag=%d\n",
                           sym_buffer.rank_idx, *counter_ptr, ptx::ld_acq_sys(signal_ptr), target, signal_phase, signal_sign, kTag);
                    DG_DEVICE_ASSERT(false and "NVLink barrier timeout");
#endif
                }
            }
            // Fast epilogue: publish completion to the other CTAs of this rank
            if (fast_epilogue && thread_idx == 0)
                ptx::st_rel_gpu(workspace.get_nvl_done_count_ptr(), done_base + done_ordinal);
        }
    }

    if (fast_epilogue) {
        // Non-SM0 CTAs: wait for SM0's completion word; SM0 falls through (it has
        // already observed all ranks). No timeout here: SM0's own wait traps first.
        if (sm_idx != 0 && thread_idx == 0) {
            const auto done_ptr = workspace.get_nvl_done_count_ptr();
            while (static_cast<int32_t>(ptx::ld_acq(done_ptr) - done_base) <
                   static_cast<int32_t>(done_ordinal)) {}
        }
        sync_scope();
        return;
    }

    // Grid sync after NVLink completion
    if (sync_epilogue)
        grid_sync<kNumSMs, kGridSyncIndex>(workspace, sm_idx, thread_idx, sync_scope);
}

} // namespace deep_gemm::comm
