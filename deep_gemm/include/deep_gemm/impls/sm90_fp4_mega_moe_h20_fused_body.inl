// Independent SM90 NVFP4 MegaMoE fused kernel body.
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900) and (__CUDA_ARCH__ < 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // =====================================================================
    // Template checks
    // =====================================================================
    DG_STATIC_ASSERT(BLOCK_M == 8 || BLOCK_M == 16 ||
                     BLOCK_M == 24 || BLOCK_M == 64 || BLOCK_M == 128,
                     "H200 fused kernel requires BM8/BM16/BM24/BM64/BM128");
    // BM8: 4..7 stages of one K128 block, or 2..4 stages of two K128 blocks
    // (kKBlocksPerStage, checked precisely below once kRFDecode is known).
    DG_STATIC_ASSERT((BLOCK_M == 8 && kNumStages >= 2 && kNumStages <= 7) ||
                     ((BLOCK_M == 16 || BLOCK_M == 24) && kNumStages == 3) ||
                     (BLOCK_M == 64 && kNumStages == 3) ||
                     (BLOCK_M == 128 && kNumStages == 6),
                     "Unexpected H200 pipeline depth");
    DG_STATIC_ASSERT((BLOCK_M == 128) == (BLOCK_N == 128),
                     "BM128 is paired with the BN128 split-M topology");
    DG_STATIC_ASSERT(!kSwapABRequested || BLOCK_M <= 24,
                     "swap-AB is only selected through the M64 bucket");

    // =====================================================================
    // Thread / warp identification
    // =====================================================================
    const uint32_t sm_idx     = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx   = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx   = ptx::get_lane_idx();

    // Optional phase timestamps (globaltimer, ns). Slots:
    //   0 min kernel entry | 1 max dispatch routing done | 2 max dispatch pull done
    //   3 min first math task | 4 max last L1 task end | 5 max last L2 task end
    //   6 max after combine NVLink barrier | 7 max combine end
    const auto stamp_min = [&](const uint32_t slot) {
        if (phase_stamps != nullptr) {
            unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
            atomicMin(phase_stamps + slot, t);
        }
    };
    const auto stamp_max = [&](const uint32_t slot) {
        if (phase_stamps != nullptr) {
            unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
            atomicMax(phase_stamps + slot, t);
        }
    };
    // Steady-state accumulators (SM 0 only): slot 13 += time from entry to
    // after the first NVLink barrier, slot 15 += entry-to-end, slot 14 += 1.
    unsigned long long t_kernel_entry = 0;
    if (phase_stamps != nullptr)
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_kernel_entry));
    const auto stamp_accumulate = [&](const uint32_t slot) {
        if (phase_stamps != nullptr) {
            unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
            atomicAdd(phase_stamps + slot, t - t_kernel_entry);
        }
    };
    if (warp_idx == 0 and cute::elect_one_sync()) {
        stamp_min(0);
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l1_output);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights);
    }

    // =====================================================================
    // Workspaces and symmetric buffer slicing. The framework reserves
    // per-64 SF capacity; this per-128 path uses its first half
    // as a dense per-128 layout so no framework allocation change is needed.
    // =====================================================================
    const auto workspace = fused_layout::Workspace(
        sym_buffer.get_base_ptr(), kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk);

    constexpr auto fp8_token_layout              = fused_layout::Data(kHidden);
    constexpr auto bf16_token_layout             = fused_layout::Data(kHidden * sizeof(nv_bfloat16));
    constexpr auto fp8_intermediate_token_layout = fused_layout::Data(kIntermediateHidden);
    // Per-128 K float SF: 4 bytes per per-128 group => `kHidden / 32` bytes/token (same as SM100 packing)
    constexpr auto fp8_sf_layout                 = fused_layout::Data(kHidden / 32, false);
    // Physical per-64 capacity: logical per-128 scales occupy the first half.
    constexpr auto fp8_intermediate_sf_layout    = fused_layout::Data(kIntermediateHidden / 16);
    constexpr auto input_topk_idx_layout         = fused_layout::Data(kNumTopk * sizeof(int64_t), false);
    constexpr auto input_topk_weights_layout     = fused_layout::Data(kNumTopk * sizeof(float), false);
    constexpr auto l1_topk_weights_layout        = fused_layout::Data(sizeof(float), false);

    // Registered input area
    const auto input_token_buffer        = fused_layout::Buffer(fp8_token_layout, 1, kNumMaxTokensPerRank, workspace.get_end_ptr());
    const auto input_sf_buffer           = fused_layout::Buffer(fp8_sf_layout, 1, kNumMaxTokensPerRank, input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer     = fused_layout::Buffer(input_topk_idx_layout, 1, kNumMaxTokensPerRank, input_sf_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = fused_layout::Buffer(input_topk_weights_layout, 1, kNumMaxTokensPerRank, input_topk_idx_buffer.get_end_ptr());

    // L1 input area
    const auto l1_token_buffer        = fused_layout::Buffer(fp8_token_layout, 1, kNumMaxPoolTokens, input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer           = fused_layout::Buffer(fp8_sf_layout, 1, kNumPaddedSFPoolTokens, l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = fused_layout::Buffer(l1_topk_weights_layout, 1, kNumMaxPoolTokens, l1_sf_buffer.get_end_ptr());

    // L2 input area
    const auto l2_token_buffer = fused_layout::Buffer(fp8_intermediate_token_layout, 1, kNumMaxPoolTokens, l1_topk_weights_buffer.get_end_ptr());
    const auto l2_sf_buffer    = fused_layout::Buffer(fp8_intermediate_sf_layout, 1, kNumPaddedSFPoolTokens, l2_token_buffer.get_end_ptr());

    // Combine input area
    const auto combine_token_buffer = fused_layout::Buffer(bf16_token_layout, kNumTopk, kNumMaxTokensPerRank, l2_sf_buffer.get_end_ptr());

    // =====================================================================
    // GEMM data types and shape constants
    // =====================================================================
    using a_dtype_t = cutlass::float_e4m3_t;
    using b_dtype_t = cutlass::float_e4m3_t;
    using task_info_t = fused_sched::TaskInfo;
    // Half-tile tasks (kHalfTileTasks; gate DG_FUSED_HALF_TILE_TASKS, default ON,
    // BM8 MXFP4 RF swapAB only == the 2-K-block-per-stage path): a task covers 128
    // of the 256 rows of a packed weight tile (sub-tile n_block_idx % 2) over the
    // full K, so task counts double (L1 10 -> 20, L2 12 -> 24 per M block) and the
    // per-task latency halves (80 L1 tasks on 78 SMs no longer cost two full task
    // lengths). Inside the CTA the two math WGs split K instead of N: WG w decodes
    // and multiplies K-block w of every stage on the SAME 128 rows (2 halves of 64,
    // identical register shape), WG1 hands its promoted partial sums to WG0 through
    // smem and WG0 alone runs the epilogue (64 SwiGLU columns for L1, 128 BF16
    // columns for L2). Weight tile layout (256 x 80 B) is unchanged; the B loader
    // fetches the two 10 KB sub-tiles of a stage with two bulk copies.
    // H20 probe (2026-09-09, 8 ranks, 78 SMs): the per-WG K-loop costs ~740 ns per
    // K128 block whatever the grouping (RF decode 349 + RS-WGMMA issue/drain ~390),
    // so a half task is ~9.9 us instead of ~16 us and 160 tasks quantise onto 78
    // SMs as badly as 80 do: M=2 (single wave) kernel end 51 -> 44.5 us, but M=8
    // 65 -> 68.6 and M=16 90 -> 101-103 us. The host therefore keeps it OFF by
    // default (DG_FP4_HALF_TILE=1 enables it).
    constexpr bool kHalfTileTasks =
        fused_layout::kSM90FusedHalfTileTasks && kHalfTileTasksRequested &&
        kSwapABRequested && kMXFP4 && BLOCK_M == 8;
    // N extent of one scheduled task (== the weight tile N unless half-tile tasks).
    constexpr uint32_t TASK_BLOCK_N = kHalfTileTasks ? BLOCK_N / 2 : BLOCK_N;
    // L1 split-K tasks (kSplitKL1; host env DG_FP4_SPLITK_L1, default ON for the BM8
    // MXFP4 RF swapAB tier == the 2-K-block-per-stage path; every other tier is
    // untouched). The L1 (expert, n_block) tasks of the last partial L1 wave (the
    // `num_l1_tasks % kNumSMs` stragglers; M=8: 2 of 80, M=16: 4 of 160) are each
    // scheduled as two adjacent task indices k_half = 0/1 covering K-blocks
    // [12*k_half, 12*k_half + 12) (6 stages), so the two halves run concurrently
    // on two SMs and the straggler wave takes ~half a task length. Splitting every
    // task instead was measured neutral (H20 09-09: 3 waves of ~11.3 us halves ==
    // 2 waves of ~17 us tasks; ~3.8 us fixed cost per task). The per-WG math
    // shape is identical to the unsplit path; `final_accum` already carries the
    // per-K-block activation SF at promote time, so the two partials are directly
    // summable. Reduction (before the L1 epilogue): half 0 publishes its partial to
    // a global scratch slot (fused_layout::Workspace, 8 KB) and releases a
    // per-(pool_block, n_block) flag, skipping the epilogue (no L1-ready notify);
    // half 1 acquires the flag, adds the slot and runs the normal epilogue +
    // notify. The scheduler only splits when the launch's pool
    // block count fits the scratch (kSM90SplitKL1MaxPoolBlocks); L2 is unchanged.
    constexpr bool kSplitKL1 =
        kSplitKL1Requested && kSwapABRequested && (kMXFP4 || kQoQ) && BLOCK_M == 8 &&
        !kHalfTileTasks && kUseInterleavedScheduler && kDenseWeightTiles && BLOCK_N == 256;
    constexpr uint32_t kNumL1KSplits = kSplitKL1 ? fused_layout::kSM90SplitKL1NumKSplits : 1u;
    // L2 half-row tasks (kL2HalfRowTasks; host env DG_FP4_L2_HALFROW, default OFF;
    // only the BM8 MXFP4 RF swapAB tier == the 2-K-block-per-stage path with dense
    // BN256 tiles and the interleaved scheduler can enable it; exclusive with
    // kHalfTileTasks; every other tier is untouched). After L1 split-K the L2 tasks are the critical tail
    // (192 tasks of ~6.5 us quantise badly on 78 SMs: M16 last L2 - last L1 = 11 us).
    // An L2 task covers 128 rows (one 10 KB half of a packed weight tile, the BN128
    // sub-tile addressing of the dense-tile work) instead of 256, and the two math
    // WGs split those 128 rows: WG w decodes and multiplies rows [64w, 64w + 64) for
    // BOTH K-blocks of every stage (one 64-row weight half per WG, 8 RS WGMMAs per
    // stage), keeps its own accumulators and runs its own epilogue for its 64
    // hidden rows (no cross-WG reduction). Per-WG per-stage work halves, so the
    // task takes ~half the time; the L2 task count doubles (12 -> 24 per M block).
    // L1 tasks are unchanged (256 rows, 10 per M block); the L1 -> L2 K-block
    // dependency is unchanged (the L2 task N does not affect its K mapping).
    // H20 probe (2026-09-09): the premise does not hold. The per-WG stage cost is a
    // latency chain (smem load -> LUT -> decode -> RS WGMMA -> drain -> promote), so
    // an L2 task still takes ~6.5 us with 64 rows per WG (same as with 128) and the
    // doubled task count doubles the L2 tail; the ON build also slowed the L1 task
    // (29 vs 22.7 us, register spills). Kept as an OFF-by-default knob (numerics
    // verified); see the host for the numbers.
    constexpr bool kL2HalfRowTasks =
        kL2HalfRowTasksRequested && kSwapABRequested && kMXFP4 && BLOCK_M == 8 &&
        !kHalfTileTasks && kUseInterleavedScheduler && kDenseWeightTiles &&
        BLOCK_N == 256 && kNumEpilogueWarpgroups == 2;
    // N extent of one scheduled L2 task (L1 tasks use TASK_BLOCK_N).
    constexpr uint32_t TASK_BLOCK_N_L2 = kL2HalfRowTasks ? BLOCK_N / 2 : TASK_BLOCK_N;
    // L2 split-K tasks (kSplitKL2; host env DG_FP4_SPLITK_L2): the L2 (expert,
    // n_block) tasks of the last partial L2 wave (`num_l2_tasks % kNumSMs`
    // stragglers; M=8: 18 of 96, M=16: 36 of 192) are each scheduled as two
    // adjacent task indices covering whole 2-K128-block stages: K half 0 =
    // K-blocks [0, 4) (2 stages, PUBLISHER), K half 1 = [4, 10) (3 stages,
    // FINISHER, claimed one index later), so the finisher's flag wait is ~free.
    // Same protocol/scratch format as kSplitKL1 (separate L2 slot index space);
    // the L2 epilogue (BF16 x scale, NVLink scatter) runs on the finisher only.
    // Each half's TMA producer waits only for the L1 bits of its own K-blocks.
    constexpr bool kSplitKL2 =
        kSplitKL2Requested && kSwapABRequested && kMXFP4 && BLOCK_M == 8 &&
        !kHalfTileTasks && !kL2HalfRowTasks && kUseInterleavedScheduler &&
        kDenseWeightTiles && BLOCK_N == 256;
    constexpr uint32_t kNumL2KSplits = kSplitKL2 ? fused_layout::kSM90SplitKL1NumKSplits : 1u;
    // Stream-K (kStreamK; host env DG_FP4_STREAMK, gated by DG_FP4_STREAMK_MAX_M on
    // the global token count): for tiny M the L1 phase is ~16 us per wave of
    // (expert, n_block) tasks over the full K (24 K128 blocks) while most of the 78
    // SMs idle (M=2: 16 tasks) or a 2-task second wave doubles it (M=8: 80 tasks).
    // Instead, all (task, K-block) units of a phase are split into 78 contiguous,
    // near-equal unit ranges (task-major, K inner; scheduler
    // `claim_next_streamk_task`), so every SM streams ~1/78 of the phase's weights
    // and the phase takes ~total_units / 78 K-block times. A range is published as
    // consecutive segments (task + K-block sub-range: the K loop / RS pipeline is
    // unchanged, partial last stages are already supported). Every segment of a
    // tile with > 1 contributor stores its promoted fp32 partial to its worker's
    // per-phase slot and takes an acq_rel ticket on the tile's arrival counter (the
    // split-K flag); the last arriver (ticket == splits - 1) sums the partials in
    // split order (deterministic: fixed order regardless of who finishes) and runs
    // the normal epilogue (L1: SwiGLU + notify; L2: scatter + arrival counters). The
    // L2 activation loader keeps its per-stage wait on the L1 bits of its own
    // K-blocks. Units are whole 2-K-block stages (the RF loop consumes full
    // stages), so only the 2-blocks-per-stage tier qualifies (L2: 10 % 2 == 0).
    // Same tier gate as kSplitKL2 otherwise; the wave scheduler (and its tail
    // splits) remains the fallback when the pool block count exceeds the scratch.
    constexpr bool kStreamK =
        kStreamKRequested && kSwapABRequested && (kMXFP4 || kQoQ) && BLOCK_M == 8 &&
        !kHalfTileTasks && !kL2HalfRowTasks && kUseInterleavedScheduler &&
        kDenseWeightTiles && BLOCK_N == 256 && kKBlocksPerStageRequested == 2;
    // Fast NVLink-barrier epilogue (kNvlFastEpilogue; host env DG_FP4_NVL_FAST_EPI):
    // see fused_comm::nvlink_barrier. Needs the first barrier (before dispatch
    // pull) to have a prologue grid sync, i.e. kDistributedExpertBcast, so that
    // SM0's first write of the done word is ordered after every CTA's kernel-start
    // snapshot of it.
    constexpr bool kNvlFastEpilogue = kNvlFastEpilogueRequested && kDistributedExpertBcast;
    // Fine-grained combine (host env DG_FP4_FINE_COMBINE, default 1): the combine
    // NVLink barrier (#2) is replaced by per-(destination rank, token) arrival
    // counters (`Workspace::get_combine_arrival_count_ptr`). Every L2 task, after its
    // CTA-wide post-scatter sync, red.release.sys-adds 1 per valid token row to the
    // row's destination counter (NVLink for remote destinations); a combine warp
    // spins (ld.acquire.sys) on its token's counter until it equals
    // popc(valid topk slots) * kNumRoutedL2BlockNs, then resets it and reads the
    // partials. The barrier's second job (all local math tasks done before the
    // dispatch warps clean the workspace) moves to a dispatch-warp grid sync, so
    // the combine warps never wait for other CTAs. Pool reuse across launches is
    // protected by NVLink barrier #1 of the next launch (a remote rank only starts
    // scattering after every CTA of every rank has entered that launch) and #3
    // (workspace cleanup), exactly as before.
    constexpr bool kFineCombine = kFineCombineRequested;
    // Push dispatch (kPushDispatch; host env DG_FP4_PUSH_DISPATCH, default 1, gated by
    // DG_FP4_PUSH_DISPATCH_MAX_M on the global token count, default 16). Pull model
    // (default for larger M): after NVLink barrier #1 the dispatch warps pull every
    // received row (TMA over NVLink, ~6.7 us round trip + local store) before the
    // first math task can start. Push model: during routing the SOURCE rank takes a
    // remote atomic ticket on the destination's per-expert count (the same
    // `expert_recv_count_sum` word the broadcast finalises; its low 32 bits now grow
    // by 1 per row, the broadcast adds only the SM-count high word), so the row's
    // position inside the expert is known before any total is, and writes the 3 KB
    // row + per-K128 SF + top-k weight + source metadata straight into the
    // destination's pool with plain 16 B stores. Since the destination's dense
    // prefix-sum pool offsets are unknown to the sender, the pool is addressed with a
    // FIXED stride of kPushBlocksPerExpert blocks per local expert (rows inside an
    // expert stay packed: block e * stride + row / BLOCK_M); the scheduler keeps
    // dense task indices and only remaps `pool_block_idx` (see
    // InterleavedMegaMoEScheduler::create_task). Visibility: the pushes precede the
    // sender's grid sync + SM0's release.sys barrier signal exactly like the top-k
    // index writes of the pull model; after barrier #1 each CTA's dispatch warp 0
    // publishes the arrival counts locally (release.gpu), so the loaders' acquire
    // wait is unchanged (plus a proxy fence before the TMA loads). Pool reuse: a rank
    // can only push launch N+1 rows after NVLink barrier #3 of launch N, which every
    // rank reaches after ALL its math tasks and its workspace cleanup, so no
    // destination still reads launch N's L1 pool / metadata or has counters pending.
    constexpr bool kPushDispatch = kPushDispatchRequested && kUseInterleavedScheduler;
    constexpr uint32_t kPushBlocksPerExpert = kPushDispatch ?
        math::constexpr_ceil_div(kNumRanks * kPushMaxTokensPerRank, BLOCK_M) : 0u;
    DG_STATIC_ASSERT(!kPushDispatch ||
                     kNumExpertsPerRank * kPushBlocksPerExpert * BLOCK_M <= kNumMaxPoolTokens,
                     "Push dispatch: strided pool must fit the token pool");
    DG_STATIC_ASSERT(!kPushDispatch ||
                     kNumExpertsPerRank * kPushBlocksPerExpert <= fused_layout::kSM90SplitKL1MaxPoolBlocks,
                     "Push dispatch: strided pool block indices must fit the split-K / tiny-M slots");
    constexpr uint32_t kNumRoutedL1BlockNs = L1_SHAPE_N / TASK_BLOCK_N;
    constexpr uint32_t kNumRoutedL2BlockNs = L2_SHAPE_N / TASK_BLOCK_N_L2;
    using interleaved_scheduler_t = fused_sched::InterleavedMegaMoEScheduler<
        BLOCK_M, TASK_BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank, kNumSMs, kNumRanks,
        kNumL1KSplits, fused_layout::kSM90SplitKL1MaxPoolBlocks,
        math::constexpr_ceil_div(kNumExpertsPerRank, 32u),
        /* phase-specific task N: L1 256-row tasks, L2 TASK_BLOCK_N_L2-row tasks */
        kNumRoutedL1BlockNs, kNumRoutedL2BlockNs,
        kNumL2KSplits, kStreamK, /* stream-K unit == one 2-K-block stage */ 2u,
        L1_SHAPE_K / BLOCK_K, L2_SHAPE_K / BLOCK_K, kPushBlocksPerExpert>;
    constexpr bool kSplitMDecodedWeightReuse =
        BLOCK_M == 128 && BLOCK_N == 128 && kNumEpilogueWarpgroups == 2;
    constexpr uint32_t WG_BLOCK_M =
        kSplitMDecodedWeightReuse ? BLOCK_M / 2 : BLOCK_M;
    constexpr uint32_t WG_BLOCK_N =
        kSplitMDecodedWeightReuse ? BLOCK_N : BLOCK_N / 2;
    constexpr uint32_t L1_OUT_BLOCK_N = TASK_BLOCK_N / 2;  // post-SwiGLU task N
    constexpr uint32_t WG_L1_OUT_BLOCK_N = WG_BLOCK_N / 2; // post-SwiGLU per-WG N
    constexpr uint32_t kSwapABTokenChunks = BLOCK_M / 8;
    constexpr uint32_t kSwapABWeightHalves = WG_BLOCK_N / 64;
    constexpr uint32_t kSwapABHalfAccumPerThread = 64 * 64 / 128;
    // Rows per WG of an L2 task: 64 (one weight half) with L2 half-row tasks,
    // otherwise WG_BLOCK_N (two halves). L1 always uses WG_BLOCK_N.
    constexpr uint32_t L2_WG_BLOCK_N = kL2HalfRowTasks ? 64u : WG_BLOCK_N;
    DG_STATIC_ASSERT(!kSwapABRequested || WG_L1_OUT_BLOCK_N == 64,
                     "swapAB expects BN256 split-N with 64 L1 output columns per WG");
    // Both dispatch warps participate in CTA-wide barriers. Selected plans may
    // use one warp for routing and token pulls, leaving the other warp's send
    // buffer available for an additional GEMM stage.
    constexpr uint32_t kNumActiveDispatchWarps =
        kSingleActiveDispatchWarp ? 1u : kNumDispatchWarps;
    constexpr uint32_t kNumActiveDispatchThreads = kNumActiveDispatchWarps * 32;
    constexpr bool kQuadDequantIlp = BLOCK_M == 8;
    // QoQ W4A8: int8 activations, (code - z) int8 weights, int32 IGMMA
    // accumulators promoted per K128 by s2[row] * s_act[token]; per-row s1 is
    // applied through the same [E, N] epilogue scale path as MXFP4's 2^e_ref.
    constexpr bool kPerRowEpilogueScale = kMXFP4 || kQoQ;
    DG_STATIC_ASSERT(!kQoQ || kSwapABRequested,
                     "QoQ currently supports the swapAB (<= 64 tokens per rank) tiers only");
    DG_STATIC_ASSERT(!(kQoQ && kSwapPipelineDecode),
                     "QoQ uses the serial swapAB main loop");
    using swap_accum_t = std::conditional_t<kQoQ, int32_t, float>;
    // RF decode (swapAB MXFP4 and QoQ tiers): each thread decodes its own WGMMA A
    // fragment straight from the packed-B rows (host stores MXFP4/QoQ rows in RF
    // fragment order, see `store_decoded_quad`) and issues RS-form WGMMAs.
    // No decoded SMEM tile, no per-stage STS, no per-WG decode barrier. QoQ uses
    // the int8 RS atoms (int32 accumulators, s2[row] * s_act[token] promote).
    DG_STATIC_ASSERT(!(kMXFP4 && kQoQ), "MXFP4 and QoQ are exclusive");
    constexpr bool kRFDecode = kSwapABRequested && (kMXFP4 || kQoQ);
    // Tiny-M CUDA-core GEMV (kTinyMGemv; host env DG_FP4_TINYM, gated by
    // DG_FP4_TINYM_MAX_M): the math warps stream the dense weight tiles with plain
    // 16 B loads and HFMA2 (MXFP4) / DP4A (QoQ) instead of the TMA + RS-WGMMA task
    // pipeline; the loader warps and the task scheduler idle. Stream-K partition per
    // phase, cross-CTA fp32 fixup through the split-K scratch slots. Communication
    // and epilogue contracts are unchanged (docs/tinym_gemv_design.md).
    constexpr bool kTinyMGemv =
        kTinyMGemvRequested && kRFDecode && BLOCK_M == 8 && BLOCK_N == 256 &&
        kDenseWeightTiles && kUseInterleavedScheduler && !kHalfTileTasks && !kL2HalfRowTasks;
    DG_STATIC_ASSERT(!kTinyMGemv || (!kSplitKL1 && !kSplitKL2 && !kStreamK),
                     "TinyM GEMV owns the split-K scratch; the host disables split-K / stream-K");
    DG_STATIC_ASSERT(!(kRFDecode && kSwapPipelineDecode),
                     "RF decode is only implemented for the serial swapAB main loop");
    // K128 blocks per pipeline stage. The tiny-M (BM8) RF swapAB path carries
    // several consecutive K128 blocks per stage (host knob, 2 or 4): H20 probes
    // put the fixed per-stage skeleton (mbarrier check + wgmma drain + arrive/loop)
    // at ~540 ns of a ~740 ns single-block stage, so amortising it over more
    // K-blocks per stage cuts the per-K128 cost (1 -> 2 blocks: 741 -> ~620 ns).
    // Every other path keeps one K128 block per stage.
    DG_STATIC_ASSERT(kKBlocksPerStageRequested == 2 || kKBlocksPerStageRequested == 4,
                     "BM8 RF path supports 2 or 4 K128 blocks per stage");
    constexpr uint32_t kKBlocksPerStage = (kRFDecode && BLOCK_M == 8) ? kKBlocksPerStageRequested : 1u;
    DG_STATIC_ASSERT(BLOCK_M != 8 ||
                     (kKBlocksPerStage == 4 ? (kNumStages == 2) :
                      kKBlocksPerStage == 2 ? (kNumStages >= 2 && kNumStages <= 4)
                                            : (kNumStages >= 4 && kNumStages <= 7)),
                     "BM8 pipeline depth: 4..7 (1 K-block/stage), 2..4 (2 K-blocks/stage) or 2 (4 K-blocks/stage)");
    // Loaders and math step `kKBlocksPerStage` K-blocks per stage; the last stage
    // of a task may be PARTIAL (min(kKBlocksPerStage, remaining) blocks: L2 1280/128
    // = 10 blocks -> 2 x 4 + 2 with 4 blocks per stage). The RF main loop keeps
    // "frag[0] holds block 0 of a stage", which needs every task K-block count to be
    // even (L1 24 / 12 per split-K half, L2 10 / 4 + 6 per split-K half).
    DG_STATIC_ASSERT((L1_SHAPE_K / BLOCK_K) % 2 == 0 && (L2_SHAPE_K / BLOCK_K) % 2 == 0 &&
                     ((L1_SHAPE_K / BLOCK_K) / 2) % kKBlocksPerStage == 0,
                     "Even K-block counts (L1 whole stages) are required");
    DG_STATIC_ASSERT(kKBlocksPerStage == 1 || !kSplitMDecodedWeightReuse,
                     "Multi-K-block stages are not implemented for the BM128 split-M path");
    DG_STATIC_ASSERT(!kHalfTileTasks ||
                     (kRFDecode && kKBlocksPerStage == 2 && kNumEpilogueWarpgroups == 2 &&
                      BLOCK_N == 256 && kDenseWeightTiles),
                     "Half-tile tasks: RF decode, 2 K-blocks per stage (one per WG), dense BN256 tiles");
    DG_STATIC_ASSERT(!kL2HalfRowTasks ||
                     (kRFDecode && kKBlocksPerStage == 2 && TASK_BLOCK_N == 256 &&
                      L2_SHAPE_N % TASK_BLOCK_N_L2 == 0 && L2_WG_BLOCK_N == 64),
                     "L2 half-row tasks: RF decode, 2 K-blocks per stage, full-tile L1 tasks, 128-row L2 tasks");
    DG_STATIC_ASSERT(!kSplitKL1 ||
                     (kRFDecode && kKBlocksPerStage >= 2 && kNumEpilogueWarpgroups == 2 &&
                      ((L1_SHAPE_K / BLOCK_K) % (kNumL1KSplits * kKBlocksPerStage)) == 0 &&
                      L1_SHAPE_N / TASK_BLOCK_N == fused_layout::kSM90SplitKL1NumL1BlockNs),
                     "Split-K L1: RF decode, multi-K-block stages, K halves of whole stages, 10 L1 N-blocks");
    DG_STATIC_ASSERT(!kSplitKL2 ||
                     (kRFDecode && kKBlocksPerStage >= 2 && kNumEpilogueWarpgroups == 2 &&
                      (L2_SHAPE_K / BLOCK_K) >= 2 * kKBlocksPerStage &&
                      L2_SHAPE_N / TASK_BLOCK_N_L2 == fused_layout::kSM90SplitKL2NumL2BlockNs &&
                      L2_WG_BLOCK_N == WG_BLOCK_N),
                     "Split-K L2: RF decode, multi-K-block stages, >= 2 whole stages, 12 L2 N-blocks, full-row L2 tasks");
    using L1WGMMA = typename mma::sm90::FP8MMASelector<WG_BLOCK_N>::type;
    static_assert(L1WGMMA::M == 64 and L1WGMMA::N == WG_BLOCK_N and L1WGMMA::K == 32,
                  "Unexpected WGMMA shape");
    // A and B are CTA-local in the fixed cluster-size-one plan.
    constexpr uint32_t LOAD_BLOCK_M    = BLOCK_M;
    constexpr uint32_t LOAD_BLOCK_N    = BLOCK_N;
    constexpr uint32_t kSwizzleAMode   = BLOCK_K * sizeof(a_dtype_t);   // 128
    // Half-tile tasks publish 64 L1 output columns per task, each with its own
    // per-token scale (like the BN128 split-M tier).
    constexpr uint32_t kL2ActsSFGranK =
        (kSplitMDecodedWeightReuse || kHalfTileTasks) ? 64u : 128u;
    DG_STATIC_ASSERT(kSplitMDecodedWeightReuse || kHalfTileTasks ||
                     WG_L1_OUT_BLOCK_N < kL2ActsSFGranK,
                     "split-N warpgroups must share one L2 activation scale");
    // L1 -> L2 data dependency is per block: L1 N-block `n` (gate/up
    // interleaved) publishes L1-output columns [n * L1_OUT_BLOCK_N, +L1_OUT_BLOCK_N)
    // plus the matching per-token activation scales, which is exactly what L2
    // K-block `n / kNumL1BlocksPerL2KBlock` consumes. The L2 A-loader waits per
    // K-block on the readiness bits of only those L1 N-blocks.
    constexpr uint32_t kNumL2KBlocks = L2_SHAPE_K / BLOCK_K;
    constexpr uint32_t kNumL1BlocksPerL2KBlock = BLOCK_K / L1_OUT_BLOCK_N;
    DG_STATIC_ASSERT(BLOCK_K % L1_OUT_BLOCK_N == 0 &&
                     kNumRoutedL1BlockNs == kNumL2KBlocks * kNumL1BlocksPerL2KBlock &&
                     kNumRoutedL1BlockNs <= 64,
                     "L1 output N-blocks must tile the L2 K dimension exactly");
    DG_STATIC_ASSERT(L1_OUT_BLOCK_N % kL2ActsSFGranK == 0,
                     "L2 activation scale groups must not straddle L1 output blocks");

    // =====================================================================
    // Shared memory layout
    // =====================================================================
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];

    constexpr uint32_t SMEM_EXPERT_COUNT_SIZE =
        math::constexpr_align<uint32_t>(kNumExperts * sizeof(uint32_t), kSharedMemoryAlignment);
    constexpr uint32_t SMEM_SEND_BUFFER_SIZE =
        math::constexpr_align(fp8_token_layout.get_num_bytes() * kNumActiveDispatchWarps, kSharedMemoryAlignment);
    constexpr uint32_t SMEM_NVFP4_LUT_SIZE =
        math::constexpr_align<uint32_t>(128u * sizeof(uint2), kSharedMemoryAlignment);
    // RF decode: per-lane replicated LUT (128 entries x 32 lanes x 8 B = 32 KB).
    // Lane l reads entry i at [i*32 + l], so a warp's 32 gathers always hit 32
    // distinct 8 B slots of one 256 B row -> bank-conflict-free regardless of i.
    constexpr uint32_t SMEM_RF_LUT_REP_SIZE = 0u;  // replicated LUT measured: no gain (bank conflicts not the limiter)
    // One BM x K128 FP8 A tile (128 B-swizzled rows). With two K-blocks per stage
    // the tiles sit back to back; each is 1 KB (BM8) so the 1024 B swizzle
    // period and the WGMMA descriptor base alignment are preserved.
    constexpr uint32_t SMEM_A_SIZE_PER_KBLOCK = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = kKBlocksPerStage * SMEM_A_SIZE_PER_KBLOCK;
    DG_STATIC_ASSERT(kKBlocksPerStage == 1 || SMEM_A_SIZE_PER_KBLOCK % 1024 == 0,
                     "Per-K-block A tiles must keep the 1024 B swizzle period");
    // RF decode feeds WGMMA A straight from registers; no decoded-B tile.
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE =
        kRFDecode ? 0u : LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
    // BM128 split-M alternates two decoded-B slots. This lets one WG begin
    // decoding K+1 after its K WGMMA completes without overwriting the slot
    // that the paired WG may still be consuming.
    // swapAB BM8 (tiny-M): every WG decodes exactly the 128 rows it consumes
    // and drains its WGMMAs (wait<0>) before decoding the next stage, so two
    // decoded slots suffice; this frees ~64 KB for deeper packed-B prefetch.
    constexpr bool kDoubleBufferDecodedB =
        kSplitMDecodedWeightReuse || (kSwapABRequested && BLOCK_M == 8);
    constexpr uint32_t kNumDecodedBStages =
        kDoubleBufferDecodedB ? 2u : kNumStages;
    constexpr uint32_t B_LOAD_BYTES_PER_ROW = 80u;
    // Half-tile tasks: a K-block of a task is one 128-row (10 KB) sub-tile.
    constexpr uint32_t SMEM_PACKED_B_SIZE_PER_KBLOCK =
        TASK_BLOCK_N * B_LOAD_BYTES_PER_ROW * sizeof(b_dtype_t);
    constexpr uint32_t SMEM_PACKED_B_SIZE_PER_STAGE =
        kKBlocksPerStage * SMEM_PACKED_B_SIZE_PER_KBLOCK;
    // Dense weight tiles: one (BLOCK_N x 80 B) tile == exactly one packed-B stage,
    // fetched with a single `cp.async.bulk` (needs 16 B size/address alignment).
    DG_STATIC_ASSERT(!kDenseWeightTiles || (kMXFP4 || kQoQ),
                     "Dense weight tiles are only packed by the MXFP4/QoQ hosts");
    DG_STATIC_ASSERT(SMEM_PACKED_B_SIZE_PER_STAGE % 16 == 0, "Bulk copy size must be 16 B aligned");
    DG_STATIC_ASSERT(SMEM_PACKED_B_SIZE_PER_KBLOCK == TASK_BLOCK_N * 80u, "Unexpected packed-B K-block size");
    // Host packer tile height (rows); a BLOCK_N < 256 kernel tile is a contiguous
    // BLOCK_N*80 B slice of the 256*80 B packed tile.
    constexpr uint32_t kPackedTileN = 256u;
    constexpr uint32_t kPackedTileBytes = kPackedTileN * B_LOAD_BYTES_PER_ROW;
    DG_STATIC_ASSERT(kPackedTileN % TASK_BLOCK_N == 0, "Task BLOCK_N must divide the packed tile height");
    constexpr uint32_t kSubTilesPerPacked = kPackedTileN / TASK_BLOCK_N;
    // L2 tasks: with L2 half-row tasks a K-block is one 128-row (10 KB) sub-tile
    // (the upper/lower half of the 20 KB packed tile); the stage slot is sized for
    // L1 (2 x 20 KB) and L2 simply uses its first 2 x 10 KB.
    constexpr uint32_t kL2SubTilesPerPacked = kPackedTileN / TASK_BLOCK_N_L2;
    constexpr uint32_t SMEM_PACKED_B_L2_SIZE_PER_KBLOCK = TASK_BLOCK_N_L2 * B_LOAD_BYTES_PER_ROW;
    DG_STATIC_ASSERT(SMEM_PACKED_B_L2_SIZE_PER_KBLOCK <= SMEM_PACKED_B_SIZE_PER_KBLOCK &&
                     SMEM_PACKED_B_L2_SIZE_PER_KBLOCK % 16 == 0,
                     "L2 packed-B K-block must fit the L1-sized stage slot (16 B aligned)");
    // Two K-blocks per stage are fetched with ONE bulk copy, which needs the
    // consecutive k tiles of an (expert, n_block) to be contiguous: dense layout
    // with the kernel tile == the packed tile (BN256).
    // (Half-tile tasks fetch the two 10 KB sub-tiles with two bulk copies instead.)
    DG_STATIC_ASSERT(kKBlocksPerStage == 1 ||
                     (kDenseWeightTiles && (kSubTilesPerPacked == 1 || kHalfTileTasks)),
                     "Multi-K-block stages need contiguous dense BN256 weight tiles");
    // L1 and L2 each consume one per-128 activation scale per row and K tile.
    constexpr uint32_t kL2SFAHalfStride =
        math::constexpr_align<uint32_t>(BLOCK_M * sizeof(float), 128u) / sizeof(float);
    constexpr uint32_t kNumL2SFAGroups =
        (kSplitMDecodedWeightReuse || kHalfTileTasks) ? 2u : 1u;
    // One kL2SFAHalfStride slot per (K-block, SF group).
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE =
        kKBlocksPerStage * kNumL2SFAGroups * kL2SFAHalfStride * sizeof(float);
    // CD output: max of L1 FP8 (BLOCK_M * (BLOCK_N/2) * 1 byte * num_wg) and
    // L2 BF16 (BLOCK_M * BLOCK_N * 2 bytes * num_wg).
    constexpr uint32_t SMEM_CD_L1_SIZE =
        kNumEpilogueWarpgroups * WG_BLOCK_M * WG_L1_OUT_BLOCK_N * sizeof(cutlass::float_e4m3_t);
    constexpr uint32_t SMEM_CD_L2_SIZE = kSwapABRequested ?
        BLOCK_M * BLOCK_N * sizeof(nv_bfloat16) : 0u;
    constexpr uint32_t SMEM_CD_OUTPUT_BASE_SIZE =
        SMEM_CD_L1_SIZE > SMEM_CD_L2_SIZE ? SMEM_CD_L1_SIZE : SMEM_CD_L2_SIZE;
    constexpr uint32_t SMEM_CD_L1_SHARED_SF_SLOTS =
        kNumEpilogueWarpgroups * BLOCK_M;
    constexpr uint32_t SMEM_CD_L1_SWAP_AMAX_SLOTS = kSwapABRequested ?
        BLOCK_M * kNumEpilogueWarps : 0u;
    constexpr uint32_t SMEM_CD_L1_EXTRA_FLOAT_SLOTS =
        SMEM_CD_L1_SHARED_SF_SLOTS > SMEM_CD_L1_SWAP_AMAX_SLOTS ?
        SMEM_CD_L1_SHARED_SF_SLOTS : SMEM_CD_L1_SWAP_AMAX_SLOTS;
    constexpr uint32_t SMEM_CD_L1_SHARED_SF_SIZE =
        SMEM_CD_L1_EXTRA_FLOAT_SLOTS * sizeof(float);
    constexpr uint32_t SMEM_CD_OUTPUT_UNALIGNED_SIZE =
        SMEM_CD_OUTPUT_BASE_SIZE + SMEM_CD_L1_SHARED_SF_SIZE;
    constexpr uint32_t SMEM_CD_SIZE = math::constexpr_align(
        SMEM_CD_OUTPUT_UNALIGNED_SIZE, kSharedMemoryAlignment);
    // Half-tile tasks: WG1 -> WG0 accumulator hand-off, [element][128 threads] floats
    // (the swapAB accumulators a thread actually uses: 2 halves x token chunks x 4).
    constexpr uint32_t kKSplitReduceElems = kSwapABWeightHalves * kSwapABTokenChunks * 4u;
    constexpr uint32_t SMEM_KSPLIT_REDUCE_SIZE =
        kHalfTileTasks ? 128u * kKSplitReduceElems * static_cast<uint32_t>(sizeof(float)) : 0u;

    constexpr uint32_t SMEM_BEFORE_BARRIER_SIZE =
        SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE + SMEM_NVFP4_LUT_SIZE + SMEM_RF_LUT_REP_SIZE +
        SMEM_CD_SIZE +
        kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_PACKED_B_SIZE_PER_STAGE) +
        kNumDecodedBStages * SMEM_B_SIZE_PER_STAGE;

    // SMEM pointers
    auto smem_expert_count = reinterpret_cast<uint32_t*>(smem_buffer);
    const auto smem_send_buffers = fused_layout::Buffer(
        fp8_token_layout, kNumActiveDispatchWarps, 1,
        math::advance_ptr(smem_buffer, SMEM_EXPERT_COUNT_SIZE));
    auto smem_nvfp4_lut = reinterpret_cast<uint2*>(math::advance_ptr<uint8_t>(
        smem_buffer, SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE));

    auto smem_rf_lut = reinterpret_cast<uint2*>(math::advance_ptr<uint8_t>(
        smem_buffer, SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE + SMEM_NVFP4_LUT_SIZE));
    auto smem_gemm_base = math::advance_ptr(
        smem_buffer, SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE + SMEM_NVFP4_LUT_SIZE +
                     SMEM_RF_LUT_REP_SIZE);

    auto smem_cd_base = smem_gemm_base;
    // CD output is shared by L1 (FP8) and L2 (BF16); reinterpret-cast as needed.
    auto smem_cd_l1 = reinterpret_cast<cutlass::float_e4m3_t*>(smem_cd_base);
    auto smem_cd_l1_shared_sf =
        math::advance_ptr<float>(smem_cd_base, SMEM_CD_OUTPUT_BASE_SIZE);
    auto smem_cd_l2 = reinterpret_cast<nv_bfloat16*>(smem_cd_base);

    auto smem_a = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<a_dtype_t>(smem_gemm_base, SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([=](const uint32_t& i) {
        const uint32_t decoded_stage =
            kDoubleBufferDecodedB ? (i & 1u) : i;
        return math::advance_ptr<b_dtype_t>(
            smem_gemm_base,
            SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE +
            decoded_stage * SMEM_B_SIZE_PER_STAGE);
    });
    auto smem_packed_b = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<b_dtype_t>(
            smem_gemm_base, SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE +
            kNumDecodedBStages * SMEM_B_SIZE_PER_STAGE +
            i * SMEM_PACKED_B_SIZE_PER_STAGE);
    });
    auto sf_start_ptr = math::advance_ptr<uint8_t>(smem_gemm_base,
        SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_PACKED_B_SIZE_PER_STAGE) +
        kNumDecodedBStages * SMEM_B_SIZE_PER_STAGE);
    auto smem_sfa = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<float*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });
    // Barriers live after SF.
    auto smem_ksplit_reduce = reinterpret_cast<float*>(
        sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE);
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(
        sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE + SMEM_KSPLIT_REDUCE_SIZE);
    auto dispatch_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + i; });
    auto full_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + i; });
    auto empty_barriers    = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + kNumStages + i; });
    auto combine_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + kNumStages * 2 + i; });
    constexpr uint32_t kNumBaseBarriers =
        kNumDispatchWarps + kNumStages * 2 + kNumEpilogueWarps * 2;
    auto task_info_full_barriers = barrier_start_ptr + kNumBaseBarriers;
    auto task_info_empty_barriers = task_info_full_barriers +
        interleaved_scheduler_t::kNumScheduleStages;
    auto task_infos = reinterpret_cast<task_info_t*>(
        task_info_empty_barriers +
        interleaved_scheduler_t::kNumScheduleStages);
    // Stream-K: CTA-wide broadcast of the tile arrival ticket (epilogue thread 0 ->
    // all epilogue threads), 16 B after the task mailbox.
    auto smem_streamk_ticket = reinterpret_cast<uint32_t*>(
        task_infos + interleaved_scheduler_t::kNumScheduleStages);
    constexpr uint32_t kInterleavedSchedulerSMEMBytes =
        2 * interleaved_scheduler_t::kNumScheduleStages * sizeof(Barrier) +
        interleaved_scheduler_t::kNumScheduleStages * sizeof(task_info_t) + 16u;
    DG_STATIC_ASSERT(
        kInterleavedSchedulerSMEMBytes ==
            fused_layout::kSM90InterleavedSchedulerSMEMBytes,
        "Host and device scheduler shared-memory layouts disagree");
    constexpr uint32_t kInterleavedSMEMEnd =
        SMEM_BEFORE_BARRIER_SIZE + kNumStages * SMEM_SFA_SIZE_PER_STAGE +
        SMEM_KSPLIT_REDUCE_SIZE +
        kNumBaseBarriers * sizeof(Barrier) +
        kInterleavedSchedulerSMEMBytes;
    DG_STATIC_ASSERT(!kUseInterleavedScheduler || kInterleavedSMEMEnd <= 232448,
                     "Interleaved scheduler exceeds the SM90 shared-memory capacity");
    // The BM8 hosts launch with the full 232448 B; the multi-K-block layout must fit
    // regardless of the scheduler variant: 2 blocks x 4 stages = 4 x (2048 + 40960 +
    // 256) = 173056 B, 4 blocks x 2 stages = 2 x (4096 + 81920 + 512) = 173056 B of
    // stages plus the fixed regions and barriers.
    DG_STATIC_ASSERT(kKBlocksPerStage == 1 || kInterleavedSMEMEnd <= 232448,
                     "Multi-K-block pipeline exceeds the SM90 shared-memory capacity");

    // =====================================================================
    // Initialization
    // =====================================================================
    // QoQ decodes with plain ALU ops (shift/mask/subtract, see `decode_stage_rf`),
    // so it has no LUT to stage; only the FP4 paths fill the 1 KB LUT.
    if constexpr (!kQoQ) {
        if (thread_idx < 64) {
            reinterpret_cast<uint4*>(smem_nvfp4_lut)[thread_idx] =
                kMXFP4 ?
                    reinterpret_cast<const uint4*>(deep_gemm::nvfp4::kE2M1AndE8M0RelToFp8Lut)[thread_idx] :
                    reinterpret_cast<const uint4*>(deep_gemm::nvfp4::kE2M1AndUe4m3ToFp8Lut)[thread_idx];
        }
    }
    // Per-lane replicated RF LUT: only when its region is actually allocated
    // (SMEM_RF_LUT_REP_SIZE, currently 0: the RF decoders gather from the 1 KB LUT
    // above). With a 0-byte region the 32 KB fill would land in the GEMM stage
    // area as dead stores in the prologue of every CTA.
    if constexpr (kRFDecode && kMXFP4 && SMEM_RF_LUT_REP_SIZE > 0) {
        const uint2* lut_src = reinterpret_cast<const uint2*>(deep_gemm::nvfp4::kE2M1AndE8M0RelToFp8Lut);
        for (uint32_t i = thread_idx; i < 128u * 32u; i += blockDim.x)
            smem_rf_lut[i] = lut_src[i >> 5];
    }

    if (warp_idx == 0) {
        // Clean expert-count shared memory
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumExperts; i += 32)
            ptx::st_shared(smem_expert_count + i, 0u);
    } else if (warp_idx == 1) {
        // Init dispatch m-barriers
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumDispatchWarps; i += 32)
            dispatch_barriers[i]->init(1);
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Init GEMM full/empty barriers and combine barriers
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++ i) {
                // Producer arrivals: A(+SFA) + B(TMA+SFB). SFB is copied with
                // cp.async.bulk and counted as B-loader transaction bytes, so
                // it does not need a separate producer arrival.
                full_barriers[i]->init(2);
                empty_barriers[i]->init(kNumEpilogueWarps);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueWarps * 2; ++ i)
                combine_barriers[i]->init(1);
            if constexpr (kUseInterleavedScheduler) {
                #pragma unroll
                for (uint32_t i = 0;
                     i < interleaved_scheduler_t::kNumScheduleStages;
                     ++ i) {
                    task_info_full_barriers[i].init(1);
                    task_info_empty_barriers[i].init(kNumEpilogueWarps);
                }
            }
        }
        cutlass::arch::fence_barrier_init();
    }
    // Fast NVLink-barrier epilogue: every thread snapshots the done count BEFORE
    // the kernel-start __syncthreads (see fused_comm::nvlink_barrier for why this
    // is race-free: SM0's first write of the word this launch is ordered after the
    // first barrier's prologue grid sync, which every CTA reaches after this sync).
    const uint32_t nvl_done_base = kNvlFastEpilogue ?
        ptx::ld_volatile(workspace.get_nvl_done_count_ptr()) : 0u;
    __syncthreads();
    if (thread_idx == 0) stamp_max(12);

    // =====================================================================
    // Scheduler (cluster=1)
    // =====================================================================
    auto scheduler = fused_sched::MegaMoEScheduler<
        BLOCK_M, TASK_BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank, kNumExpertsPerWave,
        kNumSMs, kNumRanks, 1>(workspace);
    auto interleaved_scheduler = interleaved_scheduler_t(
        workspace,
        task_info_full_barriers,
        task_info_empty_barriers,
        task_infos);

    // Pipeline state shared by TMA loaders and math warpgroups
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        k_block_idx += kKBlocksPerStage;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };
    // Intra-SM barrier indices (mirroring SM100)
    constexpr uint32_t kDispatchBarrierIdx              = 0;
    constexpr uint32_t kDispatchWithEpilogueBarrierIdx  = 1;
    constexpr uint32_t kEpilogueFullBarrierIdx          = 2;
    constexpr uint32_t kEpilogueWGBarrierStartIdx       = 3;
    constexpr uint32_t kSplitMDecodeBarrierIdx          = 8;
    constexpr uint32_t kKSplitReduceBarrierIdx          = 9;  // kHalfTileTasks WG1 -> WG0

    // Cross-rank NVLink barrier tags
    constexpr uint32_t kBeforeDispatchPullBarrierTag    = 1;
    constexpr uint32_t kBeforeCombineReduceBarrierTag   = 2;
    constexpr uint32_t kAfterWorkspaceCleanBarrierTag   = 3;

    // Register reconfiguration counts (chosen to fit in 64512 reg budget).
    constexpr uint32_t kNumDispatchRegisters    = 48;
    constexpr uint32_t kNumNonEpilogueRegisters =
        kUseInterleavedScheduler ? 64 : 40;
    constexpr uint32_t kNumEpilogueRegisters    = 208;
    DG_STATIC_ASSERT(kNumDispatchRegisters * kNumDispatchThreads +
                     kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
                     kNumEpilogueRegisters * kNumEpilogueThreads <= 64512,
                     "Too many registers");

    constexpr uint32_t kDispatchGridSyncIndex = 0;
    constexpr uint32_t kEpilogueGridSyncIndex = 1;

    const auto for_each_static_selected_block = [&](auto&& func) {
        scheduler.fetch_expert_recv_count();
        scheduler.set_expert_idx(0);
        while (true) {
            CUTE_TIE_DECL(scheduler.get_next_block(),
                          block_phase, local_expert_idx, m_block_idx, n_block_idx);
            if (block_phase == fused_sched::BlockPhase::None)
                break;
            if (block_phase == fused_sched::BlockPhase::Linear1) {
                func(std::integral_constant<fused_sched::BlockPhase, fused_sched::BlockPhase::Linear1>{},
                     local_expert_idx, L1_SHAPE_K / BLOCK_K, m_block_idx, n_block_idx,
                     scheduler.get_current_pool_block_offset() + m_block_idx,
                     scheduler.template get_valid_m<false>(), 0u, 1u, 0u, 0u);
            } else {
                func(std::integral_constant<fused_sched::BlockPhase, fused_sched::BlockPhase::Linear2>{},
                     local_expert_idx, L2_SHAPE_K / BLOCK_K, m_block_idx, n_block_idx,
                     scheduler.get_current_pool_block_offset() + m_block_idx,
                     scheduler.template get_valid_m<false>(), 0u, 1u, 0u, 0u);
            }
        }
    };

    // K-block count of K split `k_split_idx` (of `num_k_splits` in {1, 2}) of a task
    // with `total_k_blocks` K-blocks: half 0 is the lower whole-stage half (floor to
    // kKBlocksPerStage; L1 24 -> 12, L2 10 -> 4), half 1 the rest (12 / 6). The
    // task consumers derive the first absolute K-block as total - count for half 1.
    const auto get_split_k_num_blocks = [](const uint32_t& total_k_blocks,
                                           const uint32_t& k_split_idx,
                                           const uint32_t& num_k_splits) -> uint32_t {
        if (num_k_splits == 1u)
            return total_k_blocks;
        const uint32_t half_0 = ((total_k_blocks / 2u) / kKBlocksPerStage) * kKBlocksPerStage;
        return k_split_idx == 0u ? half_0 : total_k_blocks - half_0;
    };
    const auto invoke_interleaved_task = [&](const task_info_t& task_info,
                                              auto&& func) {
        // Split-K tasks (L1 tail: kSplitKL1, L2 tail: kSplitKL2) cover the K-block
        // range of their K split; every other task covers the whole K.
        // Stream-K segments carry an explicit K-block range (all tasks of a
        // stream-K launch are segments, most of them the whole K).
        // (Marker: only stream-K segments carry a non-zero K-block end; every warp
        // owns its own scheduler object and only the producer knows the mode.)
        const bool is_streamk = kStreamK && task_info.get_k_block_end() != 0u;
        if (task_info.block_phase == fused_sched::BlockPhase::Linear1) {
            const uint32_t num_k_splits = (kSplitKL1 || kStreamK) ? task_info.get_num_k_splits() : 1u;
            const uint32_t k_split_idx = (kSplitKL1 || kStreamK) ? task_info.get_k_split_idx() : 0u;
            const uint32_t num_k_blocks = is_streamk ?
                task_info.get_k_block_end() - task_info.get_k_block_begin() :
                get_split_k_num_blocks(L1_SHAPE_K / BLOCK_K, k_split_idx, num_k_splits);
            const uint32_t k_block_begin = is_streamk ? task_info.get_k_block_begin() :
                (k_split_idx == 0u ? 0u : L1_SHAPE_K / BLOCK_K - num_k_blocks);
            func(std::integral_constant<fused_sched::BlockPhase, fused_sched::BlockPhase::Linear1>{},
                 task_info.local_expert_idx, num_k_blocks,
                 task_info.m_block_idx, task_info.n_block_idx,
                 task_info.pool_block_idx, task_info.valid_m,
                 k_split_idx, num_k_splits, k_block_begin,
                 is_streamk ? task_info.get_first_worker_idx() : 0u);
        } else {
            const uint32_t num_k_splits = (kSplitKL2 || kStreamK) ? task_info.get_num_k_splits() : 1u;
            const uint32_t k_split_idx = (kSplitKL2 || kStreamK) ? task_info.get_k_split_idx() : 0u;
            const uint32_t num_k_blocks = is_streamk ?
                task_info.get_k_block_end() - task_info.get_k_block_begin() :
                get_split_k_num_blocks(L2_SHAPE_K / BLOCK_K, k_split_idx, num_k_splits);
            const uint32_t k_block_begin = is_streamk ? task_info.get_k_block_begin() :
                (k_split_idx == 0u ? 0u : L2_SHAPE_K / BLOCK_K - num_k_blocks);
            func(std::integral_constant<fused_sched::BlockPhase, fused_sched::BlockPhase::Linear2>{},
                 task_info.local_expert_idx, num_k_blocks,
                 task_info.m_block_idx, task_info.n_block_idx,
                 task_info.pool_block_idx, task_info.valid_m,
                 k_split_idx, num_k_splits, k_block_begin,
                 is_streamk ? task_info.get_first_worker_idx() : 0u);
        }
    };

    const auto for_each_published_block = [&](auto&& func) {
        task_info_t task_info;
        while (interleaved_scheduler.get_published_task(task_info))
            invoke_interleaved_task(task_info, func);
    };

    const auto produce_interleaved_blocks = [&](auto&& func) {
        interleaved_scheduler.fetch_expert_recv_count();
        while (true) {
            interleaved_scheduler.wait_task_slot_empty();
            const auto task_info = interleaved_scheduler.claim_next_task();
            interleaved_scheduler.publish_task(task_info);
            if (!task_info.is_valid())
                break;
            invoke_interleaved_task(task_info, func);
        }
    };

    const auto cleanup_workspace = [&]() {
        DG_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");
        if (sm_idx == 0) {
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads)
                *workspace.get_expert_send_count_ptr(i) = 0;
            if constexpr (kUseInterleavedScheduler) {
                if (thread_idx == 0) {
                    *workspace.get_l1_task_count_ptr() = 0;
                    *workspace.get_l2_task_count_ptr() = 0;
                }
                if constexpr (kSplitKL1 || kSplitKL2 || kStreamK || kTinyMGemv) {
                    // L1 and L2 flag slots are contiguous (L1 first)
                    for (uint32_t i = thread_idx; i < fused_layout::kSM90SplitKNumSlots; i += kNumDispatchThreads)
                        *workspace.get_splitk_l1_flag_ptr(0, i) = 0;
                }
            }
        } else {
            for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
                const auto num_recv_tokens = static_cast<uint32_t>(
                    *workspace.get_expert_recv_count_sum_ptr(i));
                const auto num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);
                const auto cleanup_pool_block_offset = kPushDispatch ?
                    i * kPushBlocksPerExpert : scheduler.get_pool_block_offset(i);

                ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

                DG_STATIC_ASSERT(kNumDispatchWarps >= 2, "Not enough dispatch warps");
                if (warp_idx == 0) {
                    *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                } else if (warp_idx == 1) {
                    if (cute::elect_one_sync() and cumulative_local_expert_recv_stats != nullptr)
                        ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
                    __syncwarp();
                }

                for (uint32_t j = thread_idx; j < kNumRanks; j += kNumDispatchThreads)
                    *workspace.get_expert_recv_count_ptr(j, i) = 0;
                __syncwarp();

                for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumDispatchThreads) {
                    *workspace.get_l1_arrival_count_ptr(cleanup_pool_block_offset + j) = 0;
                    *workspace.get_l2_arrival_mask_ptr(cleanup_pool_block_offset + j) = 0;
                }
                __syncwarp();
            }
        }
    };

    // =====================================================================
    // ROLE 1: DISPATCH WARPS
    //   Mirrors SM100 dispatch with two changes:
    //     * SF is per-128 channel float (no UTCCP transpose). We store the
    //       remote per-token SF directly into the local L1 SF buffer in
    //       MN-major layout: `local_sf[k_chunk * num_padded_sf_pool_tokens + token_idx]`.
    //     * The "token_idx_in_expert" → SF token index is now the simple
    //       per-block linear mapping (no 4×32 transpose).
    // =====================================================================
    if (warp_idx < kNumDispatchWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();

        DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
        constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
        const auto read_topk_idx = [&](const auto& process) {
            if (warp_idx < kNumActiveDispatchWarps) {
                #pragma unroll
                for (uint32_t i = (sm_idx * kNumActiveDispatchWarps + warp_idx) * kNumTokensPerWarp;
                     i < num_tokens;
                     i += kNumSMs * kNumActiveDispatchWarps * kNumTokensPerWarp) {
                    int expert_idx = -1;
                    if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes) {
                        expert_idx = static_cast<int>(
                            __ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + i * kNumTopk + lane_idx));
                        if (expert_idx >= 0)
                            process(i * kNumTopk + lane_idx, expert_idx);
                    }
                    __syncwarp();
                }
            }
        };

        // Lean routing (kLeanRouting; host env DG_FP4_LEAN_ROUTING, default 1).
        // Baseline: every CTA adds `(1 << 32) | count` to ALL 384 `expert_send_count`
        // words (78 x 384 = 30k serialised L2 atomics, ~5 us at any M, most of them
        // count 0); the high word (CTA arrivals) is forwarded by the broadcast into
        // `expert_recv_count_sum`, whose high word the destination's scheduler polls
        // for kNumSMs * kNumRanks (completeness). The grid sync before the broadcast
        // already guarantees every CTA's count landed, so the per-CTA arrival word is
        // redundant: lean routing adds only non-zero counts (no high word) and the
        // broadcast supplies the constant kNumSMs high word instead. Push dispatch
        // (kLeanPush) needs neither the local send counts (only the broadcast read
        // them), nor the extra grid sync, nor the cross-rank broadcast: the per-row
        // remote tickets already carry the low word, all of them are ordered before
        // this rank's barrier #1 signal by the barrier's prologue grid sync, and the
        // DESTINATION adds the kNumSMs * kNumRanks high word to its own 48 sums after
        // barrier #1 (one local atomic per expert). Data seen by the scheduler /
        // dispatch (`expert_recv_count`, `expert_recv_count_sum`) is unchanged.
        constexpr bool kLeanPush = kLeanRouting && kPushDispatch;
        if constexpr (!kLeanPush) {
            // Count tokens per expert
            read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
                atomicAdd_block(smem_expert_count + expert_idx, 1);
            });
            ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

            // Stake out per-expert SM offsets via global atomic
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                const uint32_t count = smem_expert_count[i];
                if constexpr (kLeanRouting) {
                    if (count != 0)
                        smem_expert_count[i] = static_cast<uint32_t>(ptx::atomic_add(
                            workspace.get_expert_send_count_ptr(i), static_cast<uint64_t>(count)));
                } else {
                    const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(count);
                    smem_expert_count[i] = static_cast<uint32_t>(
                        ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
                }
            }
            ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
        }
        if (thread_idx == 0) stamp_max(8);

        if constexpr (kPushDispatch) {
            // Push: one remote ticket per (token, top-k slot) lane (all lanes of the
            // warp in flight together, one NVLink round trip), then the warp streams
            // each routed row into the destination pool (16 B per lane per store).
            if (warp_idx < kNumActiveDispatchWarps) {
                constexpr uint32_t kNumSFFloats = kHidden / 128;
                constexpr uint32_t kNumTokenChunks = kHidden / 16;
                DG_STATIC_ASSERT(kHidden % 128 == 0 and kNumSFFloats <= 32, "Invalid SF");
                for (uint32_t i = (sm_idx * kNumActiveDispatchWarps + warp_idx) * kNumTokensPerWarp;
                     i < num_tokens;
                     i += kNumSMs * kNumActiveDispatchWarps * kNumTokensPerWarp) {
                    int expert_idx = -1;
                    const uint32_t token_topk_idx = i * kNumTopk + lane_idx;
                    if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes)
                        expert_idx = static_cast<int>(
                            __ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + token_topk_idx));
                    const bool active = expert_idx >= 0;
                    const uint32_t dst_rank_idx = active ? static_cast<uint32_t>(expert_idx) / kNumExpertsPerRank : 0u;
                    const uint32_t dst_local_expert_idx = active ? static_cast<uint32_t>(expert_idx) % kNumExpertsPerRank : 0u;
                    uint32_t row_in_expert = 0;
                    if (active)
                        row_in_expert = static_cast<uint32_t>(ptx::atomic_add_sys(
                            sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx),
                            1ull));
                    uint32_t active_mask = __ballot_sync(0xffffffff, active);
                    while (active_mask) {
                        const uint32_t src_lane = __ffs(active_mask) - 1;
                        active_mask &= active_mask - 1;
                        const uint32_t row = __shfl_sync(0xffffffff, row_in_expert, src_lane);
                        const uint32_t dr  = __shfl_sync(0xffffffff, dst_rank_idx, src_lane);
                        const uint32_t de  = __shfl_sync(0xffffffff, dst_local_expert_idx, src_lane);
                        const uint32_t tti = __shfl_sync(0xffffffff, token_topk_idx, src_lane);
                        DG_DEVICE_ASSERT(row < kPushBlocksPerExpert * BLOCK_M);
                        const uint32_t src_token_idx = tti / kNumTopk, src_topk_idx = tti % kNumTopk;
                        const uint32_t pool_token_idx = de * kPushBlocksPerExpert * BLOCK_M + row;
                        const auto* src_token = input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr<uint4>();
                        auto* dst_token = sym_buffer.map(
                            l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr<uint4>(), dr);
                        #pragma unroll
                        for (uint32_t c = lane_idx; c < kNumTokenChunks; c += 32)
                            dst_token[c] = __ldg(src_token + c);
                        const auto* src_sf = input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<float>();
                        auto* dst_sf = sym_buffer.map(l1_sf_buffer.get_base_ptr<float>(), dr);
                        if (lane_idx < kNumSFFloats)
                            dst_sf[lane_idx * kNumPaddedSFPoolTokens + pool_token_idx] = __ldg(src_sf + lane_idx);
                        if (lane_idx == 0) {
                            *sym_buffer.map(l1_topk_weights_buffer.get_data_buffer(pool_token_idx).get_base_ptr<float>(), dr) =
                                __ldg(input_topk_weights_buffer.get_base_ptr<float>() + tti);
                            *sym_buffer.map(workspace.get_token_src_metadata_ptr(pool_token_idx), dr) =
                                {static_cast<uint32_t>(sym_buffer.rank_idx), src_token_idx, src_topk_idx};
                        }
                    }
                    __syncwarp();
                }
            }
            if (thread_idx == 0) stamp_max(9);  // push issued
        } else {
            // Write source token-topk indices to remote ranks
            read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
                const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;
                const auto dst_slot_idx = atomicAdd_block(smem_expert_count + expert_idx, 1);
                const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(
                    expert_idx % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
                *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
            });
            if (thread_idx == 0) stamp_max(9);
        }

        // Lean push: no cross-CTA total is needed before barrier #1 (the barrier's
        // prologue grid sync orders the pushes), so the routing grid sync is dropped.
        if constexpr (!kLeanPush) {
            fused_comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
                workspace, sm_idx, thread_idx,
                [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
            );
        }
        if (thread_idx == 0) stamp_max(10);

        const auto broadcast_expert_status = [&](const uint32_t& i) {
            const auto dst_rank_idx = i / kNumExpertsPerRank;
            const auto dst_local_expert_idx = i % kNumExpertsPerRank;
            // Lean routing: the send count carries no arrival high word; supply the
            // constant the destination's scheduler expects (kNumSMs per source rank).
            const uint64_t expert_status = kLeanRouting ?
                ((static_cast<uint64_t>(kNumSMs) << 32) |
                 (*workspace.get_expert_send_count_ptr(i) & 0xffffffffull)) :
                *workspace.get_expert_send_count_ptr(i);
            *sym_buffer.map(
                workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx),
                dst_rank_idx) = expert_status & 0xffffffff;
            // Push mode: the low word (row count) was already added by the per-row
            // tickets (all returned before this rank's grid sync), so add only the
            // high word that finalises the count for the destination's scheduler.
            ptx::atomic_add_sys(
                sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx),
                kPushDispatch ? (expert_status & 0xffffffff00000000ull) : expert_status);
        };
        if constexpr (kLeanPush) {
            // No broadcast: the destination finalises its counts after barrier #1.
        } else if constexpr (kDistributedExpertBcast) {
            // Spread the per-expert cross-rank count updates over every SM
            // (<= 3 blocking sys-scope atomics per thread) instead of 12 serial
            // rounds on SM 0; the NVLink barrier below then needs its grid-sync
            // prologue so SM 0 only signals after all SMs have published.
            for (uint32_t i = sm_idx + thread_idx * kNumSMs; i < kNumExperts;
                 i += kNumSMs * kNumDispatchThreads)
                broadcast_expert_status(i);
        } else if (sm_idx == 0 and thread_idx < kNumActiveDispatchThreads) {
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumActiveDispatchThreads)
                broadcast_expert_status(i);
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
        unsigned long long t_before_nvl_barrier = 0;
        if (thread_idx == 0) {
            stamp_max(11);
            if (phase_stamps != nullptr)
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_before_nvl_barrier));
        }

        fused_comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            kDistributedExpertBcast || kLeanPush, true,
            /* fast epilogue: barrier #1 of the launch */ kNvlFastEpilogue, nvl_done_base, 1u);
        if (thread_idx == 0) {
            stamp_max(1);
            if (sm_idx == 0) {
                stamp_accumulate(13);
                if (phase_stamps != nullptr) {
                    atomicAdd(phase_stamps + 14, 1ull);
                    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
                    atomicAdd(phase_stamps + 16, t - t_before_nvl_barrier);  // barrier wait incl. skew
                }
            }
        }

        if constexpr (kPushDispatch) {
            // Every rank's rows are in the local pool (ordered by barrier #1). SM e
            // (e < experts per rank) publishes expert e's L1 arrival counts, one block
            // per lane, with release.gpu: the loaders' acquire on the count then also
            // covers the remotely written rows (this thread acquired barrier #1).
            if (warp_idx == 0 and sm_idx < kNumExpertsPerRank) {
                uint32_t num_recv_tokens;
                if constexpr (kLeanPush) {
                    // Finalise expert `sm_idx`'s count: every rank's row tickets landed
                    // before its barrier #1 signal (acquired by this CTA), so add the
                    // completeness high word the scheduler polls for and take the
                    // total from the same atomic.
                    uint32_t low = 0;
                    if (lane_idx == 0)
                        low = static_cast<uint32_t>(ptx::atomic_add(
                            workspace.get_expert_recv_count_sum_ptr(sm_idx),
                            static_cast<uint64_t>(kNumSMs * kNumRanks) << 32));
                    num_recv_tokens = __shfl_sync(0xffffffff, low, 0);
                } else {
                    num_recv_tokens = static_cast<uint32_t>(
                        ptx::ld_volatile(workspace.get_expert_recv_count_sum_ptr(sm_idx)));
                }
                const uint32_t num_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);
                if (lane_idx < num_blocks)
                    ptx::red_add_rel(
                        workspace.get_l1_arrival_count_ptr(sm_idx * kPushBlocksPerExpert + lane_idx),
                        cute::min(num_recv_tokens - lane_idx * BLOCK_M, BLOCK_M));
                __syncwarp();
            }
            if (thread_idx == 0) stamp_max(2);  // pool ready
        }

        // Sync with epilogue warps before pulling tokens
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Token / SF pull loop
        if (!kPushDispatch and warp_idx < kNumActiveDispatchWarps) {
            uint32_t pull_mbarrier_phase = 0;
            const auto pull_buffer = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(0);
            const auto pull_mbarrier = dispatch_barriers[warp_idx];

            scheduler.fetch_expert_recv_count();

            constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumRanks, 32u);
            int      current_expert_idx = -1;
            uint32_t stored_rank_count[kNumRanksPerLane] = {};
            uint32_t expert_start_idx = 0, expert_end_idx = 0;
            uint32_t expert_pool_block_offset = 0;

            constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumActiveDispatchWarps;
            for (uint32_t token_idx = sm_idx * kNumActiveDispatchWarps + warp_idx; ; token_idx += kNumGlobalWarps) {
                int old_expert_idx = current_expert_idx;
                while (token_idx >= expert_end_idx) {
                    if (++ current_expert_idx >= kNumExpertsPerRank)
                        break;
                    expert_pool_block_offset += math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M);
                    expert_start_idx = expert_end_idx;
                    expert_end_idx += scheduler.get_num_tokens(current_expert_idx);
                }
                if (current_expert_idx >= kNumExpertsPerRank)
                    break;

                if (old_expert_idx != current_expert_idx) {
                    old_expert_idx = current_expert_idx;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                        const uint32_t j = i * 32 + lane_idx;
                        stored_rank_count[i] = j < kNumRanks ?
                            static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(j, current_expert_idx)) : 0;
                    }
                }

                // Round-robin rank selection (identical to SM100)
                uint32_t current_rank_in_expert_idx;
                uint32_t remaining[kNumRanksPerLane];
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                    remaining[i] = stored_rank_count[i];
                uint32_t offset = 0;
                uint32_t token_idx_in_expert = token_idx - expert_start_idx;
                uint32_t slot_idx = token_idx_in_expert;
                uint32_t token_idx_in_rank;
                while (true) {
                    uint32_t num_actives_in_lane = 0;
                    uint32_t min_in_lane = 0xffffffff;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                        num_actives_in_lane += remaining[i] > 0;
                        if (remaining[i] > 0)
                            min_in_lane = cute::min(min_in_lane, remaining[i]);
                    }
                    const uint32_t num_active_ranks = __reduce_add_sync(0xffffffff, num_actives_in_lane);
                    const uint32_t length = __reduce_min_sync(0xffffffff, min_in_lane);

                    const uint32_t num_round_tokens = length * num_active_ranks;
                    if (slot_idx < num_round_tokens) {
                        const uint32_t slot_idx_in_round = slot_idx % num_active_ranks;
                        uint32_t num_seen_ranks = 0;
                        current_rank_in_expert_idx = 0;
                        #pragma unroll
                        for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                            const uint32_t mask = __ballot_sync(0xffffffff, remaining[i] > 0);
                            const uint32_t num_active_lanes = __popc(mask);
                            if (slot_idx_in_round >= num_seen_ranks and slot_idx_in_round < num_seen_ranks + num_active_lanes)
                                current_rank_in_expert_idx = i * 32 + __fns(mask, 0, slot_idx_in_round - num_seen_ranks + 1);
                            num_seen_ranks += num_active_lanes;
                        }
                        token_idx_in_rank = offset + (slot_idx / num_active_ranks);
                        break;
                    }
                    slot_idx -= num_round_tokens;
                    offset += length;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                        remaining[i] -= cute::min(remaining[i], length);
                }

                const uint32_t src_token_topk_idx = *workspace.get_src_token_topk_idx_ptr(
                    current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank);
                const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
                const uint32_t src_topk_idx  = src_token_topk_idx % kNumTopk;

                // TMA pull token data into SMEM
                if (cute::elect_one_sync()) {
                    ptx::tma_load_1d(
                        pull_buffer.get_base_ptr(),
                        sym_buffer.map(input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(),
                                       current_rank_in_expert_idx),
                        pull_mbarrier, kHidden);
                }
                __syncwarp();

                // Copy SF: per-128 K floats, written linearly (no UTCCP transpose).
                constexpr uint32_t kNumSFFloats = kHidden / 128;
                DG_STATIC_ASSERT(kNumSFFloats > 0 and kHidden % 128 == 0, "Invalid SF");
                const auto remote_sf_ptr = sym_buffer.map(
                    input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<float>(),
                    current_rank_in_expert_idx);
                const auto local_sf_ptr  = l1_sf_buffer.get_base_ptr<float>();
                const uint32_t pool_token_idx =
                    expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
                // Issue the remote top-k weight load together with the SF loads (it used to
                // wait behind the SF stores, adding a full NVLink round trip to the chain).
                float weight = 0.0f;
                if (lane_idx == 0)
                    weight = *sym_buffer.map(
                        input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                        current_rank_in_expert_idx);
                #pragma unroll
                for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumSFFloats, 32u); ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    if (j < kNumSFFloats)
                        local_sf_ptr[j * kNumPaddedSFPoolTokens + pool_token_idx] = remote_sf_ptr[j];
                }
                if (lane_idx == 0)
                    *l1_topk_weights_buffer.get_data_buffer(pool_token_idx).get_base_ptr<float>() = weight;
                __syncwarp();

                if (cute::elect_one_sync()) {

                    ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kHidden);
                    ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);

                    ptx::tma_store_1d(
                        l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr(),
                        pull_buffer.get_base_ptr(), pull_buffer.get_num_bytes());

                    *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                        {current_rank_in_expert_idx, src_token_idx, src_topk_idx};

                    cute::tma_store_arrive();
                    ptx::tma_store_wait<0>();
                    ptx::red_add_rel(
                        workspace.get_l1_arrival_count_ptr(
                            expert_pool_block_offset + token_idx_in_expert / BLOCK_M),
                        1u);
                }
                __syncwarp();
            }
        }

        if (!kPushDispatch and thread_idx == 0) stamp_max(2);
        if constexpr (kFineCombine) {
            // Fine-grained combine signaller (dispatch warp 0): consume this CTA's
            // mailbox; for every finished L2 task release-add 1 per scattered token
            // row to the destination rank's counter (sys-scope fence here, off the
            // math warps), until the epilogue posts DONE (all math tasks finished).
            DG_STATIC_ASSERT(kNumSMs <= fused_layout::kSM90FineCombineMaxSMs, "Too many SMs for the combine mailboxes");
            if (warp_idx == 0) {
                auto* mailbox = workspace.get_combine_mailbox_ptr(sm_idx);
                uint32_t consumed = ptx::ld_volatile(mailbox + 1);
                while (true) {
                    while (ptx::ld_acq(mailbox) == consumed) {}
                    const uint32_t entry = ptx::ld_volatile(
                        mailbox + 4 + (consumed & (fused_layout::kSM90FineCombineRingSize - 1)));
                    __syncwarp();
                    if (entry == fused_layout::kSM90FineCombineDoneEntry)
                        break;
                    const uint32_t signal_pool_block_idx = entry & 0xffffffu, signal_valid_m = entry >> 24;
                    for (uint32_t row = lane_idx; row < signal_valid_m; row += 32) {
                        const auto src_metadata = *workspace.get_token_src_metadata_ptr(
                            signal_pool_block_idx * BLOCK_M + row);
                        asm volatile("fence.acq_rel.sys;" ::: "memory");
                        ptx::red_add_rel_sys(
                            sym_buffer.map(workspace.get_combine_arrival_count_ptr(src_metadata.token_idx),
                                           src_metadata.rank_idx), 1);
                    }
                    __syncwarp();
                    ++ consumed;
                    if (lane_idx == 0)
                        ptx::st_rel_gpu(mailbox + 1, consumed);  // release: slot read done before reuse
                }
                // Terminal entry consumed too (keeps producer/consumer sequences aligned)
                ++ consumed;
                if (lane_idx == 0)
                    ptx::st_rel_gpu(mailbox + 1, consumed);
                __syncwarp();
            }
            // All dispatch warps: this CTA's math tasks are done (warp 0 saw DONE);
            // the cleanup below zeroes the L1 arrival counts / L2 arrival masks that
            // other CTAs' L1/L2 tasks still touch, so wait for every CTA's math tasks
            // (dispatch warps only; the combine warps are not held up).
            fused_comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
                workspace, sm_idx, thread_idx,
                [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); });
        } else {
            // Cleanup workspace, overlapping with combine
            ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
        }

        cleanup_workspace();
        fused_comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            true, false);
    } else if (warp_idx == kNumDispatchWarps) {
        // =====================================================================
        // ROLE 2: GEMM TMA LOAD warps (load A+SFA, B+SFB)
        //   The two warps inside `kNumNonEpilogueThreads` load A + SFA and
        //   B + SFB, respectively.
        // =====================================================================
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        const auto load_a_task = [&](const auto& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx,
                                     const uint32_t& pool_block_idx,
                                     const uint32_t& valid_m,
                                     const uint32_t& k_split_idx,
                                     const uint32_t& num_k_splits,
                                     const uint32_t& k_block_begin,
                                     const uint32_t& first_worker_idx) {
            using BlockPhaseTag = std::remove_cv_t<std::remove_reference_t<decltype(block_phase)>>;
            constexpr bool kBlockIsL2 = BlockPhaseTag::value == fused_sched::BlockPhase::Linear2;
            // `k_block_begin`: first absolute K-block of this task (split-K K half 1
            // / stream-K segments start mid-K).
            const auto tensor_map_a_ptr = kBlockIsL2 ?
                &tensor_map_l2_acts : &tensor_map_l1_acts;
            const auto tensor_map_sfa_ptr = kBlockIsL2 ?
                &tensor_map_l2_acts_sf : &tensor_map_l1_acts_sf;

            const bool has_valid_m = valid_m > 0;

            // Wait for the pool to be ready. While waiting, warm L2 cache with
            // this task's weight rows beyond the smem stage depth (the B loader
            // is blocked on empty barriers after kNumStages), so the tile runs
            // from L2 once the activations arrive. Matters at tiny M where the
            // per-tile latency chain, not bandwidth, bounds the kernel.
            if (has_valid_m) {
                if constexpr (kPrefetchWeightKBlocks > 0) {
                    if (cute::elect_one_sync()) {
                        constexpr uint32_t shape_n = kBlockIsL2 ? L2_SHAPE_N : L1_SHAPE_N;
                        constexpr uint32_t kInFlightKBlocks = kNumStages * kKBlocksPerStage;
                        const uint32_t k_end = cute::min(num_k_blocks, kInFlightKBlocks + kPrefetchWeightKBlocks);
                        if constexpr (kDenseWeightTiles) {
                            // Dense tiles: k-blocks of one (expert, n_block) are
                            // contiguous, so prefetch whole 20 KB tiles in 1D.
                            constexpr uint32_t shape_k = kBlockIsL2 ? L2_SHAPE_K : L1_SHAPE_K;
                            const auto* weights_base = reinterpret_cast<const uint8_t*>(
                                kBlockIsL2 ? l2_weights_ptr : l1_weights_ptr);
                            constexpr uint32_t kPhaseSubTiles =
                                kBlockIsL2 ? kL2SubTilesPerPacked : kSubTilesPerPacked;
                            constexpr uint32_t kPhaseKBlockBytes =
                                kBlockIsL2 ? SMEM_PACKED_B_L2_SIZE_PER_KBLOCK : SMEM_PACKED_B_SIZE_PER_KBLOCK;
                            const uint32_t tile_row = local_expert_idx * (shape_n / kPackedTileN) +
                                                      n_block_idx / kPhaseSubTiles;
                            const auto* tiles = weights_base +
                                static_cast<size_t>(tile_row) * (shape_k / BLOCK_K) * kPackedTileBytes +
                                (n_block_idx % kPhaseSubTiles) * kPhaseKBlockBytes;
                            for (uint32_t kb = kInFlightKBlocks; kb < k_end; ++ kb)
                                ptx::tma_prefetch_1d(tiles + static_cast<size_t>(k_block_begin + kb) * kPackedTileBytes,
                                                     kPhaseKBlockBytes);
                        } else {
                            const auto tensor_map_b_ptr = kBlockIsL2 ?
                                &tensor_map_l2_weights : &tensor_map_l1_weights;
                            const uint32_t n_idx = local_expert_idx * shape_n + n_block_idx * BLOCK_N;
                            for (uint32_t kb = kInFlightKBlocks; kb < k_end; ++ kb)
                                cute::SM90_TMA_LOAD_2D::PREFETCH::copy(
                                    tensor_map_b_ptr, (k_block_begin + kb) * B_LOAD_BYTES_PER_ROW, n_idx);
                        }
                    }
                    __syncwarp();
                }
                if constexpr (!kBlockIsL2) {
                    const auto ptr = workspace.get_l1_arrival_count_ptr(pool_block_idx);
                    while (ptx::ld_acq(ptr) != valid_m) {}
                    // Push dispatch: the rows are generic-proxy (remote st.global)
                    // writes read below through TMA (async proxy).
                    if constexpr (kPushDispatch)
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                }
                // L2: no up-front wait for all L1 N-blocks; each stage below
                // waits only for the L1 blocks that produced its K-block(s).
            }
            // Cached snapshot of the L1 readiness mask (L2 tasks only).
            uint64_t l1_ready_mask = 0;
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                if constexpr (kBlockIsL2) {
                    if (has_valid_m) {
                        // Bits of the L1 N-blocks feeding absolute K-blocks
                        // [k_block_begin + k_block_idx, + blocks in this stage).
                        constexpr uint32_t kBitsPerStage = kKBlocksPerStage * kNumL1BlocksPerL2KBlock;
                        DG_STATIC_ASSERT(kBitsPerStage < 64, "Stage readiness mask overflow");
                        const uint32_t stage_bits = cute::min(kKBlocksPerStage, num_k_blocks - k_block_idx) *
                                                    kNumL1BlocksPerL2KBlock;
                        const uint64_t need = ((1ull << stage_bits) - 1ull)
                                              << ((k_block_begin + k_block_idx) * kNumL1BlocksPerL2KBlock);
                        if ((l1_ready_mask & need) != need) {
                            const auto ptr = workspace.get_l2_arrival_mask_ptr(pool_block_idx);
                            do {
                                l1_ready_mask = ptx::ld_acq_gpu(ptr);
                            } while ((l1_ready_mask & need) != need);
                        }
                    }
                }

                if (cute::elect_one_sync()) {
                    if (has_valid_m) {
                        const uint32_t m_idx = pool_block_idx * BLOCK_M;
                        const uint32_t k_idx = (k_block_begin + k_block_idx) * BLOCK_K;

                        if constexpr (!kBlockIsL2 || !kSplitMDecodedWeightReuse) {
                            // One A tile + one SFA row per K-block of the stage,
                            // a single expect-tx for the whole stage.
                            // L2 with per-64 activation scales (kHalfTileTasks): two SF
                            // rows per K128 block, slots (kb * kNumL2SFAGroups + g).
                            constexpr uint32_t kSFRowsPerKBlock =
                                kBlockIsL2 ? BLOCK_K / kL2ActsSFGranK : 1u;
                            DG_STATIC_ASSERT(kSFRowsPerKBlock <= kNumL2SFAGroups,
                                             "Not enough SFA slots per K-block");
                            // Partial last stage: only the remaining K-blocks are fetched
                            // (and counted in expect-tx).
                            const uint32_t num_stage_blocks =
                                cute::min(kKBlocksPerStage, num_k_blocks - k_block_idx);
                            #pragma unroll
                            for (uint32_t kb = 0; kb < kKBlocksPerStage; ++ kb) {
                                if (kb >= num_stage_blocks) break;
                                tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                                    tensor_map_a_ptr, full_barriers[stage_idx],
                                    smem_a[stage_idx] + kb * SMEM_A_SIZE_PER_KBLOCK / sizeof(a_dtype_t),
                                    k_idx + kb * BLOCK_K, m_idx, 1);
                                #pragma unroll
                                for (uint32_t g = 0; g < kSFRowsPerKBlock; ++ g) {
                                    tma::copy<BLOCK_M, 1, 0, float>(
                                        tensor_map_sfa_ptr, full_barriers[stage_idx],
                                        smem_sfa[stage_idx] + (kb * kNumL2SFAGroups + g) * kL2SFAHalfStride,
                                        m_idx, (k_block_begin + k_block_idx + kb) * kSFRowsPerKBlock + g, 1);
                                }
                            }
                            full_barriers[stage_idx]->arrive_and_expect_tx(
                                num_stage_blocks * (SMEM_A_SIZE_PER_KBLOCK +
                                                    kSFRowsPerKBlock * BLOCK_M * sizeof(float)));
                        } else {
                            // TMA load A
                            tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                                tensor_map_a_ptr, full_barriers[stage_idx], smem_a[stage_idx],
                                k_idx, m_idx, 1);
                            // BN128 L1 produces per-64 activation scales. L2
                            // consumes both scale groups for each BK128 tile.
                            tma::copy<BLOCK_M, 1, 0, float>(
                                tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx],
                                m_idx, (k_block_begin + k_block_idx) * 2, 1);
                            tma::copy<BLOCK_M, 1, 0, float>(
                                tensor_map_sfa_ptr, full_barriers[stage_idx],
                                smem_sfa[stage_idx] + kL2SFAHalfStride,
                                m_idx, (k_block_begin + k_block_idx) * 2 + 1, 1);
                            full_barriers[stage_idx]->arrive_and_expect_tx(
                                SMEM_A_SIZE_PER_STAGE + 2 * BLOCK_M * sizeof(float));
                        }
                    } else {
                        full_barriers[stage_idx]->arrive();
                    }
                }
                __syncwarp();
            }
        };
        if constexpr (kTinyMGemv) {
            // TinyM GEMV: the math warps read weights and activations themselves
        } else if constexpr (kUseInterleavedScheduler) {
            for_each_published_block(load_a_task);
        } else {
            for_each_static_selected_block(load_a_task);
        }

    } else if (warp_idx == kNumDispatchWarps + 1) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        const auto load_b_task = [&](const auto& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx,
                                     const uint32_t& pool_block_idx,
                                     const uint32_t& valid_m,
                                     const uint32_t& k_split_idx,
                                     const uint32_t& num_k_splits,
                                     const uint32_t& k_block_begin,
                                     const uint32_t& first_worker_idx) {
            using BlockPhaseTag = std::remove_cv_t<std::remove_reference_t<decltype(block_phase)>>;
            constexpr bool kBlockIsL2 = BlockPhaseTag::value == fused_sched::BlockPhase::Linear2;
            const auto tensor_map_b_ptr = kBlockIsL2 ?
                &tensor_map_l2_weights : &tensor_map_l1_weights;
            constexpr uint32_t shape_n = kBlockIsL2 ? L2_SHAPE_N : L1_SHAPE_N;
            constexpr uint32_t shape_k = kBlockIsL2 ? L2_SHAPE_K : L1_SHAPE_K;
            // Dense tiles (MXFP4/QoQ): packed tile (expert, n256_block, k_block) lives at
            // ((expert * n256_blocks + n256_block) * k_blocks + k_block) * (256 * 80 B);
            // a BLOCK_N=128 kernel tile is the contiguous upper/lower 10 KB half.
            // Per-phase sub-tile geometry (L2 half-row tasks: 128-row L2 sub-tiles).
            constexpr uint32_t kPhaseSubTiles =
                kBlockIsL2 ? kL2SubTilesPerPacked : kSubTilesPerPacked;
            constexpr uint32_t kPhaseKBlockBytes =
                kBlockIsL2 ? SMEM_PACKED_B_L2_SIZE_PER_KBLOCK : SMEM_PACKED_B_SIZE_PER_KBLOCK;
            const uint8_t* dense_tiles = nullptr;
            if constexpr (kDenseWeightTiles) {
                const uint32_t tile_row = local_expert_idx * (shape_n / kPackedTileN) +
                                          n_block_idx / kPhaseSubTiles;
                dense_tiles = reinterpret_cast<const uint8_t*>(kBlockIsL2 ? l2_weights_ptr : l1_weights_ptr) +
                    static_cast<size_t>(tile_row) * (shape_k / BLOCK_K) * kPackedTileBytes +
                    (n_block_idx % kPhaseSubTiles) * kPhaseKBlockBytes;
            }

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                // Partial last stage: only the remaining K-blocks are fetched.
                const uint32_t num_stage_blocks =
                    cute::min(kKBlocksPerStage, num_k_blocks - k_block_idx);
                const uint32_t stage_bytes = num_stage_blocks * kPhaseKBlockBytes;
                if (cute::elect_one_sync()) {
                    if constexpr (kDenseWeightTiles) {
                        // One 1D bulk copy per stage (20 KB per K-block for BN256;
                        // 40 / 80 KB when kKBlocksPerStage == 2 / 4, tiles k..k+3 are
                        // adjacent in the dense layout); decoders see 80 B rows at row * 80.
                        if constexpr (kKBlocksPerStage == 1 || kPhaseSubTiles == 1) {
                            ptx::tma_load_1d(smem_packed_b[stage_idx],
                                        dense_tiles + static_cast<size_t>(k_block_begin + k_block_idx) * kPackedTileBytes,
                                        full_barriers[stage_idx], stage_bytes);
                        } else {
                            // Half-tile / L2 half-row tasks: the stage's K-blocks are 10 KB
                            // sub-tiles one packed tile (20 KB) apart -> one bulk copy each
                            // (k, k+1 at +0 / +10 KB of the stage slot), one expect-tx.
                            #pragma unroll
                            for (uint32_t kb = 0; kb < kKBlocksPerStage; ++ kb) {
                                if (kb >= num_stage_blocks) break;
                                ptx::tma_load_1d(
                                    reinterpret_cast<uint8_t*>(smem_packed_b[stage_idx]) +
                                        kb * kPhaseKBlockBytes,
                                    dense_tiles + static_cast<size_t>(k_block_begin + k_block_idx + kb) * kPackedTileBytes,
                                    full_barriers[stage_idx], kPhaseKBlockBytes);
                            }
                        }
                    } else {
                        const uint32_t n_idx = local_expert_idx * shape_n + n_block_idx * BLOCK_N;
                        // NVFP4 fused B+scale layout stores 64B packed FP4 + 8B
                        // UE4M3 scale + 8B zero padding per BK128 row.
                        const uint32_t k_idx = (k_block_begin + k_block_idx) * B_LOAD_BYTES_PER_ROW;
                        tma::copy<B_LOAD_BYTES_PER_ROW, LOAD_BLOCK_N, 0, b_dtype_t>(
                            tensor_map_b_ptr, full_barriers[stage_idx],
                            smem_packed_b[stage_idx],
                            k_idx, n_idx, 1);
                    }
                    full_barriers[stage_idx]->arrive_and_expect_tx(
                        kDenseWeightTiles ? stage_bytes : SMEM_PACKED_B_SIZE_PER_STAGE);
                }
                __syncwarp();
            }
        };
        if constexpr (kTinyMGemv) {
            // TinyM GEMV: no task pipeline (see above)
        } else if constexpr (kUseInterleavedScheduler) {
            produce_interleaved_blocks(load_b_task);
        } else {
            for_each_static_selected_block(load_b_task);
        }

    } else {
        // =====================================================================
        // ROLE 3: MATH WARPGROUPS (WGMMA + epilogue + combine)
        // =====================================================================
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();

        const uint32_t epilogue_warp_idx  = warp_idx - (kNumDispatchWarps + kNumMMANonEpilogueWarps);
        const uint32_t epilogue_wg_idx    = epilogue_warp_idx / 4;
        const uint32_t epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
        const uint32_t warp_idx_in_wg     = epilogue_warp_idx % 4;

        const auto arrive_empty_barrier = [&](const uint32_t& s) {
            if (lane_idx == 0)
                empty_barriers[s]->arrive();
        };

        const auto notify_l1_ready = [&](const uint32_t& ready_pool_block_idx,
                                         const uint32_t& ready_n_block_idx) {
            if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                ptx::red_or_rel_gpu(
                    workspace.get_l2_arrival_mask_ptr(ready_pool_block_idx),
                    1ull << ready_n_block_idx);
            }
            __syncwarp();
        };

        // WGMMA-output register layout helpers
        const uint32_t row_idx = lane_idx / 4;
        const uint32_t col_idx = lane_idx % 4;
        const uint32_t r_0 = warp_idx_in_wg * 16 + row_idx;
        const uint32_t r_1 = r_0 + 8;

        DG_STATIC_ASSERT(kSwapABRequested ||
                         (WG_BLOCK_M == L1WGMMA::M and WG_BLOCK_N == L1WGMMA::N),
                         "Split-N WGs must each run one M64N128 WGMMA per K-block");

        // Sync with dispatch
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Fine-grained combine: this CTA's mailbox producer sequence (CTA-uniform;
        // only epilogue thread 0 writes). Baseline = the word's value left by the
        // previous launch (ordered by the kernel boundary; the consumer reads the
        // same baseline from its own word).
        uint32_t combine_mailbox_seq = 0u;
        if constexpr (kFineCombine)
            combine_mailbox_seq = *workspace.get_combine_mailbox_ptr(sm_idx);
        const auto run_math_task_impl = [&](const auto& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx,
                                     const uint32_t& pool_block_idx,
                                     const uint32_t& valid_m,
                                     const uint32_t& k_split_idx,
                                     const uint32_t& num_k_splits,
                                     const uint32_t& k_block_begin,
                                     const uint32_t& first_worker_idx) {
            using BlockPhaseTag = std::remove_cv_t<std::remove_reference_t<decltype(block_phase)>>;
            constexpr bool kBlockIsL2 = BlockPhaseTag::value == fused_sched::BlockPhase::Linear2;
            // Per-phase task geometry. L2 half-row tasks: 128-row tasks, WG w owns rows
            // [64w, 64w + 64) == ONE 64-row weight half (kWGHalves == 1) of the 10 KB
            // packed sub-tile per K-block; every other case keeps the 256-row task with
            // WG_BLOCK_N rows (two halves) per WG.
            constexpr uint32_t kTaskBlockN = kBlockIsL2 ? TASK_BLOCK_N_L2 : TASK_BLOCK_N;
            constexpr uint32_t kWGBlockN = kBlockIsL2 ? L2_WG_BLOCK_N : WG_BLOCK_N;
            constexpr uint32_t kWGHalves = kWGBlockN / 64;
            constexpr uint32_t kPackedBKBlockBytes =
                kBlockIsL2 ? SMEM_PACKED_B_L2_SIZE_PER_KBLOCK : SMEM_PACKED_B_SIZE_PER_KBLOCK;
            DG_STATIC_ASSERT(kWGHalves >= 1 && kWGHalves <= kSwapABWeightHalves,
                             "Per-WG weight halves must fit the swapAB accumulator layout");
            const uint32_t m_idx = pool_block_idx * BLOCK_M;
            // Half-tile tasks: both WGs work on the task's 128 rows (K-split), so
            // the per-WG N offsets are zero and only WG0 owns the epilogue.
            const uint32_t wg_n_idx =
                (kSplitMDecodedWeightReuse || kHalfTileTasks) ? 0u : epilogue_wg_idx * kWGBlockN;
            const uint32_t wg_l1_out_n_idx =
                (kSplitMDecodedWeightReuse || kHalfTileTasks) ? 0u : epilogue_wg_idx * WG_L1_OUT_BLOCK_N;
            const uint32_t n_idx = n_block_idx * kTaskBlockN + wg_n_idx;
            const uint32_t ksplit_kb = kHalfTileTasks ? epilogue_wg_idx : 0u;
            const bool is_epilogue_wg = !kHalfTileTasks || epilogue_wg_idx == 0;
            const uint32_t row_block_offset =
                kSplitMDecodedWeightReuse ? epilogue_wg_idx * WG_BLOCK_M : 0u;
            // Fine-grained combine: called by every epilogue thread right after the
            // CTA-wide sync that follows the L2 NVLink scatter of this task. Thread 0
            // posts (pool block, valid rows) to the CTA mailbox with st.release.gpu;
            // the bar.sync made every thread's scatter stores happen-before that
            // release, and the dispatch consumer's acquire + sys-scope release make
            // them visible to the remote combine warp (cumulativity).
            const auto signal_combine_arrivals = [&]() {
                if constexpr (kFineCombine) {
                    if (epilogue_thread_idx == 0 && valid_m > 0) {
                        auto* mailbox = workspace.get_combine_mailbox_ptr(sm_idx);
                        while (combine_mailbox_seq - ptx::ld_volatile(mailbox + 1) >= fused_layout::kSM90FineCombineRingSize) {}
                        mailbox[4 + (combine_mailbox_seq & (fused_layout::kSM90FineCombineRingSize - 1))] =
                            pool_block_idx | (valid_m << 24);
                        ptx::st_rel_gpu(mailbox, combine_mailbox_seq + 1);
                        ++ combine_mailbox_seq;
                    }
                }
            };
            const uint32_t row_offset_r0 = row_block_offset + r_0;
            const uint32_t row_offset_r1 = row_block_offset + r_1;
            // NVFP4: one scale per expert. MXFP4: one scale per weight row
            // (2^e_ref folded with the global scale), indexed by the L2 output
            // column `n_idx + col` within this expert.
            const float l2_global_scale = (kPerRowEpilogueScale || l2_global_scales == nullptr) ?
                1.0f : __ldg(l2_global_scales + local_expert_idx);
            const float* __restrict__ l2_row_scales = kPerRowEpilogueScale ?
                l2_global_scales + local_expert_idx * L2_SHAPE_N + n_idx : nullptr;
            const auto l2_scale_at = [&](const uint32_t& col) -> float {
                if constexpr (kPerRowEpilogueScale)
                    return __ldg(l2_row_scales + col);
                else
                    return l2_global_scale;
            };
            const auto cast_l2_scaled_bf16_pair = [&](float x, float y, const uint32_t& col) -> uint32_t {
                if constexpr (kPerRowEpilogueScale) {
                    const float2 sc = __ldg(reinterpret_cast<const float2*>(l2_row_scales + col));
                    x *= sc.x;
                    y *= sc.y;
                } else {
                    x *= l2_global_scale;
                    y *= l2_global_scale;
                }
                return math::cast_into_bf16_and_pack(x, y);
            };

            // ---------------- GEMM ----------------
            using WGMMA = L1WGMMA;
            constexpr uint32_t kAccumPerThread = WGMMA::kNumAccum;  // 64 for M=64,N=128
            float final_accum[kAccumPerThread] = {};
            const auto decode_b_stage = [&](const uint32_t& decoded_stage) {
                if constexpr (kSplitMDecodedWeightReuse) {
                    // Both M64 consumers share the same decoded N128 tile.
                    // The two physical decoded slots remove the overwrite
                    // hazard; the trailing pair barrier publishes K+1 before
                    // either consumer issues its WGMMA.
                    DG_STATIC_ASSERT(kUseMode2RowDecoder,
                                     "BM128 split-M uses the cooperative Mode2 decoder");
                    deep_gemm::nvfp4::
                        dequant_smem_b_from_packed_mode2_nibble_split_m<kMXFP4>(
                            reinterpret_cast<uint8_t*>(smem_b[decoded_stage]),
                            reinterpret_cast<const uint8_t*>(smem_packed_b[decoded_stage]),
                            epilogue_thread_idx, smem_nvfp4_lut);
                    cutlass::arch::fence_view_async_shared();
                    asm volatile("bar.sync %0, %1;" : :
                                 "n"(kSplitMDecodeBarrierIdx), "n"(256) : "memory");
                } else {
                    if constexpr (kQoQ) {
                        deep_gemm::nvfp4::dequant_smem_b_from_packed_qoq_shiftxor(
                            reinterpret_cast<uint8_t*>(smem_b[decoded_stage]),
                            reinterpret_cast<const uint8_t*>(smem_packed_b[decoded_stage]),
                            epilogue_thread_idx);
                    } else if constexpr (kUseMode2RowDecoder) {
                        deep_gemm::nvfp4::dequant_smem_b_from_packed_mode2_nibble<
                            kQuadDequantIlp, kMXFP4>(
                            reinterpret_cast<uint8_t*>(smem_b[decoded_stage]),
                            reinterpret_cast<const uint8_t*>(smem_packed_b[decoded_stage]),
                            epilogue_thread_idx, smem_nvfp4_lut);
                    } else {
                        deep_gemm::nvfp4::dequant_smem_b_from_packed_braided_lut_window<
                            kQuadDequantIlp, kMXFP4>(
                            reinterpret_cast<uint8_t*>(smem_b[decoded_stage]),
                            reinterpret_cast<const uint8_t*>(smem_packed_b[decoded_stage]),
                            epilogue_thread_idx, smem_nvfp4_lut);
                    }
                    cutlass::arch::fence_view_async_shared();
                    ptx::sync_aligned(
                        128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                }
            };
            // Stage-level probe (SM0 / thread 0 only; phase_stamps slots 17..22 + 30/31, SM cycles):
            //   17 full-barrier wait | 18 RF decode+LUT (both halves) | 19 wgmma arrive->drain
            //   21 #L1 stages | 22 head-to-head stage total (excl. last stage of a task)
            //   (kKBlocksPerStage == 2: all per *2-K128-block* stage; 18 = both decodes,
            //    19 = wait<1> + wait<0> drains; 21 counts 2-block stages, 12/L1 task,
            //    or 6 per L1 K-half task when kSplitKL1 is active)
            const bool kstage_probe_on =
                (phase_stamps != nullptr) && (sm_idx == 0) && (epilogue_thread_idx == 0);
            unsigned long long kstage_t_prev = 0;
            // Timing-only experiments (numerics invalid), bitmask in phase_stamps[24]:
            // 1 skip decode | 2 skip wgmma issue | 4 skip k+1 barrier check | 8 skip promote.
            const uint32_t kexp = (phase_stamps != nullptr) ?
                static_cast<uint32_t>(phase_stamps[24]) : 0u;
            const auto kstage_add = [&](const uint32_t slot, const unsigned long long& v) {
                if (kstage_probe_on) atomicAdd(phase_stamps + slot, v);
            };
            // Non-blocking mbarrier phase check. The blocking `wait()` costs ~250 ns
            // even when the phase has already completed (suspend/wake path), and the
            // k+1 tile is essentially always landed by the time the math warps ask
            // for it (probe slot 23), so test first and only block on a miss.
            const auto barrier_ready = [](Barrier* bar, const uint32_t& parity) -> bool {
                uint32_t ready = 0;
                asm volatile(
                    "{\n .reg .pred P;\n"
                    " mbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2;\n"
                    " selp.u32 %0, 1, 0, P;\n}"
                    : "=r"(ready)
                    : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(bar))), "r"(parity)
                    : "memory");
                return ready != 0;
            };

            // Software-pipelined swapAB main loop (kSwapPipelineDecode): the WGMMAs
            // of stage k for both weight halves are issued asynchronously, stage
            // k+1 is decoded while the tensor cores run, then k is waited on,
            // promoted and released. The generic loop below decodes, runs and
            // waits serially per stage.
            // RF decode (kRFDecode, MXFP4/QoQ swapAB): dedicated per-stage loop shared by
            // L1 and L2 (both promote with the per-token K128 activation SF; the
            // MXFP4 row scale is folded into the FP8 fragments by the LUT, the QoQ
            // integer s2[row] (byte 64 of the packed row) rides the promote). Thread
            // (warp w, lane l) owns rows r_0 = 16w + l/4 and r_1 = r_0 + 8 of each
            // 64-row weight half; column c = l % 4 owns K = 4c..4c+3 and 16+4c..16+4c+3
            // of every K32 step, i.e. exactly word c of each 16-byte quad in the
            // RF-ordered packed row (one uint4 per row covers all four K32 steps).
            // Decoded bytes are identical to the SMEM tile decoder's (bit-exact vs SS).
            // Per stage: all smem loads for both halves first, then the 16 LUT gathers,
            // then the decode; both halves' eight m64nNk32 RS WGMMAs share ONE commit
            // group and one drain.
            if constexpr (kSwapABRequested && kRFDecode) {
                DG_STATIC_ASSERT(kSwapABWeightHalves == 2, "RF decode expects two 64-row weight halves");
                auto run_swap_ab_rf = [&]<uint32_t N_SWAP>() {
                    // QoQ: int8 RS atoms have no N=24; pad to 32 like the SS path (extra
                    // token columns are masked by `token < valid_m`).
                    using SwapRS = typename std::conditional_t<kQoQ,
                        mma::sm90::INT8MMARSSelector<(N_SWAP == 24 ? 32 : N_SWAP)>,
                        mma::sm90::FP8MMARSSelector<N_SWAP>>::type;
                    DG_STATIC_ASSERT(BLOCK_K / SwapRS::K == 4, "Expects 4 K32 steps per K128");
                    constexpr uint32_t kSwapAccum = SwapRS::kNumAccum;
                    // One accumulator set per K-block of a stage: the two K128 blocks
                    // carry different per-token activation scales, so they are promoted
                    // separately (the RS WGMMAs of both run concurrently).
                    // Half-tile tasks: each WG owns ONE K-block per stage (ksplit_kb).
                    // >2 K-blocks per stage: two accumulator sets alternate (acc[b & 1]);
                    // block b is promoted right after wait<1> following issue(b+1), which
                    // frees its set for block b+2 (4 sets = +32 regs would spill).
                    constexpr uint32_t kNumAccKBlocks =
                        kHalfTileTasks ? 1u : cute::min(kKBlocksPerStage, 2u);
                    // Per-64 L2 activation scales (kHalfTileTasks): K32 steps {0,1} and
                    // {2,3} of a K128 block accumulate separately (same commit group)
                    // and are promoted with their own SF row.
                    constexpr uint32_t kSFGroups = (kBlockIsL2 && kL2ActsSFGranK == 64u) ? 2u : 1u;
                    constexpr uint32_t kK32PerSFGroup = 4u / kSFGroups;
                    // Two accumulator chains per (SF group, half): K32 steps alternate
                    // between them so the 4 dependent RS-WGMMAs of a K128 block become
                    // two independent 2-deep chains (tensor-core latency exposed once
                    // less per block); the chains are summed at promote time.
                    constexpr uint32_t kAccChains = 2u;
                    swap_accum_t swap_accum[kNumAccKBlocks][kSFGroups][kWGHalves][kAccChains][kSwapAccum];
                    uint32_t frag[2][kWGHalves][4][4];  // [buffer][half][k32 step][a0..a3]

                    const auto fence_accum = [&]() {
                        #pragma unroll
                        for (uint32_t kb = 0; kb < kNumAccKBlocks; ++ kb) {
                            #pragma unroll
                            for (uint32_t g = 0; g < kSFGroups; ++ g) {
                                #pragma unroll
                                for (uint32_t h = 0; h < kWGHalves; ++ h) {
                                    #pragma unroll
                                    for (uint32_t c = 0; c < kAccChains; ++ c) {
                                        #pragma unroll
                                        for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                            ptx::warpgroup_fence_operand(swap_accum[kb][g][h][c][i]);
                                    }
                                }
                            }
                        }
                    };
                    const auto fence_frag = [&](uint32_t (&f)[kWGHalves][4][4]) {
                        #pragma unroll
                        for (uint32_t h = 0; h < kWGHalves; ++ h) {
                            #pragma unroll
                            for (uint32_t k = 0; k < 4; ++ k) {
                                #pragma unroll
                                for (uint32_t i = 0; i < 4; ++ i)
                                    ptx::warpgroup_fence_operand(reinterpret_cast<float&>(f[h][k][i]));
                            }
                        }
                    };
                    // Decode both 64-row halves of K-block `kb` of `stage` into `f` (this
                    // thread's A fragments for the 4 K32 steps x 2 halves).
                    const auto decode_stage_rf = [&](const uint32_t& stage, const uint32_t& kb,
                                                     uint32_t (&f)[kWGHalves][4][4]) {
                        const auto* packed_rows =
                            reinterpret_cast<const uint8_t*>(smem_packed_b[stage]) +
                            kb * kPackedBKBlockBytes;
                        uint4 w[kWGHalves][2];
                        uint32_t sw[kWGHalves][2];
                        #pragma unroll
                        for (uint32_t h = 0; h < kWGHalves; ++ h) {
                            const uint32_t row_0 = wg_n_idx + h * 64u + r_0;
                            const uint32_t row_1 = row_0 + 8u;
                            w[h][0] = *reinterpret_cast<const uint4*>(
                                packed_rows + row_0 * 80u + col_idx * 16u);
                            w[h][1] = *reinterpret_cast<const uint4*>(
                                packed_rows + row_1 * 80u + col_idx * 16u);
                            sw[h][0] = *reinterpret_cast<const uint32_t*>(packed_rows + row_0 * 80u + 64u);
                            sw[h][1] = *reinterpret_cast<const uint32_t*>(packed_rows + row_1 * 80u + 64u);
                        }
                        if constexpr (kQoQ) {
                            // QoQ: pure ALU decode, no LUT gather. Word (K32 step k, column c)
                            // holds K 4c..4c+3 in its high nibbles and K 16+4c..16+4c+3 in its
                            // low nibbles (host `_mxfp4_rf_fragment_order`, plain nibbles, no
                            // braid); byte b <-> K +b. Same borrow-guarded per-byte subtract
                            // as `dequant_smem_b_from_packed_qoq_shiftxor`: (code - z) int8,
                            // bit-exact vs the SS tile decoder. z = byte 65 of the packed row.
                            uint32_t zz[kWGHalves][2];
                            #pragma unroll
                            for (uint32_t h = 0; h < kWGHalves; ++ h) {
                                zz[h][0] = ((sw[h][0] >> 8u) & 0xffu) * 0x01010101u;
                                zz[h][1] = ((sw[h][1] >> 8u) & 0xffu) * 0x01010101u;
                            }
                            const auto zsub = [](const uint32_t& nib, const uint32_t& z) -> uint32_t {
                                return ((nib | 0x80808080u) - z) ^ 0x80808080u;
                            };
                            #pragma unroll
                            for (uint32_t h = 0; h < kWGHalves; ++ h) {
                                #pragma unroll
                                for (uint32_t k = 0; k < 4; ++ k) {
                                    const uint32_t word_r0 = k == 0 ? w[h][0].x : k == 1 ? w[h][0].y :
                                                             k == 2 ? w[h][0].z : w[h][0].w;
                                    const uint32_t word_r1 = k == 0 ? w[h][1].x : k == 1 ? w[h][1].y :
                                                             k == 2 ? w[h][1].z : w[h][1].w;
                                    // a0: (r_0, K 4c..), a1: (r_1, K 4c..), a2: (r_0, K 16+4c..), a3: (r_1, K 16+4c..)
                                    f[h][k][0] = zsub((word_r0 >> 4) & 0x0f0f0f0fu, zz[h][0]);
                                    f[h][k][1] = zsub((word_r1 >> 4) & 0x0f0f0f0fu, zz[h][1]);
                                    f[h][k][2] = zsub(word_r0 & 0x0f0f0f0fu, zz[h][0]);
                                    f[h][k][3] = zsub(word_r1 & 0x0f0f0f0fu, zz[h][1]);
                                }
                            }
                        } else {
                        uint2 lut[kWGHalves][2][4];
                        #pragma unroll
                        for (uint32_t h = 0; h < kWGHalves; ++ h) {
                            #pragma unroll
                            for (uint32_t r = 0; r < 2; ++ r) {
                                #pragma unroll
                                for (uint32_t k = 0; k < 4; ++ k)
                                    lut[h][r][k] = smem_nvfp4_lut[(sw[h][r] >> (k * 8u)) & 0x7fu];
                            }
                        }
                        #pragma unroll
                        for (uint32_t h = 0; h < kWGHalves; ++ h) {
                            #pragma unroll
                            for (uint32_t k = 0; k < 4; ++ k) {
                                const uint32_t word_r0 = k == 0 ? w[h][0].x : k == 1 ? w[h][0].y :
                                                         k == 2 ? w[h][0].z : w[h][0].w;
                                const uint32_t word_r1 = k == 0 ? w[h][1].x : k == 1 ? w[h][1].y :
                                                         k == 2 ? w[h][1].z : w[h][1].w;
                                const uint2 d0 = deep_gemm::nvfp4::dequant_mode2_nibble_word(word_r0, lut[h][0][k]);
                                const uint2 d1 = deep_gemm::nvfp4::dequant_mode2_nibble_word(word_r1, lut[h][1][k]);
                                // a0: (r_0, K 4c..), a1: (r_1, K 4c..), a2: (r_0, K 16+4c..), a3: (r_1, K 16+4c..)
                                f[h][k][0] = d0.x; f[h][k][1] = d1.x;
                                f[h][k][2] = d0.y; f[h][k][3] = d1.y;
                            }
                        }
                        }
                    };
                    // One commit group for K-block `kb` of `stage`: 4 K32 steps into
                    // acc[0] with f[0], then into acc[1] with f[1]. The B (activation)
                    // descriptor addresses the kb-th 1 KB swizzled A tile of the stage.
                    const auto issue_stage_rf = [&](const uint32_t& stage, const uint32_t& kb,
                                                    uint32_t (&f)[kWGHalves][4][4],
                                                    swap_accum_t (&acc)[kSFGroups][kWGHalves][kAccChains][kSwapAccum]) {
                        fence_accum();
                        fence_frag(f);
                        ptx::warpgroup_arrive();
                        #pragma unroll
                        for (uint32_t h = 0; h < kWGHalves; ++ h) {
                            #pragma unroll
                            for (uint32_t k = 0; k < 4; ++ k) {
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_a[stage] + kb * (SMEM_A_SIZE_PER_KBLOCK / sizeof(a_dtype_t)) +
                                    k * SwapRS::K, 1);
                                SwapRS::wgmma(f[h][k], desc_b,
                                              acc[k / kK32PerSFGroup][h][(k % kK32PerSFGroup) % kAccChains],
                                              ((k % kK32PerSFGroup) / kAccChains) > 0);
                            }
                        }
                        ptx::warpgroup_commit_batch();
                    };
                    // Promote K-block `kb` of `stage` with its per-token K128 activation SF.
                    // QoQ: also by the per-row/K128 integer s2 (byte 64 of the packed row,
                    // still resident in this stage); the two int32 chains are summed
                    // exactly, so the result is bit-exact vs the SS path's
                    // (scale * s2) * float(acc).
                    const auto promote_stage_rf = [&](const uint32_t& stage, const uint32_t& kb,
                                                      const swap_accum_t (&acc)[kSFGroups][kWGHalves][kAccChains][kSwapAccum]) {
                        #pragma unroll
                        for (uint32_t g = 0; g < kSFGroups; ++ g) {
                        const float* sfa = smem_sfa[stage] + (kb * kNumL2SFAGroups + g) * kL2SFAHalfStride;
                        #pragma unroll
                        for (uint32_t half = 0; half < kWGHalves; ++ half) {
                            float s2_r0 = 1.0f, s2_r1 = 1.0f;
                            if constexpr (kQoQ) {
                                // s2[row] = byte 64 of the packed row (meta word bytes 64..67):
                                // one LDS.32 per row and an exact u8 -> float via the 2^23 magic
                                // constant. The previous generic u8 load + I2F pair (quarter-rate
                                // XU op right on the promote critical path) measured 475 ns per
                                // 2-block stage against 151 ns for the MXFP4 promote (slot 30).
                                const auto* packed_rows =
                                    reinterpret_cast<const uint8_t*>(smem_packed_b[stage]) + kb * kPackedBKBlockBytes;
                                const uint32_t m0 = ptx::ld_shared(reinterpret_cast<const uint32_t*>(
                                    packed_rows + (wg_n_idx + half * 64u + r_0) * 80u + 64u));
                                const uint32_t m1 = ptx::ld_shared(reinterpret_cast<const uint32_t*>(
                                    packed_rows + (wg_n_idx + half * 64u + r_1) * 80u + 64u));
                                s2_r0 = __uint_as_float(0x4B000000u | (m0 & 0xffu)) - 8388608.0f;
                                s2_r1 = __uint_as_float(0x4B000000u | (m1 & 0xffu)) - 8388608.0f;
                            }
                            #pragma unroll
                            for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                const uint32_t accum_offset = half * kSwapABHalfAccumPerThread + i * 4;
                                const uint32_t token_0 = i * 8 + col_idx * 2;
                                const uint32_t token_1 = token_0 + 1;
                                const auto acc_sum = [&](const uint32_t& j) -> float {
                                    swap_accum_t v = acc[g][half][0][j];
                                    #pragma unroll
                                    for (uint32_t c = 1; c < kAccChains; ++ c)
                                        v += acc[g][half][c][j];
                                    if constexpr (kQoQ) {
                                        // int32 -> float without I2F (quarter-rate on SM90; 32
                                        // per thread per K-block here). |v| <= 127 * 15 * 128 <
                                        // 2^22, so 1.5 * 2^23 + v is exact in fp32 and the
                                        // subtract recovers v exactly (bit-exact vs cvt.rn).
                                        return __int_as_float(0x4B400000 + v) - 12582912.0f;
                                    } else {
                                        return static_cast<float>(v);
                                    }
                                };
                                if constexpr (kQoQ) {
                                    // Branch-free: the token guards compiled to one BSSY/BRA/BSYNC
                                    // block per token pair, each with an exposed LDS -> FMUL -> FFMA
                                    // chain (SASS of the 09-09 build; slot-30 probe 525 ns/stage vs
                                    // 150 for MXFP4). Load both SFs unconditionally (the SFA stage
                                    // slot holds BLOCK_M >= token entries) and zero the scale of a
                                    // padded token instead: acc is a finite int32 sum, so 0 * acc == 0
                                    // and final_accum of the padded token stays untouched.
                                    const float raw_0 = ptx::ld_shared(sfa + token_0);
                                    const float raw_1 = ptx::ld_shared(sfa + token_1);
                                    const float scale_0 = token_0 < valid_m ? raw_0 : 0.0f;
                                    const float scale_1 = token_1 < valid_m ? raw_1 : 0.0f;
                                    final_accum[accum_offset + 0] += (scale_0 * s2_r0) * acc_sum(i * 4 + 0);
                                    final_accum[accum_offset + 2] += (scale_0 * s2_r1) * acc_sum(i * 4 + 2);
                                    final_accum[accum_offset + 1] += (scale_1 * s2_r0) * acc_sum(i * 4 + 1);
                                    final_accum[accum_offset + 3] += (scale_1 * s2_r1) * acc_sum(i * 4 + 3);
                                } else {
                                if (token_0 < valid_m) {
                                    const float scale_0 = ptx::ld_shared(sfa + token_0);
                                    final_accum[accum_offset + 0] += (scale_0 * s2_r0) * acc_sum(i * 4 + 0);
                                    final_accum[accum_offset + 2] += (scale_0 * s2_r1) * acc_sum(i * 4 + 2);
                                }
                                if (token_1 < valid_m) {
                                    const float scale_1 = ptx::ld_shared(sfa + token_1);
                                    final_accum[accum_offset + 1] += (scale_1 * s2_r0) * acc_sum(i * 4 + 1);
                                    final_accum[accum_offset + 3] += (scale_1 * s2_r1) * acc_sum(i * 4 + 3);
                                }
                                }
                            }
                        }
                        }
                    };

                    // Half-tile tasks use this one-K-block-per-WG loop too: WG w takes
                    // K-block ksplit_kb = w of every (2-K-block) stage.
                    if constexpr (kKBlocksPerStage == 1 || kHalfTileTasks) {
                    // Software pipeline: the 8 WGMMAs of stage k are issued
                    // asynchronously; while the tensor cores run, the math warps wait
                    // for stage k+1's full barrier (the loader runs kNumStages ahead)
                    // and decode its fragments into the other frag buffer. Only then
                    // is stage k drained, promoted and released. Frag buffer `fcur`
                    // stays live (fenced) until the drain, so its registers cannot be
                    // reused by the k+1 decode while the RS WGMMAs read them.
                    // Probe: 17 = exposed k+1 barrier wait, 18 = k+1 decode (overlapped
                    // with k's WGMMAs), 19 = exposed drain after the decode.
                    if (num_k_blocks > 0) {
                        const unsigned long long kt_head = clock64();
                        full_barriers[stage_idx]->wait(phase);
                        if constexpr (!kBlockIsL2)
                            kstage_add(17, clock64() - kt_head);
                        if constexpr (kUseInterleavedScheduler)
                            interleaved_scheduler.release_task_info(lane_idx);
                        decode_stage_rf(stage_idx, ksplit_kb, frag[0]);
                    }
                    const auto stage_step = [&](uint32_t& k_block_idx,
                                                uint32_t (&fcur)[kWGHalves][4][4],
                                                uint32_t (&fnext)[kWGHalves][4][4]) {
                        const unsigned long long kt_head = clock64();
                        const uint32_t cur_stage = stage_idx;
                        if constexpr (!kBlockIsL2) {
                            kstage_add(21, 1ull);
                            if (k_block_idx > 0)
                                kstage_add(22, kt_head - kstage_t_prev);
                            kstage_t_prev = kt_head;
                        }
                        if ((kexp & 2u) == 0u)
                            issue_stage_rf(cur_stage, ksplit_kb, fcur, swap_accum[0]);
                        unsigned long long kt_b = clock64();
                        if (k_block_idx + kKBlocksPerStage < num_k_blocks) {
                            const uint32_t next_stage = cur_stage == kNumStages - 1 ? 0 : cur_stage + 1;
                            const uint32_t next_phase = phase ^ (next_stage == 0);
                            if ((kexp & 4u) == 0u) {
                                if (!barrier_ready(full_barriers[next_stage], next_phase)) {
                                    if constexpr (!kBlockIsL2)
                                        kstage_add(23, 1ull);
                                    full_barriers[next_stage]->wait(next_phase);
                                }
                            }
                            const unsigned long long kt_a = clock64();
                            if constexpr (!kBlockIsL2)
                                kstage_add(17, kt_a - kt_b);
                            if ((kexp & 1u) == 0u)
                                decode_stage_rf(next_stage, ksplit_kb, fnext);
                            kt_b = clock64();
                            kstage_add(18, kt_b - kt_a);
                        }
                        fence_accum();
                        ptx::warpgroup_wait<0>();
                        fence_frag(fcur);
                        kstage_add(19, clock64() - kt_b);
                        if ((kexp & 8u) == 0u)
                            promote_stage_rf(cur_stage, ksplit_kb, swap_accum[0]);
                        arrive_empty_barrier(cur_stage);
                        advance_pipeline(k_block_idx);
                    };
                    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks;) {
                        stage_step(k_block_idx, frag[0], frag[1]);
                        if (k_block_idx < num_k_blocks)
                            stage_step(k_block_idx, frag[1], frag[0]);
                    }
                    } else if constexpr (kKBlocksPerStage == 2) {
                    // Two K128 blocks per stage, one commit group each, frag buffers at
                    // K-block granularity (frag[0] always holds block 0, frag[1] block 1):
                    //   issue(blk0 -> acc0); decode blk1 -> frag[1]; issue(blk1 -> acc1);
                    //   wait<1> (blk0 group done, frag[0] free);
                    //   wait stage k+1 full; decode its blk0 -> frag[0] (overlaps blk1 WGMMAs);
                    //   wait<0>; promote acc0 (SFA slot 0) + acc1 (SFA slot 1); release stage.
                    // Probe (per 2-block stage): 17 = exposed k+1 barrier wait, 18 = both
                    // decodes (blk1 of k + blk0 of k+1), 19 = exposed drains (wait<1> +
                    // wait<0>), 21 = stage count, 22 = head-to-head stage total.
                    if (num_k_blocks > 0) {
                        const unsigned long long kt_head = clock64();
                        full_barriers[stage_idx]->wait(phase);
                        if constexpr (!kBlockIsL2)
                            kstage_add(17, clock64() - kt_head);
                        if constexpr (kUseInterleavedScheduler)
                            interleaved_scheduler.release_task_info(lane_idx);
                        decode_stage_rf(stage_idx, ksplit_kb, frag[0]);
                    }
                    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks;) {
                        const unsigned long long kt_head = clock64();
                        const uint32_t cur_stage = stage_idx;
                        if constexpr (!kBlockIsL2) {
                            kstage_add(21, 1ull);
                            if (k_block_idx > 0)
                                kstage_add(22, kt_head - kstage_t_prev);
                            kstage_t_prev = kt_head;
                        }
                        if ((kexp & 2u) == 0u)
                            issue_stage_rf(cur_stage, 0, frag[0], swap_accum[0]);
                        unsigned long long kt_b = clock64();
                        kstage_add(31, kt_b - kt_head);
                        if ((kexp & 1u) == 0u)
                            decode_stage_rf(cur_stage, 1, frag[1]);
                        unsigned long long kt_a = clock64();
                        kstage_add(18, kt_a - kt_b);
                        if ((kexp & 2u) == 0u)
                            issue_stage_rf(cur_stage, 1, frag[1], swap_accum[1]);
                        kt_b = clock64();
                        kstage_add(31, kt_b - kt_a);
                        kt_a = kt_b;
                        // Block 0's group was issued before the block-1 decode; retire it
                        // so frag[0] can take the next stage's block 0.
                        fence_accum();
                        ptx::warpgroup_wait<1>();
                        fence_frag(frag[0]);
                        kt_b = clock64();
                        kstage_add(19, kt_b - kt_a);
                        if (k_block_idx + kKBlocksPerStage < num_k_blocks) {
                            const uint32_t next_stage = cur_stage == kNumStages - 1 ? 0 : cur_stage + 1;
                            const uint32_t next_phase = phase ^ (next_stage == 0);
                            if ((kexp & 4u) == 0u) {
                                if (!barrier_ready(full_barriers[next_stage], next_phase)) {
                                    if constexpr (!kBlockIsL2)
                                        kstage_add(23, 1ull);
                                    full_barriers[next_stage]->wait(next_phase);
                                }
                            }
                            kt_a = clock64();
                            if constexpr (!kBlockIsL2)
                                kstage_add(17, kt_a - kt_b);
                            if ((kexp & 1u) == 0u)
                                decode_stage_rf(next_stage, 0, frag[0]);
                            kt_b = clock64();
                            kstage_add(18, kt_b - kt_a);
                        }
                        fence_accum();
                        ptx::warpgroup_wait<0>();
                        fence_frag(frag[1]);
                        kt_a = clock64();
                        kstage_add(19, kt_a - kt_b);
                        if ((kexp & 8u) == 0u) {
                            promote_stage_rf(cur_stage, ksplit_kb, swap_accum[0]);
                            promote_stage_rf(cur_stage, 1, swap_accum[1]);
                        }
                        // 30 = both promotes (QoQ: s2 byte loads + int32 -> float + scale).
                        kstage_add(30, clock64() - kt_a);
                        arrive_empty_barrier(cur_stage);
                        advance_pipeline(k_block_idx);
                    }
                    } else {
                    // N (= kKBlocksPerStage > 2) K128 blocks per stage, one commit group
                    // each, frag buffers frag[b & 1] and accumulator sets acc[b & 1]:
                    //   for b in 0..n-1:
                    //     issue(b -> acc[b&1]);
                    //     b + 1 < n: decode blk b+1 -> frag[(b+1)&1] (overlaps b's WGMMAs);
                    //                issue(b+1 -> acc[(b+1)&1]); wait<1> (b done);
                    //                promote acc[b&1] (SFA slot b) -> free for b+2
                    //     b == n-1:  wait stage k+1 full; decode its blk0 -> frag[0]
                    //                (frag[0] is free: block n-2 retired by the last
                    //                wait<1>, n is even); wait<0>; promote acc[b&1]; release.
                    // The last stage of a task may be partial (n = min(N, remaining), even).
                    // Probe (per N-block stage): 17 = exposed k+1 barrier wait, 18 = all
                    // decodes, 19 = exposed drains (wait<1>s + wait<0>), 21 = stage count,
                    // 22 = head-to-head stage total.
                    DG_STATIC_ASSERT(kNumAccKBlocks == 2, "Generic multi-K-block loop alternates two accumulator sets");
                    if (num_k_blocks > 0) {
                        const unsigned long long kt_head = clock64();
                        full_barriers[stage_idx]->wait(phase);
                        if constexpr (!kBlockIsL2)
                            kstage_add(17, clock64() - kt_head);
                        if constexpr (kUseInterleavedScheduler)
                            interleaved_scheduler.release_task_info(lane_idx);
                        decode_stage_rf(stage_idx, 0, frag[0]);
                    }
                    // Stage body for a compile-time block count N (kKBlocksPerStage for
                    // full stages, 2 for the even partial tail) so every frag / acc index
                    // folds to a constant (a runtime block count spilled 16 B).
                    DG_STATIC_ASSERT(kKBlocksPerStage % 2 == 0, "Partial tail stages hold 2 blocks");
                    auto stage_step_n = [&]<uint32_t N>(uint32_t& k_block_idx) {
                        const unsigned long long kt_head = clock64();
                        const uint32_t cur_stage = stage_idx;
                        if constexpr (!kBlockIsL2) {
                            kstage_add(21, 1ull);
                            if (k_block_idx > 0)
                                kstage_add(22, kt_head - kstage_t_prev);
                            kstage_t_prev = kt_head;
                        }
                        if ((kexp & 2u) == 0u)
                            issue_stage_rf(cur_stage, 0, frag[0], swap_accum[0]);
                        unsigned long long kt_b = clock64(), kt_a = kt_b;
                        #pragma unroll
                        for (uint32_t b = 0; b + 1 < N; ++ b) {
                            // Not the last block of this stage: decode + issue b+1, retire b
                            // and promote it (frees frag[b&1] / acc[b&1] for block b+2).
                            if ((kexp & 1u) == 0u)
                                decode_stage_rf(cur_stage, b + 1, frag[(b + 1) & 1]);
                            kt_a = clock64();
                            kstage_add(18, kt_a - kt_b);
                            if ((kexp & 2u) == 0u)
                                issue_stage_rf(cur_stage, b + 1, frag[(b + 1) & 1], swap_accum[(b + 1) & 1]);
                            fence_accum();
                            ptx::warpgroup_wait<1>();
                            fence_frag(frag[b & 1]);
                            kt_b = clock64();
                            kstage_add(19, kt_b - kt_a);
                            if ((kexp & 8u) == 0u)
                                promote_stage_rf(cur_stage, b, swap_accum[b & 1]);
                        }
                        // Last block (N-1, odd): prefetch-decode the next stage's block 0
                        // into frag[0], then drain and promote it.
                        if (k_block_idx + kKBlocksPerStage < num_k_blocks) {
                            const uint32_t next_stage = cur_stage == kNumStages - 1 ? 0 : cur_stage + 1;
                            const uint32_t next_phase = phase ^ (next_stage == 0);
                            if ((kexp & 4u) == 0u) {
                                if (!barrier_ready(full_barriers[next_stage], next_phase)) {
                                    if constexpr (!kBlockIsL2)
                                        kstage_add(23, 1ull);
                                    full_barriers[next_stage]->wait(next_phase);
                                }
                            }
                            kt_a = clock64();
                            if constexpr (!kBlockIsL2)
                                kstage_add(17, kt_a - kt_b);
                            if ((kexp & 1u) == 0u)
                                decode_stage_rf(next_stage, 0, frag[0]);
                            kt_b = clock64();
                            kstage_add(18, kt_b - kt_a);
                        }
                        fence_accum();
                        ptx::warpgroup_wait<0>();
                        fence_frag(frag[(N - 1) & 1]);
                        kstage_add(19, clock64() - kt_b);
                        if ((kexp & 8u) == 0u)
                            promote_stage_rf(cur_stage, N - 1, swap_accum[(N - 1) & 1]);
                        arrive_empty_barrier(cur_stage);
                        advance_pipeline(k_block_idx);
                    };
                    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks;) {
                        if (num_k_blocks - k_block_idx >= kKBlocksPerStage)
                            stage_step_n.template operator()<kKBlocksPerStage>(k_block_idx);
                        else
                            stage_step_n.template operator()<2>(k_block_idx);
                    }
                    }
                };
                if constexpr (BLOCK_M == 8) {
                    run_swap_ab_rf.template operator()<8>();
                } else if constexpr (BLOCK_M == 16) {
                    const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                    if (n_swap <= 8) {
                        run_swap_ab_rf.template operator()<8>();
                    } else {
                        run_swap_ab_rf.template operator()<16>();
                    }
                } else if constexpr (BLOCK_M == 24) {
                    const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                    if (n_swap <= 8) {
                        run_swap_ab_rf.template operator()<8>();
                    } else if (n_swap <= 16) {
                        run_swap_ab_rf.template operator()<16>();
                    } else {
                        run_swap_ab_rf.template operator()<24>();
                    }
                }
            } else if constexpr (kSwapABRequested && kSwapPipelineDecode) {
                auto run_swap_ab_pipelined = [&]<uint32_t N_SWAP>() {
                    using SwapWGMMA = typename mma::sm90::FP8MMASelector<N_SWAP>::type;
                    constexpr uint32_t kSwapAccum = SwapWGMMA::kNumAccum;
                    float swap_accum[kSwapABWeightHalves][kSwapAccum];

                    if (num_k_blocks > 0) {
                        full_barriers[stage_idx]->wait(phase);
                        if constexpr (kUseInterleavedScheduler)
                            interleaved_scheduler.release_task_info(lane_idx);
                        decode_b_stage(stage_idx);
                    }
                    for (uint32_t k_block_idx = 0;
                         k_block_idx < num_k_blocks;
                         advance_pipeline(k_block_idx)) {
                        const uint32_t cur_stage = stage_idx;
                        #pragma unroll
                        for (uint32_t half = 0; half < kSwapABWeightHalves; ++ half) {
                            #pragma unroll
                            for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                ptx::warpgroup_fence_operand(swap_accum[half][i]);
                        }
                        ptx::warpgroup_arrive();
                        #pragma unroll
                        for (uint32_t half = 0; half < kSwapABWeightHalves; ++ half) {
                            #pragma unroll
                            for (uint32_t k = 0; k < BLOCK_K / SwapWGMMA::K; ++ k) {
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_b[cur_stage] + (wg_n_idx + half * 64u) * BLOCK_K + k * SwapWGMMA::K, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_a[cur_stage] + k * SwapWGMMA::K, 1);
                                SwapWGMMA::wgmma(desc_a, desc_b, swap_accum[half], k);
                            }
                        }
                        ptx::warpgroup_commit_batch();

                        // Decode the next stage while the WGMMAs of this one run.
                        if (k_block_idx + 1 < num_k_blocks) {
                            const uint32_t next_stage = cur_stage == kNumStages - 1 ? 0 : cur_stage + 1;
                            const uint32_t next_phase = phase ^ (next_stage == 0);
                            full_barriers[next_stage]->wait(next_phase);
                            decode_b_stage(next_stage);
                        }

                        #pragma unroll
                        for (uint32_t half = 0; half < kSwapABWeightHalves; ++ half) {
                            #pragma unroll
                            for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                ptx::warpgroup_fence_operand(swap_accum[half][i]);
                        }
                        ptx::warpgroup_wait<0>();

                        #pragma unroll
                        for (uint32_t half = 0; half < kSwapABWeightHalves; ++ half) {
                            #pragma unroll
                            for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                const uint32_t accum_offset = half * kSwapABHalfAccumPerThread + i * 4;
                                const uint32_t token_0 = i * 8 + col_idx * 2;
                                const uint32_t token_1 = token_0 + 1;
                                if (token_0 < valid_m) {
                                    const float scale_0 = ptx::ld_shared(smem_sfa[cur_stage] + token_0);
                                    final_accum[accum_offset + 0] += scale_0 * swap_accum[half][i * 4 + 0];
                                    final_accum[accum_offset + 2] += scale_0 * swap_accum[half][i * 4 + 2];
                                }
                                if (token_1 < valid_m) {
                                    const float scale_1 = ptx::ld_shared(smem_sfa[cur_stage] + token_1);
                                    final_accum[accum_offset + 1] += scale_1 * swap_accum[half][i * 4 + 1];
                                    final_accum[accum_offset + 3] += scale_1 * swap_accum[half][i * 4 + 3];
                                }
                            }
                        }
                        arrive_empty_barrier(cur_stage);
                    }
                };
                if constexpr (BLOCK_M == 8) {
                    run_swap_ab_pipelined.template operator()<8>();
                } else if constexpr (BLOCK_M == 16) {
                    const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                    if (n_swap <= 8) {
                        run_swap_ab_pipelined.template operator()<8>();
                    } else {
                        run_swap_ab_pipelined.template operator()<16>();
                    }
                } else if constexpr (BLOCK_M == 24) {
                    const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                    if (n_swap <= 8) {
                        run_swap_ab_pipelined.template operator()<8>();
                    } else if (n_swap <= 16) {
                        run_swap_ab_pipelined.template operator()<16>();
                    } else {
                        run_swap_ab_pipelined.template operator()<24>();
                    }
                }
            } else {
            for (uint32_t k_block_idx = 0;
                 k_block_idx < num_k_blocks;
                 advance_pipeline(k_block_idx)) {
                const unsigned long long kt_head = clock64();
                full_barriers[stage_idx]->wait(phase);
                if constexpr (!kBlockIsL2) {
                    kstage_add(17, clock64() - kt_head);
                    kstage_add(21, 1ull);
                    if (k_block_idx > 0)
                        kstage_add(22, kt_head - kstage_t_prev);
                    kstage_t_prev = kt_head;
                }
                if constexpr (kUseInterleavedScheduler) {
                    if (k_block_idx == 0)
                        interleaved_scheduler.release_task_info(lane_idx);
                }
                if constexpr (!kRFDecode)
                    decode_b_stage(stage_idx);

                // Read SF (must precede warpgroup_arrive)
                const float scale_a_0_lo =
                    ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
                const float scale_a_1_lo =
                    ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);
                float scale_a_0_hi = 0.0f;
                float scale_a_1_hi = 0.0f;
                if constexpr (kBlockIsL2 && kSplitMDecodedWeightReuse) {
                    scale_a_0_hi = ptx::ld_shared(
                        smem_sfa[stage_idx] + kL2SFAHalfStride + row_offset_r0);
                    scale_a_1_hi = ptx::ld_shared(
                        smem_sfa[stage_idx] + kL2SFAHalfStride + row_offset_r1);
                }

                // NVFP4 UE4M3 weight scales are applied during FP4 -> FP8 smem
                // expansion, so the WGMMA accumulator only needs activation SF.

                if constexpr (!kBlockIsL2) {
                    if constexpr (kSwapABRequested) {
                        auto run_swap_ab_l1 = [&]<uint32_t N_SWAP>() {
                            // INT8 atoms have no N=24; pad to 32 (extra token columns are
                            // masked by `token < valid_m`, smem A rows beyond BLOCK_M are
                            // never consumed).
                            using SwapWGMMA = typename std::conditional_t<kQoQ,
                                mma::sm90::INT8MMASelector<(N_SWAP == 24 ? 32 : N_SWAP)>,
                                mma::sm90::FP8MMASelector<N_SWAP>>::type;
                            constexpr uint32_t kSwapAccum = SwapWGMMA::kNumAccum;
                            swap_accum_t swap_accum[kSwapAccum];

                            #pragma unroll
                            for (uint32_t half = 0; half < kSwapABWeightHalves; ++ half) {
                                {
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(swap_accum[i]);
                                ptx::warpgroup_arrive();
                                #pragma unroll
                                for (uint32_t k = 0; k < BLOCK_K / SwapWGMMA::K; ++ k) {
                                    auto desc_a = mma::sm90::make_smem_desc(
                                        smem_b[stage_idx] + (wg_n_idx + half * 64u) * BLOCK_K + k * SwapWGMMA::K, 1);
                                    auto desc_b = mma::sm90::make_smem_desc(
                                        smem_a[stage_idx] + k * SwapWGMMA::K, 1);
                                    SwapWGMMA::wgmma(desc_a, desc_b, swap_accum, k);
                                }
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(swap_accum[i]);
                                ptx::warpgroup_wait<0>();
                                }

                                // QoQ: fold the per-row/K128 integer s2 (byte 64 of the
                                // packed row, still resident in this stage) into the promote.
                                float s2_r0 = 1.0f, s2_r1 = 1.0f;
                                if constexpr (kQoQ) {
                                    const auto* packed_rows = reinterpret_cast<const uint8_t*>(smem_packed_b[stage_idx]);
                                    s2_r0 = static_cast<float>(packed_rows[(wg_n_idx + half * 64u + r_0) * 80u + 64u]);
                                    s2_r1 = static_cast<float>(packed_rows[(wg_n_idx + half * 64u + r_1) * 80u + 64u]);
                                }
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                    const uint32_t accum_offset = half * kSwapABHalfAccumPerThread + i * 4;
                                    const uint32_t token_0 = i * 8 + col_idx * 2;
                                    const uint32_t token_1 = token_0 + 1;
                                    if (token_0 < valid_m) {
                                        const float scale_0 = ptx::ld_shared(smem_sfa[stage_idx] + token_0);
                                        final_accum[accum_offset + 0] += (scale_0 * s2_r0) * static_cast<float>(swap_accum[i * 4 + 0]);
                                        final_accum[accum_offset + 2] += (scale_0 * s2_r1) * static_cast<float>(swap_accum[i * 4 + 2]);
                                    }
                                    if (token_1 < valid_m) {
                                        const float scale_1 = ptx::ld_shared(smem_sfa[stage_idx] + token_1);
                                        final_accum[accum_offset + 1] += (scale_1 * s2_r0) * static_cast<float>(swap_accum[i * 4 + 1]);
                                        final_accum[accum_offset + 3] += (scale_1 * s2_r1) * static_cast<float>(swap_accum[i * 4 + 3]);
                                    }
                                }
                            }

                            arrive_empty_barrier(stage_idx);
                        };

                        if constexpr (BLOCK_M == 8) {
                            run_swap_ab_l1.template operator()<8>();
                        } else if constexpr (BLOCK_M == 16) {
                            const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                            if (n_swap <= 8) {
                                run_swap_ab_l1.template operator()<8>();
                            } else {
                                run_swap_ab_l1.template operator()<16>();
                            }
                        } else if constexpr (BLOCK_M == 24) {
                            const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                            if (n_swap <= 8) {
                                run_swap_ab_l1.template operator()<8>();
                            } else if (n_swap <= 16) {
                                run_swap_ab_l1.template operator()<16>();
                            } else {
                                run_swap_ab_l1.template operator()<24>();
                            }
                        }
                    } else {
                        float accum[kAccumPerThread];
                        // Single per-128 K-block WGMMA group
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                            ptx::warpgroup_fence_operand(accum[i]);
                        ptx::warpgroup_arrive();
                        #pragma unroll
                        for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                            auto desc_a = mma::sm90::make_smem_desc(
                                smem_a[stage_idx] + row_block_offset * BLOCK_K +
                                k * WGMMA::K, 1);
                            auto desc_b = mma::sm90::make_smem_desc(
                                smem_b[stage_idx] + wg_n_idx * BLOCK_K + k * WGMMA::K, 1);
                            WGMMA::wgmma(desc_a, desc_b, accum, k);
                        }
                        ptx::warpgroup_commit_batch();
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                            ptx::warpgroup_fence_operand(accum[i]);
                        ptx::warpgroup_wait<0>();

                        arrive_empty_barrier(stage_idx);

                        // L1: gate/up alternate at gran=8 along N; each `i` block
                        // of 8 cols belongs entirely to one of {gate, up}, so .x
                        // and .y share the same scalar.
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                            final_accum[i*4+0] += scale_a_0_lo * accum[i*4+0];
                            final_accum[i*4+1] += scale_a_0_lo * accum[i*4+1];
                            final_accum[i*4+2] += scale_a_1_lo * accum[i*4+2];
                            final_accum[i*4+3] += scale_a_1_lo * accum[i*4+3];
                        }
                    }
                } else {
                    if constexpr (kSwapABRequested) {
                        DG_STATIC_ASSERT(kL2ActsSFGranK == 128,
                                         "L2 swap-AB requires per-128 activation scales");
                        auto run_swap_ab_l2 = [&]<uint32_t N_SWAP>() {
                            using SwapWGMMA = typename std::conditional_t<kQoQ,
                                mma::sm90::INT8MMASelector<(N_SWAP == 24 ? 32 : N_SWAP)>,
                                mma::sm90::FP8MMASelector<N_SWAP>>::type;
                            constexpr uint32_t kSwapAccum = SwapWGMMA::kNumAccum;
                            swap_accum_t swap_accum[kSwapAccum];

                            #pragma unroll
                            for (uint32_t half = 0; half < kSwapABWeightHalves; ++ half) {
                                {
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(swap_accum[i]);
                                ptx::warpgroup_arrive();
                                #pragma unroll
                                for (uint32_t k = 0; k < BLOCK_K / SwapWGMMA::K; ++ k) {
                                    auto desc_a = mma::sm90::make_smem_desc(
                                        smem_b[stage_idx] + (wg_n_idx + half * 64u) * BLOCK_K + k * SwapWGMMA::K, 1);
                                    auto desc_b = mma::sm90::make_smem_desc(
                                        smem_a[stage_idx] + k * SwapWGMMA::K, 1);
                                    SwapWGMMA::wgmma(desc_a, desc_b, swap_accum, k);
                                }
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(swap_accum[i]);
                                ptx::warpgroup_wait<0>();
                                }
                                float s2_r0 = 1.0f, s2_r1 = 1.0f;
                                if constexpr (kQoQ) {
                                    const auto* packed_rows = reinterpret_cast<const uint8_t*>(smem_packed_b[stage_idx]);
                                    s2_r0 = static_cast<float>(packed_rows[(wg_n_idx + half * 64u + r_0) * 80u + 64u]);
                                    s2_r1 = static_cast<float>(packed_rows[(wg_n_idx + half * 64u + r_1) * 80u + 64u]);
                                }
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                    const uint32_t accum_offset =
                                        half * kSwapABHalfAccumPerThread + i * 4;
                                    const uint32_t token_0 = i * 8 + col_idx * 2;
                                    const uint32_t token_1 = token_0 + 1;
                                    if (token_0 < valid_m) {
                                        const float scale_0 = ptx::ld_shared(
                                            smem_sfa[stage_idx] + token_0);
                                        final_accum[accum_offset + 0] +=
                                            (scale_0 * s2_r0) * static_cast<float>(swap_accum[i * 4 + 0]);
                                        final_accum[accum_offset + 2] +=
                                            (scale_0 * s2_r1) * static_cast<float>(swap_accum[i * 4 + 2]);
                                    }
                                    if (token_1 < valid_m) {
                                        const float scale_1 = ptx::ld_shared(
                                            smem_sfa[stage_idx] + token_1);
                                        final_accum[accum_offset + 1] +=
                                            (scale_1 * s2_r0) * static_cast<float>(swap_accum[i * 4 + 1]);
                                        final_accum[accum_offset + 3] +=
                                            (scale_1 * s2_r1) * static_cast<float>(swap_accum[i * 4 + 3]);
                                    }
                                }
                            }

                            arrive_empty_barrier(stage_idx);
                        };

                        if constexpr (BLOCK_M == 8) {
                            run_swap_ab_l2.template operator()<8>();
                        } else if constexpr (BLOCK_M == 16) {
                            const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                            if (n_swap <= 8) {
                                run_swap_ab_l2.template operator()<8>();
                            } else {
                                run_swap_ab_l2.template operator()<16>();
                            }
                        } else if constexpr (BLOCK_M == 24) {
                            const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                            if (n_swap <= 8) {
                                run_swap_ab_l2.template operator()<8>();
                            } else if (n_swap <= 16) {
                                run_swap_ab_l2.template operator()<16>();
                            } else {
                                run_swap_ab_l2.template operator()<24>();
                            }
                        }
                    } else {
                        float accum[kAccumPerThread];
                        const auto promote_l2_accum = [&](const float& scale_r0,
                                                          const float& scale_r1) {
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                final_accum[i*4+0] += scale_r0 * accum[i*4+0];
                                final_accum[i*4+1] += scale_r0 * accum[i*4+1];
                                final_accum[i*4+2] += scale_r1 * accum[i*4+2];
                                final_accum[i*4+3] += scale_r1 * accum[i*4+3];
                            }
                        };
                        if constexpr (kSplitMDecodedWeightReuse) {
                            #pragma unroll
                            for (uint32_t sf_group = 0; sf_group < 2; ++ sf_group) {
                                #pragma unroll
                                for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                                    ptx::warpgroup_fence_operand(accum[i]);
                                ptx::warpgroup_arrive();
                                #pragma unroll
                                for (uint32_t k = 0;
                                     k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                                    const uint32_t k_off =
                                        sf_group * (BLOCK_K / 2) + k * WGMMA::K;
                                    auto desc_a = mma::sm90::make_smem_desc(
                                        smem_a[stage_idx] + row_block_offset * BLOCK_K + k_off, 1);
                                    auto desc_b = mma::sm90::make_smem_desc(
                                        smem_b[stage_idx] + wg_n_idx * BLOCK_K + k_off, 1);
                                    WGMMA::wgmma(desc_a, desc_b, accum, k);
                                }
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                                    ptx::warpgroup_fence_operand(accum[i]);
                                ptx::warpgroup_wait<0>();
                                if (sf_group == 0)
                                    promote_l2_accum(scale_a_0_lo, scale_a_1_lo);
                                else
                                    promote_l2_accum(scale_a_0_hi, scale_a_1_hi);
                            }
                            arrive_empty_barrier(stage_idx);
                        } else {
                            // One per-128 scale permits a single four-instruction
                            // WGMMA group and one accumulator promotion per K tile.
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                                ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + row_block_offset * BLOCK_K +
                                    k * WGMMA::K, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + wg_n_idx * BLOCK_K + k * WGMMA::K, 1);
                                WGMMA::wgmma(desc_a, desc_b, accum, k);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                                ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_wait<0>();
                            arrive_empty_barrier(stage_idx);
                            promote_l2_accum(scale_a_0_lo, scale_a_1_lo);
                        }
                    }
                }
            }
            }  // kSwapPipelineDecode

            // Skip epilogue when block is past valid M (the GEMM loop already
            // released its pipeline stages). Drain any prior L1 async store.
            if (valid_m == 0) {
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                return;
            }

            // Half-tile tasks: K-split hand-off. WG1 publishes its promoted partial
            // sums ([element][thread] floats, conflict-free), WG0 adds them and then
            // runs the epilogue alone; WG1 only keeps the CTA-wide barriers company.
            // WAR safety: WG1's next write happens after it passes this task's
            // epilogue-wide barriers, which WG0 reaches only after reading here.
            if constexpr (kHalfTileTasks) {
                const uint32_t tid_in_wg = epilogue_thread_idx & 127u;
                if (epilogue_wg_idx == 1) {
                    #pragma unroll
                    for (uint32_t h = 0; h < kSwapABWeightHalves; ++ h) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kSwapABTokenChunks; ++ i) {
                            #pragma unroll
                            for (uint32_t j = 0; j < 4; ++ j)
                                smem_ksplit_reduce[((h * kSwapABTokenChunks + i) * 4 + j) * 128u + tid_in_wg] =
                                    final_accum[h * kSwapABHalfAccumPerThread + i * 4 + j];
                        }
                    }
                }
                asm volatile("bar.sync %0, %1;" : :
                             "n"(kKSplitReduceBarrierIdx), "n"(kNumEpilogueThreads) : "memory");
                if (epilogue_wg_idx == 0) {
                    #pragma unroll
                    for (uint32_t h = 0; h < kSwapABWeightHalves; ++ h) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kSwapABTokenChunks; ++ i) {
                            #pragma unroll
                            for (uint32_t j = 0; j < 4; ++ j)
                                final_accum[h * kSwapABHalfAccumPerThread + i * 4 + j] +=
                                    smem_ksplit_reduce[((h * kSwapABTokenChunks + i) * 4 + j) * 128u + tid_in_wg];
                        }
                    }
                }
            }

            // Split-K L1 (kSplitKL1): cross-CTA reduction of the two K halves of this
            // (pool_block, n_block). Roles are fixed by the K half so no counter
            // round-trip sits on the critical path: half 0 (PUBLISHER) stores its
            // promoted partials to the scratch slot ([element][256 threads] floats,
            // coalesced), one thread releases the ready flag after the CTA barrier
            // (bar.sync + release.gpu is cumulative over the other threads' stores,
            // as in CUTLASS's split-K semaphore) and the CTA moves on: no epilogue,
            // no L1-ready notify (every CTA-wide barrier of the task was passed by
            // all 256 threads). Half 1 (FINISHER) acquires the flag (its publisher
            // was claimed one index earlier and never waits on L2 progress, so the
            // wait is short and deadlock-free), resets it for the next launch, adds
            // the partner's slot and runs the normal epilogue + notify.
            // valid_m == 0 tasks skipped this on both halves (flag untouched).
            // Split-K L2 (kSplitKL2) uses the identical protocol on the L2 tail tasks
            // (own slot/flag index space): the L2 accumulator is also two 64-row
            // weight halves x 8 tokens per thread, the finisher runs the L2 epilogue
            // (BF16 x scale, NVLink scatter). Deadlock-free for the same reason: the
            // publisher (index i) only waits on L1-ready bits, the finisher (i + 1)
            // only on the publisher; by induction on the claim order every earlier
            // task completes.
            // Stream-K (kStreamK, active launches): n-way generalisation. Every
            // contributor of a > 1-split tile stores its partial to its own worker
            // slot (slot 0 if its segment starts at K-block 0, else slot 1: a worker
            // has at most one of each per phase), then thread 0 takes an acq_rel
            // ticket on the tile's arrival counter and broadcasts it through smem.
            // Non-last arrivers leave (no epilogue); the last one (ticket ==
            // splits - 1) resets the counter for the next launch, sums the partials
            // in split order (contributor i == worker first + i, slot i == 0 ? 0 : 1;
            // its own from registers) and runs the epilogue. The partial store of
            // the eventual finisher is wasted (8 KB) but keeps the protocol
            // role-free. Deadlock-free: no contributor waits on anything here.
            if constexpr (kStreamK) {
                if (num_k_splits > 1) {
                    constexpr uint32_t kNumPartialElems = kSwapABWeightHalves * kSwapABTokenChunks * 4u;
                    DG_STATIC_ASSERT(!kStreamK ||
                                     kNumPartialElems * kNumEpilogueThreads * sizeof(float) ==
                                     fused_layout::kSM90SplitKL1PartialBytes,
                                     "Stream-K partial slot size mismatch");
                    DG_STATIC_ASSERT(!kStreamK || kNumSMs <= fused_layout::kSM90StreamKMaxSMs,
                                     "Stream-K per-worker slots do not cover the grid");
                    DG_STATIC_ASSERT(!kBlockIsL2 || kWGHalves == kSwapABWeightHalves,
                                     "Stream-K L2 expects both weight halves per WG");
                    const uint32_t worker_idx = first_worker_idx + k_split_idx;
                    float* my_slot = workspace.get_streamk_scratch_ptr(
                        kBlockIsL2, worker_idx, k_split_idx == 0u ? 0u : 1u);
                    uint32_t* counter = kBlockIsL2 ?
                        workspace.get_splitk_l2_flag_ptr(pool_block_idx, n_block_idx) :
                        workspace.get_splitk_l1_flag_ptr(pool_block_idx, n_block_idx);
                    #pragma unroll
                    for (uint32_t h = 0; h < kSwapABWeightHalves; ++ h) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kSwapABTokenChunks; ++ i) {
                            #pragma unroll
                            for (uint32_t j = 0; j < 4; ++ j)
                                __stcg(my_slot + ((h * kSwapABTokenChunks + i) * 4 + j) * kNumEpilogueThreads + epilogue_thread_idx,
                                       final_accum[h * kSwapABHalfAccumPerThread + i * 4 + j]);
                        }
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    if (epilogue_thread_idx == 0) {
                        const uint32_t ticket = ptx::atomic_add_acq_rel(counter, 1u);
                        if (ticket == num_k_splits - 1u)
                            *counter = 0u;
                        *smem_streamk_ticket = ticket;
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    const uint32_t ticket = *smem_streamk_ticket;
                    if (ticket != num_k_splits - 1u)
                        return;  // another contributor finishes this tile
                    float partial_sum[kNumPartialElems];
                    #pragma unroll
                    for (uint32_t e = 0; e < kNumPartialElems; ++ e)
                        partial_sum[e] = 0.0f;
                    for (uint32_t c = 0; c < num_k_splits; ++ c) {
                        if (c == k_split_idx) {
                            #pragma unroll
                            for (uint32_t h = 0; h < kSwapABWeightHalves; ++ h) {
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapABTokenChunks; ++ i) {
                                    #pragma unroll
                                    for (uint32_t j = 0; j < 4; ++ j)
                                        partial_sum[(h * kSwapABTokenChunks + i) * 4 + j] +=
                                            final_accum[h * kSwapABHalfAccumPerThread + i * 4 + j];
                                }
                            }
                        } else {
                            const float* slot = workspace.get_streamk_scratch_ptr(
                                kBlockIsL2, first_worker_idx + c, c == 0u ? 0u : 1u);
                            #pragma unroll
                            for (uint32_t e = 0; e < kNumPartialElems; ++ e)
                                partial_sum[e] += __ldcg(slot + e * kNumEpilogueThreads + epilogue_thread_idx);
                        }
                    }
                    #pragma unroll
                    for (uint32_t h = 0; h < kSwapABWeightHalves; ++ h) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kSwapABTokenChunks; ++ i) {
                            #pragma unroll
                            for (uint32_t j = 0; j < 4; ++ j)
                                final_accum[h * kSwapABHalfAccumPerThread + i * 4 + j] =
                                    partial_sum[(h * kSwapABTokenChunks + i) * 4 + j];
                        }
                    }
                }
            } else if constexpr ((kSplitKL1 && !kBlockIsL2) || (kSplitKL2 && kBlockIsL2)) {
                if (num_k_splits > 1) {
                    constexpr uint32_t kNumPartialElems = kSwapABWeightHalves * kSwapABTokenChunks * 4u;
                    // (Guarded: this discarded branch is not template-dependent.)
                    DG_STATIC_ASSERT(!(kSplitKL1 || kSplitKL2) ||
                                     kNumPartialElems * kNumEpilogueThreads * sizeof(float) ==
                                     fused_layout::kSM90SplitKL1PartialBytes,
                                     "Split-K partial slot size mismatch");
                    DG_STATIC_ASSERT(fused_layout::kSM90SplitKL1NumKSplits == 2,
                                     "Split-K handshake assumes one publisher and one finisher half");
                    DG_STATIC_ASSERT(!kBlockIsL2 || kWGHalves == kSwapABWeightHalves,
                                     "Split-K L2 expects both weight halves per WG");
                    float* slot = kBlockIsL2 ?
                        workspace.get_splitk_l2_scratch_ptr(pool_block_idx, n_block_idx) :
                        workspace.get_splitk_l1_scratch_ptr(pool_block_idx, n_block_idx);
                    uint32_t* flag = kBlockIsL2 ?
                        workspace.get_splitk_l2_flag_ptr(pool_block_idx, n_block_idx) :
                        workspace.get_splitk_l1_flag_ptr(pool_block_idx, n_block_idx);
                    if (k_split_idx == 0) {
                        #pragma unroll
                        for (uint32_t h = 0; h < kSwapABWeightHalves; ++ h) {
                            #pragma unroll
                            for (uint32_t i = 0; i < kSwapABTokenChunks; ++ i) {
                                #pragma unroll
                                for (uint32_t j = 0; j < 4; ++ j)
                                    __stcg(slot + ((h * kSwapABTokenChunks + i) * 4 + j) * kNumEpilogueThreads + epilogue_thread_idx,
                                           final_accum[h * kSwapABHalfAccumPerThread + i * 4 + j]);
                            }
                        }
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                        if (epilogue_thread_idx == 0)
                            ptx::red_add_rel(flag, 1u);
                        return;  // PUBLISHER: the finisher CTA completes this task
                    }
                    if (epilogue_thread_idx == 0) {
                        while (ptx::ld_acq(flag) == 0u) {}
                        *flag = 0u;
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    #pragma unroll
                    for (uint32_t h = 0; h < kSwapABWeightHalves; ++ h) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kSwapABTokenChunks; ++ i) {
                            #pragma unroll
                            for (uint32_t j = 0; j < 4; ++ j)
                                final_accum[h * kSwapABHalfAccumPerThread + i * 4 + j] +=
                                    __ldcg(slot + ((h * kSwapABTokenChunks + i) * 4 + j) * kNumEpilogueThreads + epilogue_thread_idx);
                        }
                    }
                }
            }

            if constexpr (!kBlockIsL2) {
                const float l1_global_scale = (kPerRowEpilogueScale || l1_global_scales == nullptr) ?
                    1.0f : __ldg(l1_global_scales + local_expert_idx);
                // MXFP4 per-weight-row scales, indexed by the interleaved L1
                // weight row (gate/up interleaved by 8) within this expert.
                const float* __restrict__ l1_row_scales = kPerRowEpilogueScale ?
                    l1_global_scales + local_expert_idx * L1_SHAPE_N + n_idx : nullptr;
                const auto l1_scale_at = [&](const uint32_t& row) -> float {
                    if constexpr (kPerRowEpilogueScale)
                        return __ldg(l1_row_scales + row);
                    else
                        return l1_global_scale;
                };
                // L2 activation byte: FP8 E4M3 (FP4 paths) or INT8 (QoQ).
                const auto quantize_l1_out_byte = [](const float& v) -> uint8_t {
                    if constexpr (kQoQ) {
                        const int q = cute::min(cute::max(__float2int_rn(v), -127), 127);
                        return static_cast<uint8_t>(static_cast<int8_t>(q));
                    } else {
                        const __nv_fp8_e4m3 q(v);
                        return *reinterpret_cast<const uint8_t*>(&q);
                    }
                };
                if constexpr (kSwapABRequested) {
                    auto silu = [](float x) -> float {
                        const float e = kFastMath ? __expf(-x) : expf(-x);
                        const float sig = kFastMath ? math::fast_rcp(1.0f + e) : 1.0f / (1.0f + e);
                        return x * sig;
                    };
                    auto clamp_gate = [](float& x) {
                        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                            x = cute::min(x, kActivationClamp);
                    };
                    auto clamp_up = [](float& x) {
                        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                            x = cute::min(cute::max(x, -kActivationClamp), kActivationClamp);
                    };

                    constexpr uint32_t reduce_warp_start = 0;
                    // Half-tile tasks: only WG0's warps hold the reduced accumulators.
                    constexpr uint32_t reduce_warp_count =
                        kHalfTileTasks ? kNumEpilogueWarps / 2 : kNumEpilogueWarps;
                    const uint32_t scale_token_thread = epilogue_thread_idx;
                    constexpr uint32_t scale_token_stride = kNumEpilogueThreads;
                    const uint32_t sf_base_k_idx =
                        n_block_idx * L1_OUT_BLOCK_N / kL2ActsSFGranK;
                    float swap_v0[kSwapABWeightHalves][kSwapABTokenChunks] = {};
                    float swap_v1[kSwapABWeightHalves][kSwapABTokenChunks] = {};

                    auto store_l1_swap_chunk = [&](const uint32_t& i) {
                        const uint32_t token_0 = i * 8 + col_idx * 2;
                        const uint32_t token_1 = token_0 + 1;

                        float v0_amax = 0.0f;
                        float v1_amax = 0.0f;
                        #pragma unroll
                        for (uint32_t half = 0; half < kSwapABWeightHalves; ++ half) {
                            const uint32_t accum_offset = half * kSwapABHalfAccumPerThread + i * 4;
                            // swapAB: WGMMA rows are weight rows; r_0 = gate, r_1 = up.
                            const float gate_scale = l1_scale_at(half * 64u + r_0);
                            const float up_scale = l1_scale_at(half * 64u + r_1);
                            float v0 = 0.0f;
                            if (token_0 < valid_m) {
                                float g0 = final_accum[accum_offset + 0] * gate_scale;
                                float u0 = final_accum[accum_offset + 2] * up_scale;
                                clamp_gate(g0);
                                clamp_up(u0);
                                const float weight_0 = *l1_topk_weights_buffer
                                    .get_data_buffer(m_idx + token_0)
                                    .template get_base_ptr<float>();
                                v0 = silu(g0) * u0 * weight_0;
                                swap_v0[half][i] = v0;
                                v0_amax = cute::max(v0_amax, cute::abs(v0));
                            }

                            float v1 = 0.0f;
                            if (token_1 < valid_m) {
                                float g1 = final_accum[accum_offset + 1] * gate_scale;
                                float u1 = final_accum[accum_offset + 3] * up_scale;
                                clamp_gate(g1);
                                clamp_up(u1);
                                const float weight_1 = *l1_topk_weights_buffer
                                    .get_data_buffer(m_idx + token_1)
                                    .template get_base_ptr<float>();
                                v1 = silu(g1) * u1 * weight_1;
                                swap_v1[half][i] = v1;
                                v1_amax = cute::max(v1_amax, cute::abs(v1));
                            }
                        }

                        const float amax0 = math::warp_reduce<4, true>(
                            v0_amax, math::ReduceMax<float>());
                        const float amax1 = math::warp_reduce<4, true>(
                            v1_amax, math::ReduceMax<float>());
                        if (row_idx == 0) {
                            if (token_0 < valid_m)
                                smem_cd_l1_shared_sf[token_0 * kNumEpilogueWarps + epilogue_warp_idx] = amax0;
                            if (token_1 < valid_m)
                                smem_cd_l1_shared_sf[token_1 * kNumEpilogueWarps + epilogue_warp_idx] = amax1;
                        }
                    };

                    const uint32_t num_swap_token_chunks = (valid_m + 7u) / 8u;
                    if (is_epilogue_wg) {
                        store_l1_swap_chunk(0);
                        if (valid_m > 8) {
                            #pragma unroll
                            for (uint32_t i = 1; i < kSwapABTokenChunks; ++ i) {
                                if (i < num_swap_token_chunks)
                                    store_l1_swap_chunk(i);
                            }
                        }
                    }

                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

                    for (uint32_t token = scale_token_thread;
                         token < valid_m;
                         token += scale_token_stride) {
                        float amax = 0.0f;
                        #pragma unroll
                        for (uint32_t w = 0; w < reduce_warp_count; ++ w)
                            amax = cute::max(
                                amax, smem_cd_l1_shared_sf[token * kNumEpilogueWarps + reduce_warp_start + w]);
                        float2 amax_pair = {amax, amax};
                        float2 sf_pair, sf_inv_pair;
                        if constexpr (kQoQ) {
                            // INT8 activation for L2: symmetric per-token scale amax/127.
                            sf_pair.x = sf_pair.y = amax * (1.0f / 127.0f);
                            sf_inv_pair.x = sf_inv_pair.y = amax > 0.0f ? 127.0f / amax : 0.0f;
                        } else {
                            math::get_e4m3_sf_and_sf_inv(amax_pair, sf_pair, sf_inv_pair);
                        }

                        auto sf_base_ptr = l2_sf_buffer.get_base_ptr<float>();
                        const uint32_t token_idx = m_idx + token;
                        sf_base_ptr[sf_base_k_idx * kNumPaddedSFPoolTokens + token_idx] =
                            sf_pair.x;
                        smem_cd_l1_shared_sf[token * kNumEpilogueWarps + reduce_warp_start] = sf_inv_pair.x;
                    }

                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

                    #pragma unroll
                    for (uint32_t i = 0; i < kSwapABTokenChunks; ++ i) {
                        if (!is_epilogue_wg) break;
                        const uint32_t token_0 = i * 8 + col_idx * 2;
                        const uint32_t token_1 = token_0 + 1;
                        #pragma unroll
                        for (uint32_t half = 0; half < kSwapABWeightHalves; ++ half) {
                            const uint32_t out_col_base =
                                wg_l1_out_n_idx + half * 32u + warp_idx_in_wg * 8 + row_idx;
                            if (token_0 < valid_m) {
                                const float sf_inv =
                                    smem_cd_l1_shared_sf[token_0 * kNumEpilogueWarps + reduce_warp_start];
                                reinterpret_cast<uint8_t*>(smem_cd_l1)[token_0 * L1_OUT_BLOCK_N + out_col_base] =
                                    quantize_l1_out_byte(swap_v0[half][i] * sf_inv);
                            }
                            if (token_1 < valid_m) {
                                const float sf_inv =
                                    smem_cd_l1_shared_sf[token_1 * kNumEpilogueWarps + reduce_warp_start];
                                reinterpret_cast<uint8_t*>(smem_cd_l1)[token_1 * L1_OUT_BLOCK_N + out_col_base] =
                                    quantize_l1_out_byte(swap_v1[half][i] * sf_inv);
                            }
                        }
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    if (epilogue_wg_idx == 0 and warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            &tensor_map_l1_output,
                            smem_cd_l1,
                            out_n_idx,
                            m_idx);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                    ptx::tma_store_wait<0>();
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    notify_l1_ready(pool_block_idx, n_block_idx);
                } else {
                    // ---------------- L1 EPILOGUE: SwiGLU + FP8 quantize + TMA store ----------------
                    const bool valid_r0 = row_offset_r0 < valid_m;
                    const bool valid_r1 = row_offset_r1 < valid_m;
                    // Layout in `final_accum`:
                    //   16 chunks of 8 N-cols, each chunk = 4 floats per thread = (r0c0, r0c1, r1c0, r1c1).
                    //   Gate chunks: even (0, 2, ..., 14). Up chunks: odd (1, 3, ..., 15).
                    //   Pair `p` ∈ [0, 8): gate chunk = 2p, up chunk = 2p+1.
                    //
                    // For each pair we produce 4 post-SwiGLU floats per thread, mapped to
                    // output cols (p*8 + col_idx*2 + {0,1}) for both r0 and r1.

                    constexpr uint32_t kNumPairs = kAccumPerThread / 8;
                    constexpr uint32_t kNumSFGroups = 1;
                    float swiglu_r0[kNumPairs][2];
                    float swiglu_r1[kNumPairs][2];

                    // Per-row amax, one scale for each 64-col L1 output group.
                    float amax_r0[kNumSFGroups] = {};
                    float amax_r1[kNumSFGroups] = {};

                    // Compute SwiGLU + per-group amax.
                    #pragma unroll
                    for (uint32_t p = 0; p < kNumPairs; ++ p) {
                        const uint32_t gate = 2 * p, up = 2 * p + 1;
                        const uint32_t sf_group = p / 8;

                        auto clamp_gate = [](float& x) {
                            if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                                x = cute::min(x, kActivationClamp);
                        };
                        auto clamp_up = [](float& x) {
                            if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                                x = cute::min(cute::max(x, -kActivationClamp), kActivationClamp);
                        };
                        // Weight rows: gate chunk covers interleaved rows
                        // 16p + col_idx*2 + {0,1}, up chunk the same + 8.
                        const float gs_c0 = l1_scale_at(gate * 8 + col_idx * 2);
                        const float gs_c1 = l1_scale_at(gate * 8 + col_idx * 2 + 1);
                        const float us_c0 = l1_scale_at(up * 8 + col_idx * 2);
                        const float us_c1 = l1_scale_at(up * 8 + col_idx * 2 + 1);
                        float g_r0_c0 = final_accum[gate*4 + 0] * gs_c0; clamp_gate(g_r0_c0);
                        float g_r0_c1 = final_accum[gate*4 + 1] * gs_c1; clamp_gate(g_r0_c1);
                        float g_r1_c0 = final_accum[gate*4 + 2] * gs_c0; clamp_gate(g_r1_c0);
                        float g_r1_c1 = final_accum[gate*4 + 3] * gs_c1; clamp_gate(g_r1_c1);
                        float u_r0_c0 = final_accum[up*4   + 0] * us_c0; clamp_up(u_r0_c0);
                        float u_r0_c1 = final_accum[up*4   + 1] * us_c1; clamp_up(u_r0_c1);
                        float u_r1_c0 = final_accum[up*4   + 2] * us_c0; clamp_up(u_r1_c0);
                        float u_r1_c1 = final_accum[up*4   + 3] * us_c1; clamp_up(u_r1_c1);

                        auto silu = [](float x) -> float {
                            const float e = kFastMath ? __expf(-x) : expf(-x);
                            const float sig = kFastMath ? math::fast_rcp(1.0f + e) : 1.0f / (1.0f + e);
                            return x * sig;
                        };

                        if (valid_r0) {
                            swiglu_r0[p][0] = silu(g_r0_c0) * u_r0_c0;
                            swiglu_r0[p][1] = silu(g_r0_c1) * u_r0_c1;
                            amax_r0[sf_group] = cute::max(
                                amax_r0[sf_group],
                                cute::max(cute::abs(swiglu_r0[p][0]), cute::abs(swiglu_r0[p][1])));
                        } else {
                            swiglu_r0[p][0] = 0.0f;
                            swiglu_r0[p][1] = 0.0f;
                        }
                        if (valid_r1) {
                            swiglu_r1[p][0] = silu(g_r1_c0) * u_r1_c0;
                            swiglu_r1[p][1] = silu(g_r1_c1) * u_r1_c1;
                            amax_r1[sf_group] = cute::max(
                                amax_r1[sf_group],
                                cute::max(cute::abs(swiglu_r1[p][0]), cute::abs(swiglu_r1[p][1])));
                        } else {
                            swiglu_r1[p][0] = 0.0f;
                            swiglu_r1[p][1] = 0.0f;
                        }
                    }


                    const float weight_r0 = valid_r0 ? *l1_topk_weights_buffer
                        .get_data_buffer(m_idx + row_offset_r0)
                        .template get_base_ptr<float>() : 0.0f;
                    const float weight_r1 = valid_r1 ? *l1_topk_weights_buffer
                        .get_data_buffer(m_idx + row_offset_r1)
                        .template get_base_ptr<float>() : 0.0f;
                    #pragma unroll
                    for (uint32_t p = 0; p < kNumPairs; ++ p) {
                        swiglu_r0[p][0] *= weight_r0;
                        swiglu_r0[p][1] *= weight_r0;
                        swiglu_r1[p][0] *= weight_r1;
                        swiglu_r1[p][1] *= weight_r1;
                    }
                    #pragma unroll
                    for (uint32_t g = 0; g < kNumSFGroups; ++ g) {
                        amax_r0[g] *= cute::abs(weight_r0);
                        amax_r1[g] *= cute::abs(weight_r1);
                    }
                    #pragma unroll
                    for (uint32_t g = 0; g < kNumSFGroups; ++ g) {
                        amax_r0[g] = math::warp_reduce<4, false>(amax_r0[g], math::ReduceMax<float>());
                        amax_r1[g] = math::warp_reduce<4, false>(amax_r1[g], math::ReduceMax<float>());
                    }

                    if (col_idx == 0) {
                        smem_cd_l1_shared_sf[epilogue_wg_idx * BLOCK_M + row_offset_r0] = amax_r0[0];
                        smem_cd_l1_shared_sf[epilogue_wg_idx * BLOCK_M + row_offset_r1] = amax_r1[0];
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    if constexpr (kSplitMDecodedWeightReuse) {
                        amax_r0[0] = smem_cd_l1_shared_sf[
                            epilogue_wg_idx * BLOCK_M + row_offset_r0];
                        amax_r1[0] = smem_cd_l1_shared_sf[
                            epilogue_wg_idx * BLOCK_M + row_offset_r1];
                    } else {
                        amax_r0[0] = cute::max(
                            smem_cd_l1_shared_sf[row_offset_r0],
                            smem_cd_l1_shared_sf[BLOCK_M + row_offset_r0]);
                        amax_r1[0] = cute::max(
                            smem_cd_l1_shared_sf[row_offset_r1],
                            smem_cd_l1_shared_sf[BLOCK_M + row_offset_r1]);
                    }

                    float sf_r0[kNumSFGroups], sf_inv_r0[kNumSFGroups];
                    float sf_r1[kNumSFGroups], sf_inv_r1[kNumSFGroups];
                    #pragma unroll
                    for (uint32_t g = 0; g < kNumSFGroups; ++ g) {
                        float2 amax_pair = {amax_r0[g], amax_r1[g]};
                        float2 sf_pair, sf_inv_pair;
                        math::get_e4m3_sf_and_sf_inv(amax_pair, sf_pair, sf_inv_pair);
                        sf_r0[g] = sf_pair.x; sf_inv_r0[g] = sf_inv_pair.x;
                        sf_r1[g] = sf_pair.y; sf_inv_r1[g] = sf_inv_pair.y;
                    }

                    // Quantize and write to smem_cd_l1 (row-major, no swizzle).
                    #pragma unroll
                    for (uint32_t p = 0; p < kNumPairs; ++ p) {
                        const uint32_t sf_group = p / 8;
                        const float v00 = swiglu_r0[p][0] * sf_inv_r0[sf_group];
                        const float v01 = swiglu_r0[p][1] * sf_inv_r0[sf_group];
                        const float v10 = swiglu_r1[p][0] * sf_inv_r1[sf_group];
                        const float v11 = swiglu_r1[p][1] * sf_inv_r1[sf_group];

                        const __nv_fp8x2_e4m3 r0_pair(make_float2(v00, v01));
                        const __nv_fp8x2_e4m3 r1_pair(make_float2(v10, v11));

                        const uint32_t col = p * 8 + col_idx * 2;
                        auto* p0 = reinterpret_cast<uint16_t*>(
                            smem_cd_l1 + row_offset_r0 * L1_OUT_BLOCK_N +
                            wg_l1_out_n_idx + col);
                        auto* p1 = reinterpret_cast<uint16_t*>(
                            smem_cd_l1 + row_offset_r1 * L1_OUT_BLOCK_N +
                            wg_l1_out_n_idx + col);
                        if (valid_r0)
                            *p0 = r0_pair.__x;
                        if (valid_r1)
                            *p1 = r1_pair.__x;
                    }

                    // Write one physical L2-activation scale per 128 output columns.
                    if (col_idx == 0) {
                        auto sf_base_ptr = l2_sf_buffer.get_base_ptr<float>();
                        const uint32_t token_r0 = m_idx + row_offset_r0;
                        const uint32_t token_r1 = m_idx + row_offset_r1;
                        const uint32_t base_k_sf_idx =
                            (n_block_idx * L1_OUT_BLOCK_N + wg_l1_out_n_idx) / kL2ActsSFGranK;
                        #pragma unroll
                        for (uint32_t g = 0; g < kNumSFGroups; ++ g) {
                            const uint32_t sf_k_idx = base_k_sf_idx + g;
                            if ((kSplitMDecodedWeightReuse || epilogue_wg_idx == 0) && valid_r0)
                                sf_base_ptr[sf_k_idx * kNumPaddedSFPoolTokens + token_r0] = sf_r0[g];
                            if ((kSplitMDecodedWeightReuse || epilogue_wg_idx == 0) && valid_r1)
                                sf_base_ptr[sf_k_idx * kNumPaddedSFPoolTokens + token_r1] = sf_r1[g];
                        }
                    }

                    // Issue TMA store of the entire tile. Padding rows beyond
                    // `valid_m` are written with stale/garbage FP8 to the L1-output
                    // pool buffer, but they are never consumed downstream: the L2
                    // GEMM tile loads them, but its NVLink-scatter epilogue is
                    // gated by `m_idx_in_block >= valid_m`, and stale SF in the
                    // padding rows can produce NaN accumulators that simply stay
                    // in registers (only valid rows are converted to BF16 and
                    // STSM'd into smem). Using TMA for partial tiles is a large
                    // win for low-batch / decode where every tile is partial.
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                        const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            &tensor_map_l1_output,
                            smem_cd_l1,
                            out_n_idx,
                            m_idx);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                    ptx::tma_store_wait<0>();
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    notify_l1_ready(pool_block_idx, n_block_idx);
                }
            } else {
                // ---------------- L2 EPILOGUE: BF16 cast + NVLink scatter ----------------
                if constexpr (kSwapABRequested) {
                    // Each active warp scatters a contiguous group of up to 16 rows.
                    constexpr uint32_t kNumRowsPerWarp =
                        BLOCK_M == 8 ? 4u : 8u;
                    auto store_swap_bf16 = [&](const uint32_t& token, const uint32_t& col, const float& value) {
                        if (token < valid_m)
                            smem_cd_l2[token * BLOCK_N + wg_n_idx + col] =
                                __float2bfloat16_rn(value * l2_scale_at(col));
                    };

                    const uint32_t num_swap_token_chunks = (valid_m + 7u) / 8u;
                    auto store_l2_swap_chunk = [&](const uint32_t& i) {
                        const uint32_t token_0 = i * 8 + col_idx * 2;
                        const uint32_t token_1 = token_0 + 1;
                        // L2 half-row tasks: one 64-row half per WG (own accumulators,
                        // own 64 hidden rows [n_idx, n_idx + 64) of the combine row).
                        #pragma unroll
                        for (uint32_t half = 0; half < kWGHalves; ++ half) {
                            const uint32_t accum_offset = half * kSwapABHalfAccumPerThread + i * 4;
                            const uint32_t col_offset = half * 64u;
                            store_swap_bf16(token_0, col_offset + r_0, final_accum[accum_offset + 0]);
                            store_swap_bf16(token_0, col_offset + r_1, final_accum[accum_offset + 2]);
                            store_swap_bf16(token_1, col_offset + r_0, final_accum[accum_offset + 1]);
                            store_swap_bf16(token_1, col_offset + r_1, final_accum[accum_offset + 3]);
                        }
                    };

                    if (is_epilogue_wg) {
                        store_l2_swap_chunk(0);
                        if (valid_m > 8) {
                            #pragma unroll
                            for (uint32_t i = 1; i < kSwapABTokenChunks; ++ i) {
                                if (i < num_swap_token_chunks)
                                    store_l2_swap_chunk(i);
                            }
                        }
                    }

                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    const uint32_t row_in_warp_block = lane_idx / 16;
                    const uint32_t lane_in_row = lane_idx % 16;
                    constexpr uint32_t kColsPerScatterLane = kWGBlockN / 16;
                    DG_STATIC_ASSERT(kWGBlockN % 16 == 0,
                                     "SwapAB L2 scatter expects an even lane partition");
                    DG_STATIC_ASSERT(kColsPerScatterLane == 4 || kColsPerScatterLane == 8,
                                     "SwapAB L2 scatter supports WG_BLOCK_N=64 or 128");

                    #pragma unroll
                    for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                        const uint32_t token = warp_idx_in_wg * 16 + j * 2 + row_in_warp_block;
                        if (token >= valid_m || !is_epilogue_wg) break;

                        const auto src_metadata = *workspace.get_token_src_metadata_ptr(
                            pool_block_idx * BLOCK_M + token);
                        const uint32_t dst_rank_idx = src_metadata.rank_idx;
                        const uint32_t dst_token_idx = src_metadata.token_idx;
                        const uint32_t dst_topk_idx = src_metadata.topk_idx;
                        const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                               .get_data_buffer(dst_token_idx);
                        auto smem_ptr = smem_cd_l2
                            + token * BLOCK_N
                            + wg_n_idx
                            + lane_in_row * kColsPerScatterLane;
                        if constexpr (kColsPerScatterLane == 8) {
                            const auto packed = *reinterpret_cast<uint4*>(smem_ptr);
                            auto dst_ptr = math::advance_ptr<uint4>(
                                dst_token.get_base_ptr(),
                                n_idx * sizeof(nv_bfloat16) + lane_in_row * sizeof(uint4));
                            *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                        } else {
                            const auto packed = *reinterpret_cast<uint2*>(smem_ptr);
                            auto dst_ptr = math::advance_ptr<uint2>(
                                dst_token.get_base_ptr(),
                                n_idx * sizeof(nv_bfloat16) + lane_in_row * sizeof(uint2));
                            *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                        }
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    signal_combine_arrivals();
                } else {
                    DG_STATIC_ASSERT(WG_BLOCK_N == 64 || WG_BLOCK_N == 128,
                                     "Direct L2 scatter requires N64/N128");
                    const bool valid_r0 = row_offset_r0 < valid_m;
                    const bool valid_r1 = row_offset_r1 < valid_m;

                    auto scatter_direct_row = [&](const uint32_t& row_offset, const bool& valid_row, const uint32_t& row_accum_offset) {
                        if (valid_row) {
                            const auto src_metadata = *workspace.get_token_src_metadata_ptr(
                                pool_block_idx * BLOCK_M + row_offset);
                            const uint32_t dst_rank_idx = src_metadata.rank_idx;
                            const uint32_t dst_token_idx = src_metadata.token_idx;
                            const uint32_t dst_topk_idx = src_metadata.topk_idx;
                            const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                                   .get_data_buffer(dst_token_idx);
                            auto dst_base = math::advance_ptr<uint8_t>(
                                dst_token.get_base_ptr(), n_idx * sizeof(nv_bfloat16));
                            auto mapped_dst_base = sym_buffer.map(dst_base, dst_rank_idx);

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 8; ++ i) {
                                const uint32_t chunk_lo = 2 * i, chunk_hi = 2 * i + 1;
                                const uint32_t col_lo = chunk_lo * 8 + col_idx * 2;
                                const uint32_t col_hi = chunk_hi * 8 + col_idx * 2;
                                const uint32_t packed_lo = cast_l2_scaled_bf16_pair(
                                    final_accum[chunk_lo * 4 + row_accum_offset + 0],
                                    final_accum[chunk_lo * 4 + row_accum_offset + 1], col_lo);
                                const uint32_t packed_hi = cast_l2_scaled_bf16_pair(
                                    final_accum[chunk_hi * 4 + row_accum_offset + 0],
                                    final_accum[chunk_hi * 4 + row_accum_offset + 1], col_hi);
                                *reinterpret_cast<uint32_t*>(mapped_dst_base + col_lo * sizeof(nv_bfloat16)) = packed_lo;
                                *reinterpret_cast<uint32_t*>(mapped_dst_base + col_hi * sizeof(nv_bfloat16)) = packed_hi;
                            }
                        }
                    };

                    scatter_direct_row(row_offset_r0, valid_r0, 0);
                    scatter_direct_row(row_offset_r1, valid_r1, 2);
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    signal_combine_arrivals();
                }
            }
        };
        unsigned long long ktask_prev_end = 0ull;
        const auto run_math_task = [&](const auto& block_phase,
                                       const uint32_t& local_expert_idx,
                                       const uint32_t& num_k_blocks,
                                       const uint32_t& m_block_idx, const uint32_t& n_block_idx,
                                       const uint32_t& pool_block_idx,
                                       const uint32_t& valid_m,
                                       const uint32_t& k_split_idx,
                                       const uint32_t& num_k_splits,
                                       const uint32_t& k_block_begin,
                                       const uint32_t& first_worker_idx) {
            using BlockPhaseTag = std::remove_cv_t<std::remove_reference_t<decltype(block_phase)>>;
            constexpr bool kBlockIsL2 = BlockPhaseTag::value == fused_sched::BlockPhase::Linear2;
            if (epilogue_thread_idx == 0) stamp_min(3);
            // Per-task probe (SM0 thread0, SM cycles): 25/27 = L1 task time / count,
            // 26/28 = L2 task time / count, 29 = gap between consecutive tasks.
            // (kL2HalfRowTasks: an L2 task is 5 stages of 128 rows, 64 per WG.)
            const bool ktask_probe_on =
                (phase_stamps != nullptr) && (sm_idx == 0) && (epilogue_thread_idx == 0);
            unsigned long long kt_task0 = 0;
            if (ktask_probe_on) {
                kt_task0 = clock64();
                if (ktask_prev_end != 0ull) atomicAdd(phase_stamps + 29, kt_task0 - ktask_prev_end);
            }
            run_math_task_impl(block_phase, local_expert_idx, num_k_blocks,
                               m_block_idx, n_block_idx, pool_block_idx, valid_m,
                               k_split_idx, num_k_splits, k_block_begin, first_worker_idx);
            if (epilogue_thread_idx == 0) stamp_max(kBlockIsL2 ? 5 : 4);
            if (ktask_probe_on) {
                const unsigned long long kt_task1 = clock64();
                if (valid_m > 0) {
                    atomicAdd(phase_stamps + (kBlockIsL2 ? 26 : 25), kt_task1 - kt_task0);
                    atomicAdd(phase_stamps + (kBlockIsL2 ? 28 : 27), 1ull);
                }
                // 32/33 = K-blocks (2 per stream-K unit) run by SM0 in L1 / L2
                atomicAdd(phase_stamps + (kBlockIsL2 ? 33 : 32), static_cast<unsigned long long>(num_k_blocks));
                ktask_prev_end = kt_task1;
            }
        };
        if constexpr (kTinyMGemv) {
#include <deep_gemm/impls/sm90_fp4_mega_moe_h20_tinym_math.inl>
        } else if constexpr (kUseInterleavedScheduler) {
            for_each_published_block(run_math_task);
        } else {
            for_each_static_selected_block(run_math_task);
        }

        // Fine-grained combine: tell the dispatch warps that this CTA's math tasks
        // are done (replaces the epilogue/dispatch pairing below).
        if constexpr (kFineCombine) {
            if (epilogue_thread_idx == 0) {
                auto* mailbox = workspace.get_combine_mailbox_ptr(sm_idx);
                while (combine_mailbox_seq - ptx::ld_volatile(mailbox + 1) >= fused_layout::kSM90FineCombineRingSize) {}
                mailbox[4 + (combine_mailbox_seq & (fused_layout::kSM90FineCombineRingSize - 1))] =
                    fused_layout::kSM90FineCombineDoneEntry;
                ptx::st_rel_gpu(mailbox, combine_mailbox_seq + 1);
            }
        }

        // ---------------- COMBINE ----------------
        // Barrier path: NVLink barrier first, signals remote ranks that this rank's
        // GEMM outputs (NVLink scatter targets) are fully written. Fine-grained
        // combine (kFineCombine) skips it: readiness is per token (see the loop).
        if constexpr (!kFineCombine) {
            fused_comm::nvlink_barrier<kNumRanks, kNumSMs, kNumEpilogueThreads,
                                 kEpilogueGridSyncIndex, kBeforeCombineReduceBarrierTag>(
                workspace, sym_buffer, sm_idx, epilogue_thread_idx,
                [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); },
                true, true,
                /* fast epilogue: barrier #2 of the launch (#3 = after cleanup, no epilogue) */
                kNvlFastEpilogue, nvl_done_base, 2u);
            if (epilogue_thread_idx == 0) stamp_max(6);
        }

        // Sync with dispatch (paired with dispatch's pre-cleanup sync) so that
        // dispatch may now safely clean workspace state (kFineCombine: the mailbox
        // DONE entry carries that information instead).
        if constexpr (!kFineCombine)
            ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
        constexpr uint32_t kNumElemsPerUint4 = sizeof(uint4) / sizeof(nv_bfloat162);

        constexpr uint32_t kNumChunkSlots = 3;
        constexpr uint32_t kNumMaxRegistersForBuffer = 128;
        constexpr uint32_t kNumChunks =
            (kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes <= SMEM_BEFORE_BARRIER_SIZE
             and kHidden <= 32 * kNumMaxRegistersForBuffer) ? 1 : 2;
        constexpr uint32_t kNumChunkBytes = kNumHiddenBytes / kNumChunks;
        constexpr uint32_t kNumChunkUint4 = kNumChunkBytes / sizeof(uint4);
        constexpr uint32_t kNumUint4PerLane = kNumChunkUint4 / 32;
        DG_STATIC_ASSERT(kHidden % kNumChunks == 0, "Hidden must be divisible by number of chunks");
        DG_STATIC_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes / kNumChunks <= SMEM_BEFORE_BARRIER_SIZE, "Hidden is too large");
        DG_STATIC_ASSERT(kNumChunkBytes % 16 == 0, "Combine chunk must be TMA-aligned (16 bytes)");
        DG_STATIC_ASSERT(kNumChunkBytes % sizeof(uint4) == 0, "Combine chunk must be divisible by 16 bytes");
        DG_STATIC_ASSERT(kNumChunkUint4 % 32 == 0, "Combine chunk must be a multiple of 32 16-byte elements");
        DG_STATIC_ASSERT(kNumTopk <= 32, "Top-k must fit in a single warp");

        const auto combine_load_buffer = utils::PatternVisitor([&](const uint32_t& i) {
            return math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + i * kNumEpilogueWarps) * kNumChunkBytes);
        });
        const auto combine_store_buffer = math::advance_ptr<uint4>(
            smem_buffer, (epilogue_warp_idx + kNumEpilogueWarps * 2) * kNumChunkBytes);

        auto combine_load_barriers = utils::PatternVisitor([&](const uint32_t& i) {
            return combine_barriers[i + epilogue_warp_idx * 2];
        });

        uint32_t combine_phase = 0;
        uint32_t load_stage_idx = 0;
        for (uint32_t token_idx = sm_idx * kNumEpilogueWarps + epilogue_warp_idx;
             token_idx < num_tokens;
             token_idx += kNumSMs * kNumEpilogueWarps) {
            const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                static_cast<int>(__ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + lane_idx)) : -1;
            const uint32_t total_mask = __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0);

            if constexpr (kFineCombine) {
                // Wait until every (topk slot, L2 N-block) slice of this token has
                // landed, then hand the counter back (see the layout header) and
                // order the generic-proxy acquire before the TMA (async-proxy) loads.
                if (lane_idx == 0) {
                    const auto counter_ptr = workspace.get_combine_arrival_count_ptr(token_idx);
                    const int target = static_cast<int>(__popc(total_mask) * kNumRoutedL2BlockNs);
                    while (ptx::ld_acq_sys(counter_ptr) != target) {}
                    *counter_ptr = 0;
                    asm volatile("fence.proxy.async.global;" ::: "memory");
                    stamp_max(6);
                }
                __syncwarp();
            }

            for (uint32_t chunk = 0; chunk < kNumChunks; ++ chunk) {
                const uint32_t chunk_byte_offset = chunk * kNumChunkBytes;

                uint32_t mask = total_mask;
                const auto move_mask_and_load = [&](const uint32_t& i) {
                    if (mask) {
                        const uint32_t slot_idx = __ffs(mask) - 1;
                        mask ^= 1 << slot_idx;
                        if (cute::elect_one_sync()) {
                            const auto src_ptr = math::advance_ptr<uint8_t>(
                                combine_token_buffer.get_rank_buffer(slot_idx)
                                                    .get_data_buffer(token_idx).get_base_ptr(),
                                chunk_byte_offset);
                            ptx::tma_load_1d(combine_load_buffer[i], src_ptr, combine_load_barriers[i], kNumChunkBytes);
                            ptx::mbarrier_arrive_and_set_tx(combine_load_barriers[i], kNumChunkBytes);
                        }
                        __syncwarp();
                        return true;
                    }
                    return false;
                };

                bool do_reduce = move_mask_and_load(load_stage_idx);

                float2 reduced[kNumUint4PerLane * kNumElemsPerUint4] = {};
                while (do_reduce) {
                    do_reduce = move_mask_and_load(load_stage_idx ^ 1);
                    combine_load_barriers[load_stage_idx]->wait(combine_phase);
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                        const auto uint4_values = combine_load_buffer[load_stage_idx][j * 32 + lane_idx];
                        const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&uint4_values);
                        #pragma unroll
                        for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                            ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);
                    }
                    combine_phase ^= load_stage_idx;
                    load_stage_idx ^= 1;
                }

                #pragma unroll
                for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                    uint4 casted;
                    auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                    #pragma unroll
                    for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                        casted_bf16[l] = __float22bfloat162_rn(reduced[j * kNumElemsPerUint4 + l]);

                    if (j == 0) {
                        ptx::tma_store_wait<0>();
                        __syncwarp();
                    }
                    ptx::st_shared(combine_store_buffer + j * 32 + lane_idx,
                                   casted.x, casted.y, casted.z, casted.w);
                }
                __syncwarp();

                if (cute::elect_one_sync()) {
                    cute::tma_store_fence();
                    ptx::tma_store_1d(
                        math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes + chunk_byte_offset),
                        combine_store_buffer, kNumChunkBytes);
                    cute::tma_store_arrive();
                }
                __syncwarp();
            }
        }
        if (epilogue_thread_idx == 0) {
            stamp_max(7);
            if (sm_idx == 0) stamp_accumulate(15);
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_90");
#endif
