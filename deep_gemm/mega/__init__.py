import os
import weakref
import torch
import types
import warnings
from typing import Tuple, Optional, Union
from ..utils.math import align

# noinspection PyBroadException
try:
    # noinspection PyProtectedMember
    import torch.distributed._symmetric_memory as symm_mem
    import torch.distributed as dist
except Exception as exception:
    print(f'Failed to load mega kernels, please check your PyTorch version: {exception}')

from .. import _C


class SymmBuffer:
    def __init__(self, group: dist.ProcessGroup,
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 num_ring_tokens: int,
                 mma_type: str = 'fp8xfp4',
                 activation: str = 'swiglu'):
        assert activation == 'swiglu', f'Only `swiglu` activation is supported, got `{activation}`'
        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.num_ring_tokens = num_ring_tokens

        # Allocate a symmetric buffer
        num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_mega_moe(
            group.size(), num_experts,
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden,
            mma_type, activation,
            num_ring_tokens
        )
        allocator = torch if group.size() == 1 else symm_mem
        self.buffer = allocator.empty(num_bytes, dtype=torch.int8, device='cuda')
        self.handle = (
            types.SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()])
            if group.size() == 1
            else symm_mem.rendezvous(self.buffer, group=group)
        )
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

        # Create input buffer views
        (self.x, self.x_sf,
         self.topk_idx, self.topk_weights,
         self.l1_acts, self.l1_acts_sf,
         self.l2_acts, self.l2_acts_sf) = slice_input_buffers(self.buffer)

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None
        self.x = None
        self.x_sf = None


def get_symm_buffer_for_mega_moe(group: dist.ProcessGroup,
                                 num_experts: int,
                                 num_max_tokens_per_rank: int, num_topk: int,
                                 hidden: int, intermediate_hidden: int,
                                 use_fp8_dispatch: Union[bool, None] = None,
                                 mma_type: str = 'fp8xfp4',
                                 activation: str = 'swiglu') -> SymmBuffer:
    # Align token count
    num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_mega_moe())

    # To save buffer size, we enable ring buffer
    # TODO: move the wave concept into kernel and dynamically schedule
    # TODO: currently decoding may consume more memory than prefill
    # TODO: finer-grained wave
    num_min_ring_tokens, num_max_ring_tokens = \
        _C.get_ring_limit_for_mega_moe(num_max_tokens_per_rank, num_experts // group.size(), num_topk, group.size())
    if num_max_tokens_per_rank >= 6144:
        # We assume must be prefill (decode cannot have such size)
        # We try to give ~8 GB budget (within V4 Pro config)
        # And batch size is mostly stable, to save buffer size, we use 1 expert per wave
        num_ring_tokens = align(768 * 1024, _C.get_token_alignment_for_mega_moe())
    else:
        # Otherwise, we must ensure, like for EP64, 4K decoding batch size,
        # the wave heuristics can select the best number of experts per wave
        # In this case, the budget is roughly ~18 GB
        num_ring_tokens = _C.get_ring_limit_for_mega_moe(
            align(4096, _C.get_token_alignment_for_mega_moe()), 432 // 72, 6, 72)[1]
    num_ring_tokens = max(num_ring_tokens, num_min_ring_tokens)
    num_ring_tokens = min(num_ring_tokens, num_max_ring_tokens)

    # The SM90 MegaMoE family (`fp8xmxfp4`, including the W4A8-int variants)
    # currently runs the legacy arrival-count protocol expressed over the ring
    # counters; it requires the ring to cover the full pool (lap == 0 forever).
    # Use the legacy (pre-rewrite) full-pool size: one kMaxCandidateBlockM (192)
    # padding block per expert instead of the max ring limit's two, which
    # inflates the acts/SF working set by ~10-14% on typical decode shapes.
    # Every real block_m candidate is <= 192, so all pool offsets stay covered
    # and lap == 0 remains valid (also >= the one-expert-per-wave minimum).
    if mma_type == 'fp8xmxfp4':
        num_ring_tokens = _C.get_legacy_pool_tokens_for_mega_moe(
            num_max_tokens_per_rank, num_experts // group.size(), num_topk, group.size())
        num_ring_tokens = max(num_min_ring_tokens, min(num_ring_tokens, num_max_ring_tokens))

    # Backward compat: derive `mma_type` from `use_fp8_dispatch` if provided
    if use_fp8_dispatch is not None:
        assert use_fp8_dispatch == (mma_type.split('x')[0] == 'fp8')
        warnings.warn(
            f'`use_fp8_dispatch` will be deprecated in the future, please use `mma_type`',
            DeprecationWarning, stacklevel=3
        )

    return SymmBuffer(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        num_ring_tokens,
        mma_type=mma_type, activation=activation
    )


def _interleave_weights(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
    # [gate: 0..7, up: 0..7, gate: 8..15, up: 8..15, ...] instead of [gate | up]
    g, n, *rest = t.shape
    half = n // 2
    gate = t[:, :half].reshape(g, half // gran, gran, *rest)
    up = t[:, half:].reshape(g, half // gran, gran, *rest)
    return torch.empty_like(t).copy_(torch.stack([gate, up], dim=2).reshape(g, n, *rest))


def _transpose_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
    num_groups, mn, packed_sf_k = sf.shape
    assert sf.dtype == torch.int and mn % 128 == 0
    result = (sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
                .transpose(2, 3)
                .reshape(num_groups, mn, packed_sf_k))
    return torch.empty_like(sf).copy_(result)


def transform_weights_for_mega_moe(
    l1_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    l2_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    activation: str = 'swiglu'
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
             Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
    assert activation == 'swiglu', f'Only `swiglu` activation is supported, got `{activation}`'
    if isinstance(l1_weights, tuple):
        # FP8: interleave gate/up for weight and SF, then transpose L1 SF for UTCCP
        l1_w = _interleave_weights(l1_weights[0])
        l1_sf = _transpose_sf_for_utccp(_interleave_weights(l1_weights[1]))
        l1_transformed = (l1_w, l1_sf)
        # L2: only transpose SF for UTCCP
        l2_transformed = (l2_weights[0], _transpose_sf_for_utccp(l2_weights[1]))
    else:
        # BF16: L1 interleave gate/up, L2 unchanged
        l1_transformed = _interleave_weights(l1_weights)
        l2_transformed = l2_weights
    return l1_transformed, l2_transformed



def fp8_fp4_mega_moe(y: torch.Tensor,
                     l1_weights: Tuple[torch.Tensor, torch.Tensor],
                     l2_weights: Tuple[torch.Tensor, torch.Tensor],
                     sym_buffer: SymmBuffer,
                     cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                     recipe: Tuple[int, int, int] = (1, 1, 32),
                     activation: str = 'swiglu',
                     activation_clamp: Optional[float] = None,
                     fast_math: bool = True):
    _C.fp8_fp4_mega_moe(
        y,
        l1_weights, l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe,
        activation, activation_clamp,
        fast_math,
        sym_buffer.num_ring_tokens
    )

def bf16_mega_moe(y: torch.Tensor,
                  l1_weights: torch.Tensor,
                  l2_weights: torch.Tensor,
                  sym_buffer: SymmBuffer,
                  cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                  activation: str = 'swiglu',
                  activation_clamp: Optional[float] = None,
                  fast_math: bool = True):
    _C.bf16_mega_moe(
        y,
        l1_weights,
        l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts,
        sym_buffer.num_topk,
        activation, activation_clamp,
        fast_math,
        sym_buffer.num_ring_tokens
    )


# =============================================================================
# SM90 (Hopper) MegaMoE family: MXFP4 / W4A8-int split kernels
# =============================================================================

def _interleave_l1_weights(l1_weights: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    return _interleave_weights(l1_weights[0]), _interleave_weights(l1_weights[1])


def get_nvfp4_mega_moe_sm90_block_n(intermediate_hidden: int) -> int:
    """Choose the measured H20 deployment layout for an SM90 NVFP4 model."""
    assert intermediate_hidden > 0 and intermediate_hidden % 128 == 0
    return 128


def get_nvfp4_mega_moe_sm90_weight_layout(
        hidden: int, intermediate_hidden: int) -> str:
    """Choose the one lossless packed-value layout cached for an SM90 model."""
    assert hidden > 0 and hidden % 128 == 0
    assert intermediate_hidden > 0 and intermediate_hidden % 128 == 0
    return "prmt_groups" if hidden > 2 * intermediate_hidden else "marlin"


def get_mxfp4_mega_moe_sm90_block_n(
        hidden: int, intermediate_hidden: int,
        expected_m_per_rank: Optional[int] = None) -> int:
    """Pick the single deployment BLOCK_N from the workload's typical M.

    Measured H20 crossovers (MXFP4, 8 ranks): BN128 (swapAB+RF small-M path)
    wins below ~M190 on Flash-class shapes (ih<=2048) and ~M380 on Pro-class;
    BN256 wins above (-28% at large M). Defaults to BN128 when no hint given
    (decode-serving is the common case).
    """
    assert hidden % 128 == 0 and intermediate_hidden % 128 == 0
    if expected_m_per_rank is None:
        return 128
    crossover = 192 if intermediate_hidden <= 2048 else 384
    return 256 if expected_m_per_rank >= crossover else 128


def transform_mxfp4_weights_for_mega_moe_sm90(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    block_n: Optional[int] = None,
    block_k: int = 128,
    group_size: int = 32,
    expected_m_per_rank: Optional[int] = None,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Losslessly prepack OCP MXFP4 weights for the SM90 split kernel.

    Inputs contain packed E2M1 values ``(E, N, K/2)`` and row-major E8M0
    scales ``(E, N, K/32)``. Deployment policy: BN128 default, direct PRMT
    value groups (kernel policy is unconditional for MXFP4). The fused rows
    keep the 80-byte stride (64B values + 4B E8M0 + 12B padding).
    """
    from ..quantization_mxfp4 import (
        mxfp4_fuse_packed_with_scale_tile_major,
        mxfp4_scale_to_tile_major,
    )

    l1_packed, l1_scale = l1_weights
    l2_packed, l2_scale = l2_weights
    hidden = l1_packed.size(-1) * 2
    intermediate_hidden = l2_packed.size(-1) * 2
    use_prmt_groups = True
    if block_n is None:
        block_n = get_mxfp4_mega_moe_sm90_block_n(
            hidden, intermediate_hidden, expected_m_per_rank)

    assert block_n in (128, 256)
    assert block_k == 128
    assert group_size == 32
    assert l1_packed.dtype == torch.uint8 and l2_packed.dtype == torch.uint8
    assert l1_scale.dtype == torch.uint8 and l2_scale.dtype == torch.uint8
    assert l1_packed.dim() == 3 and l2_packed.dim() == 3
    assert l1_scale.dim() == 3 and l2_scale.dim() == 3

    l1_packed_il, l1_scale_il = _interleave_l1_weights((l1_packed, l1_scale))
    l1_scale_tm = mxfp4_scale_to_tile_major(
        l1_scale_il, block_n=block_n, block_k=block_k, group_size=group_size)
    l2_scale_tm = mxfp4_scale_to_tile_major(
        l2_scale, block_n=block_n, block_k=block_k, group_size=group_size)
    l1_packed_out = mxfp4_fuse_packed_with_scale_tile_major(
        l1_packed_il.contiguous(), l1_scale_tm, block_k=block_k,
        use_prmt_groups=use_prmt_groups, use_rf_fragments=True)
    l2_packed_out = mxfp4_fuse_packed_with_scale_tile_major(
        l2_packed.contiguous(), l2_scale_tm, block_k=block_k,
        use_prmt_groups=use_prmt_groups, use_rf_fragments=True)
    return (l1_packed_out, l1_scale_tm), (l2_packed_out, l2_scale_tm)


def transform_qoq_int4_weights_for_mega_moe_sm90(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    block_n: int = 128,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Prepack the sole supported INT4 format: QoQ + asymmetric zero point.

    Inputs are ``(packed_uint4, coeff_plane_int32)`` from
    :func:`quantize_to_qoq_int4`. The coefficient bytes are
    ``[s2, z, s1_bf16]`` for every output row and K128 group.
    """
    from ..quantization_mxfp4 import (
        mxfp4_fuse_packed_with_scale_tile_major,
        qoq_plane_to_tile_major,
    )
    assert block_n in (128, 256)
    l1_packed, l1_plane = _interleave_l1_weights(l1_weights)
    l2_packed, l2_plane = l2_weights
    l1_tm = qoq_plane_to_tile_major(l1_plane, block_n)
    l2_tm = qoq_plane_to_tile_major(l2_plane, block_n)
    direct_nibble = os.environ.get("DG_W4A8_INT_DIRECT_NIBBLE", "0") != "0"
    use_prmt_groups = not direct_nibble

    def fuse(packed, plane_tm):
        fused = mxfp4_fuse_packed_with_scale_tile_major(
            packed.contiguous(), plane_tm, block_k=128,
            use_prmt_groups=use_prmt_groups, use_rf_fragments=True)
        return fused, plane_tm

    return fuse(l1_packed, l1_tm), fuse(l2_packed, l2_tm)


def mxfp4_mega_moe(y: torch.Tensor,
                   l1_weights: Tuple[torch.Tensor, torch.Tensor],
                   l2_weights: Tuple[torch.Tensor, torch.Tensor],
                   sym_buffer: SymmBuffer,
                   cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                   l1_global_scales: Optional[torch.Tensor] = None,
                   l2_global_scales: Optional[torch.Tensor] = None,
                   recipe: Optional[Tuple[int, int, int]] = None,
                   activation: str = 'swiglu',
                   activation_clamp: Optional[float] = None,
                   fast_math: bool = True,
                   attn_tp_size: int = 1,
                   attn_tp_group: Optional[dist.ProcessGroup] = None,
                   broadcast_output: bool = False,
                   router_input: Optional[torch.Tensor] = None,
                   router_weight: Optional[torch.Tensor] = None,
                   router_logits: Optional[torch.Tensor] = None,
                   router_renormalize: bool = True,
                   tp_combine_output: Optional[torch.Tensor] = None,
                   hidden_input: Optional[torch.Tensor] = None,
                   _cpp_op=None):
    """Run the SM90 split MegaMoE kernel with prepacked OCP MXFP4 weights.

    Weights come from ``transform_mxfp4_weights_for_mega_moe_sm90``. Values
    decode through an unscaled E2M1->E4M3 LUT and the per-32-K E8M0 dequant
    coefficient is applied in the WGMMA promotion (epilogue) stage.
    """
    l1_scale_metadata = l1_weights[1]
    l2_scale_metadata = l2_weights[1]
    assert l1_scale_metadata.dim() == 5 and l2_scale_metadata.dim() == 5
    cached_block_n = int(l1_scale_metadata.size(3))
    assert int(l2_scale_metadata.size(3)) == cached_block_n
    if recipe is None:
        recipe = (128, cached_block_n, 128)
    else:
        assert recipe[1] == cached_block_n

    assert attn_tp_size >= 1
    assert (router_input is None) == (router_weight is None)
    assert router_logits is None or router_input is None
    if router_input is not None:
        assert router_input.dtype == torch.bfloat16 and router_input.is_contiguous()
        assert router_weight.dtype == torch.bfloat16 and router_weight.is_contiguous()
        assert router_input.shape == (y.size(0), sym_buffer.hidden)
        assert router_weight.shape == (sym_buffer.num_experts, sym_buffer.hidden)
    if router_logits is not None:
        assert router_logits.dtype == torch.bfloat16 and router_logits.is_contiguous()
        assert router_logits.shape == (y.size(0), sym_buffer.num_experts)
    if tp_combine_output is not None:
        assert attn_tp_size > 1 and y.size(0) % attn_tp_size == 0
        assert tp_combine_output.dtype == torch.bfloat16
        assert tp_combine_output.is_contiguous()
        assert tp_combine_output.shape == (y.size(0) // attn_tp_size, y.size(1))
    if hidden_input is not None:
        assert hidden_input.dtype == torch.bfloat16 and hidden_input.is_contiguous()
        assert hidden_input.shape == y.shape
    assert sym_buffer.group.size() % attn_tp_size == 0
    if attn_tp_size > 1:
        assert attn_tp_group is not None, \
            "attn_tp_group is required when attn_tp_size > 1"
        assert attn_tp_group.size() == attn_tp_size
        # Rank layout is contiguous (DP outer, attention TP inner). This is
        # the layout used by SGLang DP-attention and by the SM90 kernel's
        # physical-source mapping.
        assert sym_buffer.group.rank() % attn_tp_size == attn_tp_group.rank()

    if _cpp_op is None:
        _cpp_op = _C.mxfp4_mega_moe
    _cpp_op(
        y,
        l1_weights,
        l2_weights,
        cumulative_local_expert_recv_stats,
        l1_global_scales,
        l2_global_scales,
        router_input,
        router_weight,
        router_logits,
        router_renormalize,
        tp_combine_output,
        hidden_input,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts,
        sym_buffer.num_topk,
        attn_tp_size,
        recipe,
        activation,
        activation_clamp,
        fast_math,
        sym_buffer.num_ring_tokens,
    )

    if tp_combine_output is not None:
        dist.all_gather_into_tensor(y, tp_combine_output, group=attn_tp_group)
    elif attn_tp_size > 1 and broadcast_output:
        # Every expert rank combines a TP group's replicated token stream only
        # onto that group's rank 0. Replicate the completed MoE result back to
        # the remaining attention-TP ranks for the following attention block.
        root_global_rank = dist.get_global_rank(attn_tp_group, 0)
        dist.broadcast(y, src=root_global_rank, group=attn_tp_group)


# W4A8 INT4 has one supported format: canonical QoQ+ZP with full INT4 L1/L2
# and SHIFTXOR decode. DG_W4A8_INT=1 selects it; optional DIRECT_NIBBLE and
# ZSUB_XOR only change the equivalent decode implementation.
int4_mega_moe = mxfp4_mega_moe


def mxfp4_mega_moe_from_bf16(
    y: torch.Tensor,
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    sym_buffer: SymmBuffer,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
    l1_global_scales: Optional[torch.Tensor] = None,
    l2_global_scales: Optional[torch.Tensor] = None,
    recipe: Optional[Tuple[int, int, int]] = None,
    activation: str = 'swiglu',
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
    attn_tp_size: int = 1,
    attn_tp_group: Optional[dist.ProcessGroup] = None,
    broadcast_output: bool = False,
    router_renormalize: bool = True,
    tp_combine_output: Optional[torch.Tensor] = None,
    router_logits_buffer: Optional[torch.Tensor] = None,
):
    """End-to-end BF16-hidden MegaMoE entry point.

    The timed operation starts from rank-local BF16 hidden states and performs
    router GEMM, activation quantization, fused TopK/dispatch, expert L1/L2,
    combine, and the final attention-TP collective. Output is BF16 ``y``.
    """
    assert hidden_states.dtype == torch.bfloat16 and hidden_states.is_contiguous()
    assert router_weight.dtype == torch.bfloat16 and router_weight.is_contiguous()
    if tp_combine_output is None:
        assert hidden_states.shape == y.shape
        local_y = y
    else:
        assert attn_tp_size > 1 and attn_tp_group is not None
        assert tp_combine_output.dtype == torch.bfloat16 and tp_combine_output.is_contiguous()
        assert tp_combine_output.shape == hidden_states.shape
        assert y.shape == (hidden_states.size(0) * attn_tp_size, hidden_states.size(1))
        local_y = tp_combine_output
    assert router_weight.shape == (sym_buffer.num_experts, sym_buffer.hidden)
    assert attn_tp_group is not None or attn_tp_size == 1

    def finalize_tp_output():
        if tp_combine_output is None:
            return
        if broadcast_output:
            # Gather-to-root + broadcast, represented as disjoint token slots
            # followed by SUM reduction. Produces the same global token layout
            # as AllGather while exercising the root-combine communication mode.
            local_m = hidden_states.size(0)
            tp_rank = attn_tp_group.rank()
            y.zero_()
            y[tp_rank * local_m:(tp_rank + 1) * local_m].copy_(local_y)
            root_global_rank = dist.get_global_rank(attn_tp_group, 0)
            dist.reduce(y, dst=root_global_rank, group=attn_tp_group)
            dist.broadcast(y, src=root_global_rank, group=attn_tp_group)
        else:
            dist.all_gather_into_tensor(y, local_y, group=attn_tp_group)

    # Router weights are replicated over attention TP ranks; hidden tokens are
    # rank-local DP shards, so every rank runs the frontend.
    fuse_qoq_input_quant = os.environ.get("DG_W4A8_INT", "0") != "0"
    router_logits = None
    if fuse_qoq_input_quant:
        if router_logits_buffer is None:
            router_logits = torch.empty(
                (hidden_states.size(0), sym_buffer.num_experts),
                dtype=torch.bfloat16, device=hidden_states.device)
        else:
            assert router_logits_buffer.shape == (hidden_states.size(0), sym_buffer.num_experts)
            assert router_logits_buffer.dtype == torch.bfloat16 and router_logits_buffer.is_contiguous()
            router_logits = router_logits_buffer
        if recipe is None:
            recipe = (128, int(l1_weights[1].size(3)), 128)
        _C.qoq_bf16_mega_moe(
            local_y, hidden_states, router_weight, router_logits,
            sym_buffer.x[:hidden_states.size(0)],
            sym_buffer.x_sf[:hidden_states.size(0)],
            sym_buffer.topk_idx[:hidden_states.size(0)],
            sym_buffer.topk_weights[:hidden_states.size(0)],
            l1_weights, l2_weights,
            cumulative_local_expert_recv_stats,
            l1_global_scales, l2_global_scales, None,
            sym_buffer.buffer, sym_buffer.handle.buffer_ptrs,
            sym_buffer.group.rank(), sym_buffer.num_max_tokens_per_rank,
            sym_buffer.num_experts, sym_buffer.num_topk, attn_tp_size,
            recipe, activation, activation_clamp, fast_math,
            sym_buffer.num_ring_tokens)
        finalize_tp_output()
        return y
    else:
        # TP router weights are replicated while tokens are DP-sharded. Every
        # physical rank runs the local fused Router+FP8-quant+TopK frontend and
        # writes directly into the symmetric input views consumed by L1.
        num_tokens = hidden_states.size(0)
        for token_begin in range(0, num_tokens, 64):
            token_end = min(token_begin + 64, num_tokens)
            _C.mxfp4_router_quant_topk(
                hidden_states[token_begin:token_end], router_weight,
                sym_buffer.x[token_begin:token_end],
                sym_buffer.x_sf[token_begin:token_end],
                sym_buffer.topk_idx[token_begin:token_end],
                sym_buffer.topk_weights[token_begin:token_end],
            )

    mxfp4_mega_moe(
        local_y, l1_weights, l2_weights, sym_buffer,
        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
        l1_global_scales=l1_global_scales,
        l2_global_scales=l2_global_scales,
        recipe=recipe,
        activation=activation,
        activation_clamp=activation_clamp,
        fast_math=fast_math,
        attn_tp_size=attn_tp_size,
        attn_tp_group=attn_tp_group,
        broadcast_output=broadcast_output,
        router_logits=None,
        router_renormalize=router_renormalize,
        tp_combine_output=None,
        hidden_input=None,
    )
    finalize_tp_output()
    return y


# Explicit split API names.  Unlike the legacy ``mxfp4_mega_moe`` /
# ``int4_mega_moe`` aliases, these select their backend directly and therefore
# do not depend on mutating DG_W4A8_INT between calls.
def mxfp4_mega_moe_split(*args, **kwargs):
    kwargs["_cpp_op"] = _C.mxfp4_mega_moe_split
    return mxfp4_mega_moe(*args, **kwargs)


def qoq_mega_moe_split(*args, **kwargs):
    kwargs["_cpp_op"] = _C.qoq_mega_moe_split
    return mxfp4_mega_moe(*args, **kwargs)

from .fused import FusedSymmBuffer, get_fused_symm_buffer_for_mega_moe, transform_mxfp4_weights_for_mega_moe_fused, transform_qoq_weights_for_mega_moe_fused, mxfp4_mega_moe_fused, qoq_mega_moe_fused


_FRONTEND_STAMPS_BYTES = 256 * 8 * 8


def fable_frontend_workspace_bytes(e: int) -> int:
    return 256 + 4 * 64 * e * 4 + _FRONTEND_STAMPS_BYTES


def _fe_grid_from_env(grid):
    """DG_FE_TINYM_GRID: unset -> None (follows the MMA: cc -> auto, else 96; see _fe_resolve_knobs);
    '96' -> legacy 24x4 K-split tiny-M grid; 'auto' -> 0 = full-K scheme sized to the SM count;
    N -> full-K scheme with N CTAs in total."""
    if grid is None:
        grid = os.environ.get("DG_FE_TINYM_GRID")
        if grid is None:
            return None
    if isinstance(grid, str):
        grid = 0 if grid.strip().lower() in ("auto", "", "0") else int(grid)
    return int(grid)


def _fe_mma_from_env(mma):
    """DG_FE_TINYM_MMA: 'auto' (default) -> -1 = swapab (+ fragment) on the 96 grid (_fe_resolve_knobs);
    'wmma' -> 0, 'fma' -> 1 (full-K grid only), 'swapab' -> 2 (experts on the MMA M dimension,
    mma.sync m16n8k16, A fragments straight from global; legacy 96 x 4 grid and full-K grid),
    'cc' -> 4 / 'cc6' -> 5 (full-K grid, m <= 2: CUDA-core K-split router, 5 experts x 4 | 6 warps
    per CTA, weights straight into registers)."""
    if mma is None:
        mma = os.environ.get("DG_FE_TINYM_MMA", "auto")
    if isinstance(mma, str):
        mma = {"auto": -1, "wmma": 0, "fma": 1, "swapab": 2, "cc": 4, "cc6": 5, "cc44": 6, "ccfp8": 7,
               "-1": -1, "0": 0, "1": 1, "2": 2, "4": 4, "5": 5, "6": 6, "7": 7}[mma.strip().lower()]
    return int(mma)


def _fe_resolve_knobs(m, grid=None, mma=None, k_parts=None, l2_persist=None):
    """Resolve (grid, mma, k_parts, l2_persist) for m rows. Default scheme (DG_FE_TINYM_MMA=auto) =
    swapab (+ fragment layout) on the legacy 96 grid for every row count: under the customer method
    (nsys kernel span, H20 1830 MHz, 1 row/rank) it measures 7.5-7.7 us vs 8.0-8.7 wmma and
    8.9-9.4 for the CUDA-core router (cc + auto grid + L2 persist), although the cc router's
    kernel-end stamps are the shortest (4.35 / 4.61 us) -- see the README knob table. cc / cc6 /
    cc44 stay opt-in (DG_FE_TINYM_MMA=cc); with them an unset grid follows the MMA (-> auto) and an
    unset DG_FE_ROUTER_L2_PERSIST defaults to 1 (cc + persist 8.9-9.0 vs cc alone 9.2-9.4 us nsys).
    Explicit knobs / env values pass through unchanged."""
    grid = _fe_grid_from_env(grid)
    mma = _fe_mma_from_env(mma)
    k_parts = _fe_kparts_from_env(k_parts)
    if mma == -1:
        mma = 2
    if grid is None:
        grid = 0 if mma in (4, 5, 6, 7) else 96
    if l2_persist is None:
        l2_persist = os.environ.get("DG_FE_ROUTER_L2_PERSIST")
        l2_persist = (1 if mma in (4, 5, 6, 7) else 0) if l2_persist is None else int(l2_persist)
    return grid, mma, k_parts, int(l2_persist)


def _fe_wlayout_from_env(wlayout):
    """DG_FE_ROUTER_WLAYOUT: 'row' -> 0 = [e][h] router weights; 'fragment' (default) -> 1 =
    one-time host permutation into m16n8k16 A-fragment order (swapab only; cached per weight
    tensor); 'pre' -> 2 = the caller already passes the permuted tensor."""
    if wlayout is None:
        wlayout = os.environ.get("DG_FE_ROUTER_WLAYOUT", "fragment")
    if isinstance(wlayout, str):
        wlayout = {"row": 0, "fragment": 1, "pre": 2, "0": 0, "1": 1, "2": 2}[wlayout.strip().lower()]
    return int(wlayout)


def fable_router_weight_fragment_layout(router_weight: torch.Tensor) -> torch.Tensor:
    """Permute [E, K] bf16 router weights (E % 16 == 0, K % 32 == 0) into the swapab
    A-fragment order the frontend reads with DG_FE_ROUTER_WLAYOUT=fragment: same bytes,
    laid out as [E/16 groups][K/32 blocks][half: rows g | g+8][lane = 4 g + t][8 bf16] where
    lane (g, t) holds W[16 G + 8 half + g][32 b + 8 t .. + 7]. One warp load instruction of the
    kernel (16 B per lane) then reads one contiguous 512 B run = 4 full 128 B lines. Returned
    tensor has the same shape [E, K] (contiguous) so it drops into the frontend API unchanged."""
    e, k = router_weight.shape
    assert router_weight.dtype == torch.bfloat16 and e % 16 == 0 and k % 32 == 0
    v = router_weight.contiguous().view(e // 16, 2, 8, k // 32, 4, 8)      # [G][half][g][b][t][8]
    return v.permute(0, 3, 1, 2, 4, 5).contiguous().view(e, k)          # [G][b][half][g][t][8]


_fe_fragment_weight_cache = {}        # id(weight tensor) -> (version, permuted copy); entry purged when the tensor dies


def _fe_router_weight_for_layout(router_weight, wlayout):
    """Cached one-time fragment permutation. Keyed by id() of the source tensor OBJECT with a
    weakref.finalize purge (a freed weight whose address / id is reused by a new tensor -- the
    per-seed weights of the tests -- cannot hit a stale entry; tensor keys themselves are unusable
    because dict key comparison would call Tensor.__eq__) plus the in-place version counter.
    Production callers should permute once at weight-transform time
    (fable_router_weight_fragment_layout) and pass wlayout='pre'."""
    if wlayout == 0:
        return router_weight
    key = id(router_weight)
    hit = _fe_fragment_weight_cache.get(key)
    if hit is None or hit[0] != router_weight._version:
        if hit is None:
            weakref.finalize(router_weight, _fe_fragment_weight_cache.pop, key, None)
        hit = _fe_fragment_weight_cache[key] = (router_weight._version, fable_router_weight_fragment_layout(router_weight))
    return hit[1]


_fe_fp8_weight_cache = {}


def fable_router_weight_fp8(router_weight: torch.Tensor) -> torch.Tensor:
    """EXPERIMENT (DG_FE_TINYM_MMA=ccfp8): router weights [E, K] bf16 -> packed uint8 [E, K + 16]:
    e4m3 row (scale = row amax / 448) followed by the fp32 row scale and 12 pad bytes (16 B row
    alignment). Numeric effect vs bf16 is reported by tests/test_frontend_fe78.py --mma ccfp8."""
    assert router_weight.dtype == torch.bfloat16 and router_weight.dim() == 2
    e, k = router_weight.shape
    w = router_weight.float()
    scale = (w.abs().amax(dim=1) / 448.0).clamp_min(1e-30)
    q = (w / scale[:, None]).to(torch.float8_e4m3fn).view(torch.uint8)
    packed = torch.zeros(e, k + 16, dtype=torch.uint8, device=router_weight.device)
    packed[:, :k] = q
    packed[:, k:k + 4] = scale.view(torch.uint8).view(e, 4)
    return packed.contiguous()


def _fe_router_weight_fp8_cached(router_weight):
    key = id(router_weight)
    hit = _fe_fp8_weight_cache.get(key)
    if hit is None or hit[0] != router_weight._version:
        if hit is None:
            weakref.finalize(router_weight, _fe_fp8_weight_cache.pop, key, None)
        hit = _fe_fp8_weight_cache[key] = (router_weight._version, fable_router_weight_fp8(router_weight))
    return hit[1]


def _fe_kparts_from_env(k_parts):
    """DG_FE_TINYM_KPARTS: 1 (default) | 2 | 4 K-parts per expert group (full-K grid only)."""
    if k_parts is None:
        k_parts = os.environ.get("DG_FE_TINYM_KPARTS", "1")
    return int(k_parts)


def fable_frontend_router_ctas(m: int, e: int, h: int = 3072, topk: int = 8, tinym=None, grid=None, k_parts=None, mma=None) -> int:
    """Router CTA count the frontend launch uses for (m, h, e, topk) under the current knobs."""
    if tinym is None:
        tinym = int(os.environ.get("DG_FE_TINYM", "1"))
    grid, mma, k_parts, _ = _fe_resolve_knobs(m, grid, mma, k_parts)
    return _C.fable_frontend_router_ctas(m, h, e, topk, int(bool(tinym)), grid, k_parts)


def fable_frontend_workspace(sym_buffer, e: int, device) -> torch.Tensor:
    """The Fable frontend workspace of ``sym_buffer`` (zero-initialised once, cached):
    [0, 256) tickets / hand-off counters, then the fp32 K-split partial logits, then the
    optional phase stamps. Shared by the standalone FE launch and the fused-FE kernel."""
    cache = getattr(sym_buffer, "_fable_frontend_cache", None)
    if cache is None:
        cache = sym_buffer._fable_frontend_cache = {}
    workspace = cache.get("workspace")
    required = fable_frontend_workspace_bytes(e)
    if workspace is None or workspace.numel() < required:
        workspace = cache["workspace"] = torch.zeros(required, dtype=torch.uint8, device=device)
    return workspace


def fable_router_quant_topk_frontend(hidden: torch.Tensor, router_weight: torch.Tensor,
                                     sym_buffer, quant: str = "mxfp4", tinym=None, stamps=None,
                                     l2_persist=None, pdl=None, grid=None, mma=None, k_parts=None, wlayout=None):
    """Fable dynamic-M fused Router + Quant + TopK8 + Softmax frontend.

    ``tinym`` (env ``DG_FE_TINYM``, default 1): for m <= 16 use the single-wave
    3-stage configuration (bit-identical outputs, ~4x lower latency on 78-SM H20).
    ``stamps`` (env ``DG_FE_STAMPS``, default 0): record per-CTA %globaltimer phase
    stamps into the workspace; read them back with ``fable_frontend_stamps``.
    ``l2_persist`` (env ``DG_FE_ROUTER_L2_PERSIST``, default 1 with the opt-in cc router, else 0): 1 = pin the router
    weights in L2 with the persisting-L2 set-aside (access policy window launch
    attribute), 2 = PTX ``L2::evict_last`` hint on the router weight loads.
    ``pdl`` (env ``DG_FE_PDL``, default 0): programmatic-dependent-launch trigger for
    the fused Mega that follows: 1 = at CTA start, 2 = after the CTA's last store.
    ``grid`` (env ``DG_FE_TINYM_GRID``, default: follows ``mma`` -- ``auto`` for the cc router,
    ``96`` otherwise): tiny-M CTA scheme. ``auto`` =
    full-K router CTAs sized to the SM count (H20: 77 x 5 experts + 1 merger CTA = 78,
    final logits per CTA, streaming top-8 merge, quant on the router CTAs' idle time);
    ``96`` = legacy 24 expert groups x 4 K-parts + m quant/top-k CTAs; ``N`` = full-K
    scheme with N CTAs in total. Full-K is deterministic but not bit-identical to 96
    (different fp32 accumulation order before the bf16 logit rounding).
    ``mma`` (env ``DG_FE_TINYM_MMA``, default ``auto`` = ``swapab`` + fragment layout on the 96 grid;
    nsys FE span 7.5-7.7 us on H20 at 1830 MHz): ``cc`` = opt-in CUDA-core K-split router on the
    ``auto`` grid with L2-pinned weights (5 experts x 4 warps per CTA, weights straight into
    registers, last-arriving CTA does top-8 + softmax; kernel-end stamps 4.35 / 4.61 us for rows 1 / 2
    but 8.9-9.4 us nsys kernel span under the customer method); ``wmma`` = legacy cp.async ring / TMA row
    pieces into smem + WMMA bf16 m16n16k16 fp32-accumulate; ``fma`` = CUDA-core fp32 FMA
    straight from global memory (16 B ld.global.nc, warp butterfly + 8-warp smem sum);
    ``swapab`` (legacy 96 x 4 grid AND full-K) = experts on the MMA M dimension, tokens on N
    (mma.sync m16n8k16 bf16 -> fp32, rows pad to 8), weight A fragments loaded straight from
    global into registers, activation rows staged once in smem.
    ``wlayout`` (env ``DG_FE_ROUTER_WLAYOUT``, default ``fragment``; swapab on the legacy 96 tiny-M grid
    only -- every other launch reads row-major weights and ignores this knob): ``fragment`` =
    the router weights are permuted ONCE on the host (``fable_router_weight_fragment_layout``,
    cached per weight tensor) into m16n8k16 A-fragment order so every warp load instruction is
    one contiguous 512 B run (4 full lines instead of 8 half lines); identical numerics; ``pre`` =
    the caller already passes the permuted tensor (no per-call work in this wrapper).
    ``k_parts`` (env ``DG_FE_TINYM_KPARTS``, default 1; full-K grid only): 1 | 2 | 4 K-parts per
    expert group; K-part CTAs exchange fp32 partials through the workspace (release/acquire
    flag), the part-0 CTA sums in fixed order (part 0, 1, ..) and emits the keys.
    """
    assert quant in ("mxfp4", "qoq")
    m = hidden.size(0)
    e = router_weight.size(0)
    if tinym is None:
        tinym = int(os.environ.get("DG_FE_TINYM", "1"))
    if stamps is None:
        stamps = int(os.environ.get("DG_FE_STAMPS", "0"))
    if pdl is None:
        pdl = int(os.environ.get("DG_FE_PDL", "0"))
    grid, mma, k_parts, l2_persist = _fe_resolve_knobs(m, grid, mma, k_parts, l2_persist)
    wlayout = _fe_wlayout_from_env(wlayout)
    if mma == 2 and wlayout in (1, 2):
        # The fragment layout is read ONLY by the swapab kernel of the legacy 96 tiny-M grid (m <= 16,
        # top-8, h == 3072, 16-expert groups). Every other launch (tinym=0, m > 16, the full-K grid whose
        # expert groups are not 16 wide, other shapes) reads row-major weights, so the permutation must
        # not be applied there: it silently produced wrong top-8 sets (test_frontend_tinym vs tinym=0).
        fragment_ok = bool(tinym) and m <= 16 and grid == 96 and hidden.size(1) == 3072 and sym_buffer.topk_idx.size(1) == 8
        if fragment_ok:
            if wlayout == 1:
                router_weight = _fe_router_weight_for_layout(router_weight, 1)
            mma = 3      # swapab + fragment weight layout (permuted here and cached, or 'pre' = permuted by the caller)
        elif wlayout == 2:
            raise ValueError("wlayout='pre' (fragment-layout router weight) needs the legacy 96 tiny-M grid: "
                             f"tinym={tinym} m={m} grid={grid} h={hidden.size(1)} topk={sym_buffer.topk_idx.size(1)}")
    elif mma == 7 and router_weight.dtype == torch.bfloat16:
        router_weight = _fe_router_weight_fp8_cached(router_weight)     # ccfp8 experiment: e4m3 + row scale, cached
    workspace = fable_frontend_workspace(sym_buffer, e, hidden.device)
    cache = sym_buffer._fable_frontend_cache
    views = cache.get(m)
    if views is None:
        views = cache[m] = (sym_buffer.x[:m], sym_buffer.x_sf[:m],
                            sym_buffer.topk_idx[:m], sym_buffer.topk_weights[:m])
    _C.fable_router_quant_topk_frontend(
        hidden, router_weight, views[0], views[1], views[2], views[3], workspace,
        0 if quant == "mxfp4" else 1, int(bool(tinym)), int(bool(stamps)), int(l2_persist), int(pdl), grid, mma, k_parts)


def fable_frontend_stamps(sym_buffer, e: int) -> torch.Tensor:
    """[num_ctas, 8] int64 ns %globaltimer stamps of the last DG_FE_STAMPS=1 launch.

    Legacy grid (DG_FE_TINYM_GRID=96): router CTAs (first 96 for E=384, m <= 16): start /
    chunk0 landed / mma done / ticket bumped; quant CTAs (next m): start / quant done /
    ticket seen / top-k done / [tiny: partials loaded / rounds done].
    Full-K grid (auto): router CTAs (``fable_frontend_router_ctas``): start / chunk0 landed /
    mma done / keys written / quant done (CTAs t < m) / all chunks issued; merger CTA (last,
    token 0's warp): start / first CTA seen / last CTA seen / merge done / top-k written.
    """
    workspace = sym_buffer._fable_frontend_cache["workspace"]
    off = 256 + 4 * 64 * e * 4
    return workspace[off:off + _FRONTEND_STAMPS_BYTES].view(torch.int64).view(-1, 8).clone()
