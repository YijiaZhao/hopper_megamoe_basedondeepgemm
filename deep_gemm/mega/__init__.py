import os
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
    direct_int4 = (
        os.environ.get("DG_W4A8_INT", "0") != "0" and
        os.environ.get("DG_W4A8_INT_DIRECT_NIBBLE", "0") != "0"
    )
    use_prmt_groups = not direct_int4
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
                   fast_math: bool = True):
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

    _C.mxfp4_mega_moe(
        y,
        l1_weights,
        l2_weights,
        cumulative_local_expert_recv_stats,
        l1_global_scales,
        l2_global_scales,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts,
        sym_buffer.num_topk,
        recipe,
        activation,
        activation_clamp,
        fast_math,
        sym_buffer.num_ring_tokens,
    )


# W4A8-int entry point: the int4 paths (GPTQ fp32-scale / QServe QoQ / QoQ+zero-point)
# share the mxfp4 kernel infrastructure and are selected via DG_W4A8_INT* env vars.
# This alias gives the int4 deployment its properly-named API.
int4_mega_moe = mxfp4_mega_moe
