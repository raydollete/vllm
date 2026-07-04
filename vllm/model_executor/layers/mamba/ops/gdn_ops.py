# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ops for GDN linear attention.

``rearrange_mixed_qkv`` dispatches between a ``torch.compile``d kernel and an
eager fallback: the compiled function cannot run while a CUDA graph is being
captured, so capture-time calls take the eager path (which is then baked into
the captured graph) and every other call gets the compiled single-kernel copy.
"""

import torch


def _rearrange_mixed_qkv(
    mixed_qkv: torch.Tensor,
    key_dim: int,
    value_dim: int,
    tp_size: int,
    head_k_dim: int,
    head_v_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split packed qkv into contiguous (1, seq, heads, dim) tensors.

    Flattens all three splits into a single buffer via ``torch.cat`` so that
    one copy kernel materializes the contiguous q/k/v instead of three
    separate ``contiguous()`` calls.
    """
    seq_len = mixed_qkv.shape[0]
    q_dim = key_dim // tp_size
    k_dim = key_dim // tp_size
    v_dim = value_dim // tp_size

    query, key, value = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)

    fused = torch.cat([query.reshape(-1), key.reshape(-1), value.reshape(-1)], dim=0)

    q_size = seq_len * q_dim
    k_size = seq_len * k_dim

    q_contig = fused[0:q_size]
    k_contig = fused[q_size : q_size + k_size]
    v_contig = fused[q_size + k_size :]

    query = q_contig.view(1, seq_len, -1, head_k_dim)
    key = k_contig.view(1, seq_len, -1, head_k_dim)
    value = v_contig.view(1, seq_len, -1, head_v_dim)

    return query, key, value


_compiled_rearrange_mixed_qkv = torch.compile(_rearrange_mixed_qkv, fullgraph=True)


def rearrange_mixed_qkv(
    mixed_qkv: torch.Tensor | None,
    key_dim: int,
    value_dim: int,
    tp_size: int,
    head_k_dim: int,
    head_v_dim: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if mixed_qkv is None:
        return None, None, None

    if mixed_qkv.is_cuda and torch.cuda.is_current_stream_capturing():
        # torch.compile cannot run under CUDA graph capture; the eager kernel
        # gets captured instead.
        return _rearrange_mixed_qkv(
            mixed_qkv,
            key_dim,
            value_dim,
            tp_size,
            head_k_dim,
            head_v_dim,
        )

    return _compiled_rearrange_mixed_qkv(
        mixed_qkv,
        key_dim,
        value_dim,
        tp_size,
        head_k_dim,
        head_v_dim,
    )
