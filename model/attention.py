#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math

import numpy
import torch
from torch import nn

from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention

class SparseAttention(nn.Module):
    def __init__(self, n_head, n_feat, ktop, use_sparsemax, v_trans, scale, tr_scale):
        """Construct an MultiHeadedAttention object."""
        super(SparseAttention, self).__init__()
        assert n_feat % n_head == 0
        self.v_trans = bool(v_trans)
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = None
        self.relu = torch.nn.ReLU(inplace=True)
        self.ktop = ktop
        self.use_sparsemax = bool(use_sparsemax)
        if scale > 0:
            if tr_scale is not None:
                self.scale = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
                self.scale.data.fill_(scale)
            else:
                self.scale = torch.FloatTensor([scale])
        elif scale == 0:
            assert self.use_sparsemax is True, 'trainable scale just for sparsemax'
            self.linear_scale = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1), nn.ReLU(inplace=True))
            self.scale = None
        else:
            self.scale = None

    def sparsemax(self, score, scale=1, dim=-1):
        org_shape = score.shape
        sort_score, idxs = torch.sort(score)
        # (b, s)
        sort_score = sort_score.view(-1, score.shape[dim])
        tau_z = score.new_zeros(sort_score.shape[0])

        # matrix
        i = score.shape[dim] - 1
        while i >= 0:
            sum_score = (torch.sum(sort_score[:, i:], dim=-1) - scale) / (score.shape[dim] - i)
            mask = (sort_score[:, i] > sum_score)
            get_z = torch.sum(mask.int())
            if get_z == 0:
                break
            tau_z += (-tau_z + sum_score) * mask
            i = i - 1

        re_shape = list(org_shape)
        re_shape[dim] = 1
        tau_z = tau_z.view(re_shape)
        if torch.is_tensor(scale):
            p = self.relu(score - tau_z) / scale.view(re_shape)
        else:
            p = self.relu(score - tau_z) / scale
        return p

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        if self.v_trans:
            v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        else:
            v = value.unsqueeze(-2)
        if self.scale is not None:
            q = q.transpose(1, 2) / torch.norm(q, dim=-1, keepdim=True) * self.scale # (batch, head, time1, d_k)
            k = k.transpose(1, 2) / torch.norm(k.transpose(1, 2), dim=-1, keepdim=True) * self.scale # (batch, head, time2, d_k)
        else:
            q = q.transpose(1, 2)  # (batch, head, time1, d_k)
            k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = float(
                numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min
            )
            scores = scores.masked_fill(mask, min_value)
            if self.use_sparsemax is True:
                self.attn = self.sparsemax(scores).masked_fill(
                    mask, 0.0
                )  # (batch, head, time1, time2)
            else:
                self.attn = torch.softmax(scores, dim=-1).masked_fill(
                    mask, 0.0
                )  # (batch, head, time1, time2)
        else:
            if self.use_sparsemax is True:
                if getattr(self, 'linear_scale', None) is not None:
                    score_norm = torch.norm(scores[:, 0, 0, :], dim=-1, keepdim=True)
                    enc_info = score_norm.new_zeros(score_norm.size())
                    enc_info = enc_info[:].data.fill_(scores.shape[-1])
                    scale = 1 + self.linear_scale(torch.cat([score_norm, enc_info], dim=-1)).squeeze()
                else:
                    scale = 1
                self.attn = self.sparsemax(scores, scale=scale)
            else:
                self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        x = torch.matmul(self.attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.attn, x

    def forward(self, query, key, value, mask, ktop=None):
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        return self.forward_attention(v, scores, mask)

class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Sparse Attention layer with relative position encoding.

    Paper: https://arxiv.org/abs/1901.02860

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        # linear transformation for positional ecoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x, zero_triu=False):
        """Compute relative positinal encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, size).
            zero_triu (bool): If true, return the lower triangular part of the matrix.

        Returns:
            torch.Tensor: Output tensor.

        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, mask):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            pos_emb (torch.Tensor): Positional embedding tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        # (Q + u) * K
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        # ()
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k
        )  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask)
