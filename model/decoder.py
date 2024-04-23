#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging

from typing import Any
from typing import List
from typing import Tuple

import torch

from espnet.nets.pytorch_backend.nets_utils import rename_state_dict
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.scorer_interface import BatchScorerInterface

# customize models
from model.attention import SparseAttention
from model.decoder_layer import (
    StreamAttentionDecoderLayer,
    NoForwardDecoderLayer,
    ExtractDecoderLayer,
)

def _pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    # https://github.com/espnet/espnet/commit/3d422f6de8d4f03673b89e1caef698745ec749ea#diff-bffb1396f038b317b2b64dd96e6d3563
    rename_state_dict(prefix + "output_norm.", prefix + "after_norm.", state_dict)

class DoubleDecoder(BatchScorerInterface, torch.nn.Module):
    def __init__(
        self,
        odim,
        selfattention_layer_type="selfattn",
        attention_dim=256,
        attention_heads=4,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        self_attention_dropout_rate=0.0,
        src_attention_dropout_rate=0.0,
        input_layer="embed",
        use_output_layer=True,
        pos_enc_class=PositionalEncoding,
        normalize_before=True,
        concat_after=False,
    ):
        """Construct an Decoder object."""
        torch.nn.Module.__init__(self)
        self._register_load_state_dict_pre_hook(_pre_hook)
        self.num_bloks = num_blocks

        if input_layer == "embed":
            self.embed_air = torch.nn.Sequential(
                torch.nn.Embedding(odim, attention_dim),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )

            self.embed_bone = torch.nn.Sequential(
                torch.nn.Embedding(odim, attention_dim),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )

        elif input_layer == "linear":
            self.embed_air = torch.nn.Sequential(
                torch.nn.Linear(odim, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )

            self.embed_bone = torch.nn.Sequential(
                torch.nn.Linear(odim, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )

        elif isinstance(input_layer, torch.nn.Module):
            self.embed_air = torch.nn.Sequential(
                input_layer, pos_enc_class(attention_dim, positional_dropout_rate)
            )

            self.embed_bone = torch.nn.Sequential(
                input_layer, pos_enc_class(attention_dim, positional_dropout_rate)
            )

        else:
            raise NotImplementedError("only `embed` or torch.nn.Module is supported.")
        self.normalize_before = normalize_before
        if selfattention_layer_type == "selfattn":
            logging.info("decoder self-attention layer type = self-attention")

            # normal decoders
            self.decoders_air = repeat(
                num_blocks - 1,
                lambda lnum: ExtractDecoderLayer(
                    attention_dim,
                    # self attn
                    MultiHeadedAttention(
                        attention_heads, attention_dim, self_attention_dropout_rate
                    ),
                    # src attn
                    MultiHeadedAttention(
                        attention_heads, attention_dim, src_attention_dropout_rate
                    ),
                    PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ),
            )

            self.decoders_bone = repeat(
                num_blocks - 1,
                lambda lnum: ExtractDecoderLayer(
                    attention_dim,
                    # self attn
                    MultiHeadedAttention(
                        attention_heads, attention_dim, self_attention_dropout_rate
                    ),
                    # src attn
                    MultiHeadedAttention(
                        attention_heads, attention_dim, src_attention_dropout_rate
                    ),
                    PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ),
            )

            # decoder block without feedforward
            self.nf_decoder_air = repeat(
                1,
                lambda lnum: NoForwardDecoderLayer(
                    attention_dim,
                    # self attn
                    MultiHeadedAttention(
                        attention_heads, attention_dim, self_attention_dropout_rate
                    ),
                    # src attn
                    MultiHeadedAttention(
                        attention_heads, attention_dim, src_attention_dropout_rate
                    ),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ),
            )

            self.nf_decoder_bone = repeat(
                1,
                lambda lnum: NoForwardDecoderLayer(
                    attention_dim,
                    # self attn
                    MultiHeadedAttention(
                        attention_heads, attention_dim, self_attention_dropout_rate
                    ),
                    # src attn
                    MultiHeadedAttention(
                        attention_heads, attention_dim, src_attention_dropout_rate
                    ),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ),
            )

            # the MMT module
            self.str_attn = StreamAttentionDecoderLayer(
                    attention_dim,
                    PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                    SparseAttention(1, attention_dim, None,
                                      use_sparsemax=True,
                                      v_trans=True,
                                      scale=0,
                                      tr_scale=True,
                                      ),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                )
        else:
            raise NotImplementedError

        self.selfattention_layer_type = selfattention_layer_type
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
        if use_output_layer:
            self.output_layer = torch.nn.Linear(attention_dim, odim)
        else:
            self.output_layer = None

        self.pre_att = None

    def forward(self, tgt, tgt_mask, memory_air, memory_bone, memory_mask, num_encs):
        """Forward decoder.

        Args:
            tgt (torch.Tensor): Input token ids, int64 (#batch, maxlen_out) if
                input_layer == "embed". In the other case, input tensor
                (#batch, maxlen_out, odim).
            tgt_mask (torch.Tensor): Input token mask (#batch, maxlen_out).
                dtype=torch.uint8 in PyTorch 1.2- and dtype=torch.bool in PyTorch 1.2+
                (include 1.2).
            memory (torch.Tensor): Encoded memory, float32 (#batch, maxlen_in, feat).
            memory_mask (torch.Tensor): Encoded memory mask (#batch, maxlen_in).
                dtype=torch.uint8 in PyTorch 1.2- and dtype=torch.bool in PyTorch 1.2+
                (include 1.2).

        Returns:
            torch.Tensor: Decoded token score before softmax (#batch, maxlen_out, odim)
                   if use_output_layer is True. In the other case,final block outputs
                   (#batch, maxlen_out, attention_dim).
            torch.Tensor: Score mask before softmax (#batch, maxlen_out).

        """
        # embed -> embed air ?
        x_air = self.embed_air(tgt)
        x_bone = self.embed_bone(tgt)

        # air
        # TODO: extract guide x here
        # extract guide vector ...
        # guide_x_air
        for decoder in self.decoders_air:
            x_air, tgt_mask, memory_air, memory_mask = decoder(
                x_air, tgt_mask, memory_air, memory_mask
            )

        # b, l, d
        guide_air = self.decoders_air[0].guide

        for decoder in self.nf_decoder_air:
            x_air, tgt_mask, memory_air, memory_mask = decoder(
                x_air, tgt_mask, memory_air, memory_mask
            )

        # bone
        # guide_x_bone
        for decoder in self.decoders_bone:
            x_bone, tgt_mask, memory_bone, memory_mask = decoder(
                x_bone, tgt_mask, memory_bone, memory_mask
            )

        guide_bone = self.decoders_bone[0].guide

        for decoder in self.nf_decoder_bone:
            x_bone, tgt_mask, memory_bone, memory_mask = decoder(
                x_bone, tgt_mask, memory_bone, memory_mask
            )

        # TODO: cat context vector
        # (b, l, d) cat (b, l, d) -> (b, c, l, d)
        context_mix = torch.cat([x_air.unsqueeze(1), x_bone.unsqueeze(1)], axis=1)
        # b, c, l, d -> b*c, l, d
        b, c, l, d = context_mix.size()
        context_mix = context_mix.view(b*c, l, d)

        # TODO: mean the guide_x
        # b, t, d
        guide_x = 0.5 * guide_air + 0.5 * guide_bone
        # str attn
        x, num_encs = self.str_attn(
            context_mix, use_str_attn=True, num_encs=num_encs, guide_x=guide_x,
        )

        if self.normalize_before:
            x = self.after_norm(x)
        if self.output_layer is not None:
            x = self.output_layer(x)
        return x, tgt_mask

    def forward_one_step(self, tgt, tgt_mask, memory, num_encs, cache=None):
        """Forward one step.

        Args:
            tgt (torch.Tensor): Input token ids, int64 (#batch, maxlen_out).
            tgt_mask (torch.Tensor): Input token mask (#batch, maxlen_out).
                dtype=torch.uint8 in PyTorch 1.2- and dtype=torch.bool in PyTorch 1.2+
                (include 1.2).
            memory (torch.Tensor): Encoded memory, float32 (#batch, maxlen_in, feat).
            cache (List[torch.Tensor]): List of cached tensors.
                Each tensor shape should be (#batch, maxlen_out - 1, size).

        Returns:
            torch.Tensor: Output tensor (batch, maxlen_out, odim).
            List[torch.Tensor]: List of cache tensors of each decoder layer.

        """
        # TODO: ktop_list(in test, get it adaptively)
        # is_adapt = True
        # import math
        # ktops = []
        # b = num_encs + 1
        # a = 1 / self.num_bloks * math.log(b - 1)
        # for i in range(self.num_bloks):
        #     if is_adapt is True:
        #         ktops.append(None)
        #     else:
        #         ktops.append(int(b - math.exp(a * (i + 1))))

        # memory -> c, t, d
        memory_air = memory[0].unsqueeze(0)
        memory_bone = memory[1].unsqueeze(0)

        x_air = self.embed_air(tgt)
        x_bone = self.embed_bone(tgt)

        if cache is None:
            # (normal decoder) * 2 + str_attn
            cache = [None] * (len(self.decoders_air) * 2 + 1)
        new_cache = []

        # for air
        for c, decoder in zip(cache[:len(self.decoders_air)], self.decoders_air):
            x_air, tgt_mask, memory_air, memory_mask = decoder(
                x_air, tgt_mask, memory_air, None, cache=c
            )
            new_cache.append(x_air)

        # b, l, d
        guide_air = self.decoders_air[0].guide

        for decoder in self.nf_decoder_air:
            x_air, tgt_mask, memory_air, memory_mask = decoder(
                x_air, tgt_mask, memory_air, None, cache=cache[-1]
            )
            # new_cache.append(x_air)

        # for bone
        # cache -> add: len(decoders) + 1
        for c, decoder in zip(cache[len(self.decoders_air): 2 * len(self.decoders_air)], self.decoders_bone):
            x_bone, tgt_mask, memory_bone, memory_mask = decoder(
                x_bone, tgt_mask, memory_bone, None, cache=c
            )
            new_cache.append(x_bone)

        # b, l, d
        guide_bone = self.decoders_bone[0].guide

        for decoder in self.nf_decoder_bone:
            x_bone, tgt_mask, memory_bone, memory_mask = decoder(
                x_bone, tgt_mask, memory_bone, None, cache=cache[-1]
            )
            # new_cache.append(x_bone)

        # cat
        # x_air/x_bone -> b=1, l=1, d -> cat to: b=1, c=2, l=1, d
        context_mix = torch.cat([x_air.unsqueeze(1), x_bone.unsqueeze(1)], axis=1)
        # b=1, c=2, l=1, d -> b*c=2, l=1, d
        b, ch, l, d = context_mix.size()
        # b*c=2, l=1, d
        context_mix = context_mix.view(b * ch, l, d)

        # b=1, l=1, d
        # TODO: pre_att
        if self.pre_att is None:
            guide_x = 0.5 * guide_air + 0.5 * guide_bone
        else:
            guide_x = self.pre_att[0, 0, 0, 0] * guide_air + self.pre_att[0, 0, 0, 1] * guide_bone

        # str attn
        # x -> b=1, l>=1, d
        num_encs = ch
        x, num_encs = self.str_attn(
            context_mix, cache=cache[-1], use_str_attn=True, num_encs=num_encs, guide_x=guide_x,
        )
        new_cache.append(x)

        # self.pre_att = self.str_attn.strattn.attn

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.output_layer is not None:
            # B, D_v
            y = torch.log_softmax(self.output_layer(y), dim=-1)

        return y, new_cache

    # beam search API (see ScorerInterface)
    def score(self, ys, state, x):
        """Score."""
        # x -> c, t, d
        # TODO: transpose
        x = x.transpose(0, 1)
        num_encs = len(x)
        ys_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
        if self.selfattention_layer_type != "selfattn":
            # TODO(karita): implement cache
            logging.warning(
                f"{self.selfattention_layer_type} does not support cached decoding."
            )
            state = None
        logp, state = self.forward_one_step(
            ys.unsqueeze(0), ys_mask, x, num_encs, cache=state
        )

        # get scores (1, 1, 1, C) -> (C,)
        # channel_score = self.decoders[-1].strattn.attn
        # state.append(channel_score.squeeze())

        return logp.squeeze(0), state

    # batch beam search API (see BatchScorerInterface)
    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch (required).

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        # merge states
        n_batch = len(ys)
        n_layers = len(self.decoders)
        if states[0] is None:
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                torch.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]

        # batch decoding
        ys_mask = subsequent_mask(ys.size(-1), device=xs.device).unsqueeze(0)
        logp, states = self.forward_one_step(ys, ys_mask, xs, cache=batch_state)

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
        return logp, state_list
