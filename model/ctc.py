"""ScorerInterface implementation for CTC."""

import numpy as np
import torch

from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.ctc_prefix_score import CTCPrefixScoreTH
from espnet.nets.scorer_interface import BatchPartialScorerInterface

class CTCPrefixScorer_two_str(BatchPartialScorerInterface):
    """Decoder interface wrapper for CTCPrefixScore."""

    def __init__(self, ctc_air: torch.nn.Module, ctc_bone: torch.nn.Module, eos: int, weighter=None):
        """Initialize class.

        Args:
            ctc (torch.nn.Module): The CTC implementaiton.
                For example, :class:`espnet.nets.pytorch_backend.ctc.CTC`
            eos (int): The end-of-sequence id.

        """
        self.ctc_air = ctc_air
        self.ctc_bone = ctc_bone
        self.eos = eos
        # self.impl = None

        self.impl_air = None
        self.impl_bone = None

        self.weighter = weighter

    def init_state(self, x: torch.Tensor):
        """Get an initial state for decoding.

        Args:
            x (torch.Tensor): The encoded feature tensor

        Returns: initial state

        """
        # x -> c, t, d
        x = x.transpose(0, 1)
        logp_air = self.ctc_air.log_softmax(x[0].unsqueeze(0)).detach().squeeze(0).cpu().numpy()
        logp_bone = self.ctc_bone.log_softmax(x[1].unsqueeze(0)).detach().squeeze(0).cpu().numpy()

        # logp = 0.2 * logp_air + 0.8 * logp_bone
        # logp = self.ctc.log_softmax(x.unsqueeze(0)).detach().squeeze(0).cpu().numpy()
        # TODO(karita): use CTCPrefixScoreTH
        # self.impl = CTCPrefixScore(logp, 0, self.eos, np)
        self.impl_air = CTCPrefixScore(logp_air, 0, self.eos, np)
        self.impl_bone = CTCPrefixScore(logp_bone, 0, self.eos, np)

        return 0, [self.impl_air.initial_state(), self.impl_bone.initial_state()]

    def select_state(self, state, i, new_id=None):
        """Select state with relative ids in the main beam search.

        Args:
            state: Decoder state for prefix tokens
            i (int): Index to select a state in the main beam search
            new_id (int): New label id to select a state if necessary

        Returns:
            state: pruned state

        """
        # state -> (score, List -> [air, bone])
        if type(state) == tuple:
            if len(state) == 2:  # for CTCPrefixScore
                sc, st = state
                assert type(st) is list and len(st) == 2
                return sc[i], [st[0][i], st[1][i]]
            else:  # for CTCPrefixScoreTH (need new_id > 0)
                r, log_psi, f_min, f_max, scoring_idmap = state
                s = log_psi[i, new_id].expand(log_psi.size(1))
                if scoring_idmap is not None:
                    return r[:, :, i, scoring_idmap[i, new_id]], s, f_min, f_max
                else:
                    return r[:, :, i, new_id], s, f_min, f_max
        return None if state is None else state[i]

    def score_partial(self, y, ids, state, x):
        """Score new token.

        Args:
            y (torch.Tensor): 1D prefix token
            next_tokens (torch.Tensor): torch.int64 next token to score
            state: decoder state for prefix tokens
            x (torch.Tensor): 2D encoder feature that generates ys

        Returns:
            tuple[torch.Tensor, Any]:
                Tuple of a score tensor for y that has a shape `(len(next_tokens),)`
                and next state for ys

        """
        prev_score, state = state
        assert type(state) is list and len(state) == 2
        state_air = state[0]
        state_bone = state[1]

        presub_score_air, new_st_air = self.impl_air(y.cpu(), ids.cpu(), state_air)
        presub_score_bone, new_st_bone = self.impl_bone(y.cpu(), ids.cpu(), state_bone)
        # TODO:
        # weight_air = 1
        # weight_bone = 0
        weight_air = self.weighter.attn[0, 0, 0, 0].detach().cpu().numpy()
        weight_bone = self.weighter.attn[0, 0, 0, 1].detach().cpu().numpy()
        presub_score = 1 * (weight_air * presub_score_air + weight_bone * presub_score_bone)
        tscore = torch.as_tensor(
            presub_score - prev_score, device=x.device, dtype=x.dtype
        )
        return tscore, (presub_score, [new_st_air, new_st_bone])

    def batch_init_state(self, x: torch.Tensor):
        """Get an initial state for decoding.

        Args:
            x (torch.Tensor): The encoded feature tensor

        Returns: initial state

        """
        logp = self.ctc.log_softmax(x.unsqueeze(0))  # assuming batch_size = 1
        xlen = torch.tensor([logp.size(1)])
        self.impl = CTCPrefixScoreTH(logp, xlen, 0, self.eos)
        return None

    def batch_score_partial(self, y, ids, state, x):
        """Score new token.

        Args:
            y (torch.Tensor): 1D prefix token
            ids (torch.Tensor): torch.int64 next token to score
            state: decoder state for prefix tokens
            x (torch.Tensor): 2D encoder feature that generates ys

        Returns:
            tuple[torch.Tensor, Any]:
                Tuple of a score tensor for y that has a shape `(len(next_tokens),)`
                and next state for ys

        """
        batch_state = (
            (
                torch.stack([s[0] for s in state], dim=2),
                torch.stack([s[1] for s in state]),
                state[0][2],
                state[0][3],
            )
            if state[0] is not None
            else None
        )
        return self.impl(y, batch_state, ids)
