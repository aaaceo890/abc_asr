import logging
import torch
import math
import chainer
from chainer import reporter
from argparse import Namespace

from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E as E2E
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E as E2ETransformer
from espnet.nets.pytorch_backend.conformer.argument import (
    add_arguments_conformer_common,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.dynamic_conv2d import DynamicConvolution2D
from espnet.nets.pytorch_backend.transformer.dynamic_conv import DynamicConvolution
from espnet.nets.pytorch_backend.ctc import CTC

from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport

# customize models
from model.encoder import Encoder
from model.decoder import DoubleDecoder as Decoder
from model.attention import SparseAttention, RelPositionMultiHeadedAttention
from model.ctc import CTCPrefixScorer_two_str

class Reporter(chainer.Chain):
    """A chainer reporter wrapper."""

    def report(self, loss_ctc, loss_ctc_air, loss_ctc_bone, loss_att, acc, cer_ctc_air, cer, wer, mtl_loss):
        """Report at every step."""
        reporter.report({"loss_ctc": loss_ctc}, self)
        reporter.report({"loss_ctc_air": loss_ctc_air}, self)
        reporter.report({"loss_ctc_bone": loss_ctc_bone}, self)
        reporter.report({"loss_att": loss_att}, self)
        reporter.report({"acc": acc}, self)
        reporter.report({"cer_ctc_air": cer_ctc_air}, self)
        reporter.report({"cer": cer}, self)
        reporter.report({"wer": wer}, self)
        logging.info("mtl loss:" + str(mtl_loss))
        reporter.report({"loss": mtl_loss}, self)


class E2E(E2ETransformer):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        E2ETransformer.add_arguments(parser)
        E2E.add_conformer_arguments(parser)
        return parser

    @staticmethod
    def add_conformer_arguments(parser):
        """Add arguments for conformer model."""
        group = parser.add_argument_group("conformer model specific setting")
        group = add_arguments_conformer_common(group)
        return parser

    @property
    def attention_plot_class(self):
        """Return PlotAttentionReport."""
        return PlotAttentionReport

    def __init__(self, idim, odim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        super().__init__(idim, odim, args, ignore_id)
        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate
        if type(idim) == list:
            idim = idim[0]
        self.subsample_list = [self.subsample, self.subsample]
        self.reporter = Reporter()

        self.encoder = None

        self.encoder_air = Encoder(
            idim=idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
            pos_enc_layer_type=args.transformer_encoder_pos_enc_layer_type,
            selfattention_layer_type=args.transformer_encoder_selfattn_layer_type,
            activation_type=args.transformer_encoder_activation_type,
            macaron_style=args.macaron_style,
            use_cnn_module=args.use_cnn_module,
            cnn_module_kernel=args.cnn_module_kernel,
        )

        self.encoder_bone = Encoder(
            idim=idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
            pos_enc_layer_type=args.transformer_encoder_pos_enc_layer_type,
            selfattention_layer_type=args.transformer_encoder_selfattn_layer_type,
            activation_type=args.transformer_encoder_activation_type,
            macaron_style=args.macaron_style,
            use_cnn_module=args.use_cnn_module,
            cnn_module_kernel=args.cnn_module_kernel,
        )

        self.decoder = Decoder(
            odim=odim,
            selfattention_layer_type=args.transformer_decoder_selfattn_layer_type,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.dunits,
            num_blocks=args.dlayers,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            self_attention_dropout_rate=args.transformer_attn_dropout_rate,
            src_attention_dropout_rate=args.transformer_attn_dropout_rate,
        )

        self.ctc = None

        self.ctc_air = CTC(
            odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True
        )

        self.ctc_bone = CTC(
            odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True
        )

        self.reset_parameters(args)

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder, ctc=CTCPrefixScorer_two_str(self.ctc_air, self.ctc_bone, self.eos, weighter=self.decoder.str_attn.strattn))

    def encode(self, x):
        """Encode acoustic features.

                :param list of ndarray x: source acoustic feature C x (T, D)
                :return: encoder outputs (C,T,D)
                :rtype: torch.Tensor
                """
        # self.eval()
        # # (1, T, D)
        # x = tuple([torch.as_tensor(xs).unsqueeze(0) for xs in x])
        # # (C, T, D)
        # x = torch.cat(x, dim=0)
        # # x = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        #
        # # mat_x -> (C, T, D)
        # # mat_x = torch.as_tensor(x[0]).new_zeros((len(x), ) + x[0].shape[1:])
        # # x = torch.as_tensor(x).unsqueeze(0)
        # # for i in range(len(x)):
        # #     mat_x[i] = torch.as_tensor(x[i])
        # enc_output, _ = self.encoder(x, None)

        self.eval()
        # print("len: {}, shape: ".format(len(x), x[0].shape))
        if type(x) == list or x.ndim == 3:
            num_encs = len(x)
            assert num_encs == 2, "numbers of encoders is not equal to 2"

        # print('multi enc test, enc nums: {}'.format(num_encs))
        # enc_outputs = []
        x_enc_air = torch.as_tensor(x[0]).unsqueeze(0)
        enc_output_air, _ = self.encoder_air(x_enc_air, None)

        x_enc_bone = torch.as_tensor(x[1]).unsqueeze(0)
        enc_output_bone, _ = self.encoder_bone(x_enc_bone, None)

        enc_output = torch.cat([enc_output_air, enc_output_bone], dim=0)
        # c, t, d
        return enc_output.transpose(0, 1)

    def forward(self, xs_pad, ilens, ys_pad):
        """E2E forward.

        :param List xs_pad: a list of batch of padded source sequences C x [(B, Tmax, idim), ...]
        :param List ilens: a list of batch of lengths of source sequences C x (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        import logging

        # xs_pad -> (b, 2, t, d)
        # ilens -> (b，)
        ilens = ilens[:, 0]
        num_encs = xs_pad.shape[1]
        assert num_encs == 2, "encoder numbers not equal to 2"
        # (b*c, t, d)
        xs_air = xs_pad[:, 0, : max(ilens), :]
        xs_bone = xs_pad[:, 1, : max(ilens), :]

        src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)

        # hs_pad & hs_mask -> (b*c,t,d)
        hs_pad_air, _ = self.encoder_air(xs_air, src_mask)
        hs_pad_bone, hs_mask = self.encoder_bone(xs_bone, src_mask)

        ## hs_pad -> (B, T, D)
        self.hs_pad_air = hs_pad_air
        self.hs_pad_bone = hs_pad_bone
        # 2. forward decoder
        if self.decoder is not None:
            ys_in_pad, ys_out_pad = add_sos_eos(
                ys_pad, self.sos, self.eos, self.ignore_id
            )
            ys_mask = target_mask(ys_in_pad, self.ignore_id)
            pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad_air, hs_pad_bone, hs_mask, num_encs)
            self.pred_pad = pred_pad

            # 3. compute attention loss
            loss_att = self.criterion(pred_pad, ys_out_pad)
            self.acc = th_accuracy(
                pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
            )
        else:
            loss_att = None
            self.acc = None

        cer_ctc = None
        cer_ctc_air = None
        if self.mtlalpha == 0.0:
            loss_ctc = None
        else:
            batch_size = xs_pad.size(0)
            hs_len = hs_mask.view(batch_size, -1).sum(1)
            loss_ctc_air = self.ctc_air(hs_pad_air.view(batch_size, -1, self.adim), hs_len, ys_pad)
            loss_ctc_bone = self.ctc_bone(hs_pad_bone.view(batch_size, -1, self.adim), hs_len, ys_pad)
            loss_ctc = 0.5 * loss_ctc_air + 0.5 * loss_ctc_bone
            if not self.training and self.error_calculator is not None:
                ys_hat_air = self.ctc_air.argmax(hs_pad_air.view(batch_size, -1, self.adim)).data
                cer_ctc_air = self.error_calculator(ys_hat_air.cpu(), ys_pad.cpu(), is_ctc=True)
            # for visualization
            if not self.training:
                self.ctc_air.softmax(hs_pad_air)

        # 5. compute cer/wer
        if self.training or self.error_calculator is None or self.decoder is None:
            cer, wer = None, None
        else:
            ys_hat = pred_pad.argmax(dim=-1)
            cer, wer = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        # copied from e2e_asr
        alpha = self.mtlalpha
        if alpha == 0:
            self.loss = loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = None
            loss_ctc_air_data = None
            loss_ctc_bone_data = None
        elif alpha == 1:
            self.loss = loss_ctc
            loss_att_data = None
            loss_ctc_data = float(loss_ctc)
        else:
            self.loss = alpha * loss_ctc + (1 - alpha) * loss_att
            loss_att_data = float(loss_att)

            loss_ctc_air_data = float(loss_ctc_air)
            loss_ctc_bone_data = float(loss_ctc_bone)
            loss_ctc_data = float(loss_ctc)

        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(
                loss_ctc_data, loss_ctc_air_data, loss_ctc_bone_data, loss_att_data, self.acc, cer_ctc_air, cer, wer, loss_data
            )
        else:
            logging.warning("loss (=%f) is not correct", loss_data)

        return self.loss

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
        """E2E attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: attention weights (B, H, Lmax, Tmax)
        :rtype: float ndarray
        """
        # TODO: should convert tensor to list
        # attn -> (B*C, H, Lmax, Tmax) -> C *
        self.eval()
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad)
        ret = dict()
        for name, m in self.named_modules():
            if (
                isinstance(m, MultiHeadedAttention)
                or isinstance(m, DynamicConvolution)
                or isinstance(m, RelPositionMultiHeadedAttention)
            ):
                ret[name] = m.attn.cpu().numpy()
            if isinstance(m, SparseAttention):
                # (B*T, H, 1, C)
                if m.attn is not None:
                    ret[name] = m.attn.cpu().numpy()
            if isinstance(m, DynamicConvolution2D):
                ret[name + "_time"] = m.attn_t.cpu().numpy()
                ret[name + "_freq"] = m.attn_f.cpu().numpy()
        self.train()
        return ret
