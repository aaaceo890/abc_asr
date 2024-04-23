import librosa
import numpy as np
import logging
from feat import feats_utils

# copy frome espnet and modify some detail
# base on librosa


def stft(x, n_fft, hop_length, win_length, window="hann", center=False, dtype=np.float32, pad_mode="reflect"):
    """
    :param y: input time domain signal (channel, time, freqency)
    :param n_fft: point nums of fft
    :param hop_length: frame hop length
    :param win_length: frame window length
    :param window: window type, default "hann"
    :param center: If True, let target frame in center
    :param dtype: default float32
    :param pad_mode: padding type, default "reflect"
    :return: y : output stft (channel, time, frequency)
    """

    # single channel case
    if x.ndim == 1:
        channel_nums = 1
        x = x[np.newaxis, :]
    # multi channel case
    else:
        channel_nums = x.shape[0]

    x = x.astype(dtype)

    # stack multi channel
    # x -> (c, t, d)
    # t = (x_len - n_fft) // hop + 1
    # d = n_fft / 2 + 1
    y = np.stack(
        [
            feats_utils.stft(
                x[ch, :],
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=center,
                pad_mode=pad_mode,
            )
            for ch in range(channel_nums)
        ],
        axis=0,
    )

    if channel_nums == 1:
        # y -> (t, d)
        y = y[0, :]
    return y


def istft(y, hop_length, win_length, window="hann", center=True, dtype=None, length=None):
    """
    :param stft: input time domain signal (channel, time, frequency)
    :param hop_length: frame hop length
    :param win_length: frame window length
    :param window: window type, default "hann"
    :param center: If True, let target frame in center
    :param dtype: default float32
    :param length: target reconstruct length
    :return:
    """

    # single channel case
    if y.ndim == 2:
        channel_nums = 1
        y = y[np.newaxis, :]
    # multi channel case
    else:
        channel_nums = y.shape[0]

    # stack multi channel
    # x -> (c,t)
    x = np.stack(
        [
            librosa.istft(
                y[ch, :].T,  # (f, t)
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=center,
                dtype=dtype,
                length=length,
            )
            for ch in range(channel_nums)
        ],
        axis=0,
    )

    if channel_nums == 1:
        # x -> (t,)
        x = x[0, :]
    return x


def stft2logmelspectrogram(y, fs, n_fft, n_mels, power=2, fmin=None, fmax=None, norm="slaney", dtype=np.float32,
                           eps=1e-10):
    """
    :param y: input stft shape -> (c, t, f) or (t, f)
    :param fs: input sample rate
    :param n_fft: num of point fft point
    :param n_mels: num of mel bands
    :param power: 1 for energy, 2 for power, etc.
    :param fmin: lowest frequency
    :param fmax: highest frequency
    :param norm: {None, 'slaney', or number}, default = "slaney"
    :param dtype: np.dtype
    :param eps:
    :return: fbank: shape -> (t, n_mels)
    """
    # y : (c, t, f) or (t, f)
    if fmin is None:
        fmin = 0
    if fmax is None:
        fmax = fs / 2

    # get spectrogram
    # spec -> (c, t, f) or (t, f)
    spec = np.abs(y) ** power
    # get mel filters
    # mel_basis -> (n_mels, f)
    # TODO: no norm?
    mel_basis = feats_utils.get_mel_filter(fs, n_fft, n_mels, fmin, fmax)
    # get mel_spec
    # mel_spec -> (*, t, f) * (f, n_mels) -> (c, t, n_mels) or (t, n_mels)
    mel_spec = np.dot(spec, mel_basis.T)
    # get fbank
    # fbank -> (t, n_mels)
    fbank = np.log(np.maximum(eps, mel_spec))
    return fbank.astype(dtype)


def spectrogram(x, n_fft, hop_length, win_length, window="hann", power=2):
    # will have some default setting
    # stft: center=True, dtype=np.float32, pad_mode="reflect"
    # x -> (c, t)
    # y -> (c, t, f)
    y = stft(x, n_fft, hop_length, win_length, window=window)
    # spec -> (c, t, f)
    spec = np.abs(y) ** power
    return spec


def logmelspectrogram(x, fs, n_fft, n_mels, hop_length, win_length, window="povey", fmin=None, fmax=None, eps=1e-10):
    # will have some default setting
    # stft: center=True, dtype=np.float32, pad_mode="reflect"
    # stft2logmelspectrogram: norm="slaney" power=1
    # x -> (c, t)
    # y -> (c, t, f)
    y = stft(x, n_fft, hop_length, win_length, window=window)
    # fbank -> (c, t, n_mels)
    return stft2logmelspectrogram(y, fs, n_fft, n_mels, fmin=fmin, fmax=fmax, eps=eps)


def logmel2mfcc():
    # M = scipy.fftpack.dct(S, axis=0, type=dct_type, norm=norm)[:n_mfcc]
    pass


def mfcc():
    # logmelspectrogram
    # logmel2mfcc
    pass


def compute_cmvn(files, utt2spk=None):
    # feat_old file -> fname{feat_old.npy: path, shape: np.shape}
    for file in files:
        feat = np.load(file['feat.npy'])
        shape = np.load(file['shape'])
        if utt2spk is not None:
            pass
        else:
            # feat_sums -> (t,f).mean1 + (t,f).mean2 + ...
            # feat_square_sums -> (t,f)^2.mean1 + (t,f)^2.mean2
            feat_sums, feat_square_sums = None, None
        # single channel case
        if shape.ndim == 2:
            # fbank -> (t, f)
            # compute feat_old sum
            feat_sums = feat.mean(axis=0) if feat_sums is None else feat_sums + feat.mean(axis=0)
            # compute feat_old square sum
            feat_square_sums = np.square(feat).mean(axis=0) if feat_square_sums is None \
                else feat_square_sums + np.square(feat).mean(axis=0)
            # save kaldi format cmvn file
            # shape -> (2, n_feat + 1)
            # [0, :-1] -> sum of feat_old ; [0, :-1] -> count
            # [1, :-1] -> sum of feat_old square ; [1, :-1] -> count
            cmvn = np.zeros(2, feat_sums.shape + 1)
            cmvn[0, :] = np.append(feat_sums, len(feat_sums))
            cmvn[1, :] = np.append(feat_square_sums, len(feat_square_sums))
            return cmvn
        # multi channel case
        else:
            pass


def utterence_cmvn(x, norm_means=True, norm_vas=True, eps=1.0e-20):
    # x -> (c, t, f) or (t, f)
    # single channel case
    if x.ndim == 2:
        square_sums = (x ** 2).mean(axis=0)
        mean = x.mean(axis=0)
        if norm_means:
            x = np.substract(x, mean)
        if norm_vas:
            # E^2[x] = E[x^2] - (E[x])^2
            var = square_sums / x.shape[0] - mean ** 2
            std = np.maximum(np.sqrt(var), eps)
            x = np.divide(x, std)

        return x
    # multi channel case
    elif x.ndim == 3:
        pass


def global_cmvn(x, cmvn, norm_means=True, norm_vas=True, eps=1.0e-20):
    # cmvn -> (2, feat_dim + 1)
    # x -> (c, t, f) or (t, f)
    # single channel case
    if x.ndim == 2:
        cmvn = np.load(cmvn)
        feat_sums = cmvn[0, :-1]
        feat_square_sums = cmvn[1, :-1]
        counts = cmvn[0, -1]
        mean = feat_sums / counts
        if norm_means:
            x = np.substract(x, mean)

        if norm_vas:
            var = feat_square_sums / counts - mean ** 2
            std = np.maximum(np.sqrt(var), eps)
            x = np.divide(x, std)
        return x

    # multi channel case
    elif x.ndim == 3:
        pass



if __name__ == '__main__':
    pass
