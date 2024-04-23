import numpy as np
import librosa
import os
import logging
import decimal


def load(file, add_info, fs, dtype=np.int16):
    """
    :param file: file path
    :param fs: sample rate
    :param soundtrack: extract the sound track, eg. 0 for left track, 1 for right track
    :param dtype: np.dtype
    :return: x: target audio -> shape(c, t)
    """
    # TODO: read add_info
    path = file
    if 'speed' in add_info:
        speed = add_info['speed']
        tmp_path = './sp{}-{}'.format(speed, os.path.basename(path))
        os.system('sox {} {} speed {} >/dev/null 2>&1'.format(path, tmp_path, speed))
        path = tmp_path
    else:
        tmp_path = None

    if 'soundtrack' in add_info and add_info['soundtrack'] is not None:
        soundtrack = add_info['soundtrack']
        x = librosa.load(path, sr=fs, mono=False, dtype=np.float32)[0][soundtrack, :]
    else:
        # single channel case
        x = librosa.load(path, sr=fs, mono=True, dtype=np.float32)[0]

    # delete tmp file
    if tmp_path is not None:
        os.system('rm -f {}'.format(tmp_path))

    if dtype is np.int16:
        return (x * 2 ** 15).astype(dtype)
    else:
        raise ValueError('not implement for read type of {}'.format(dtype))


def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))


def rolling_window(a, window, step=1):
    # http://ellisvalentiner.com/post/2017-03-21-np-strides-trick
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step]


def framesig(sig, frame_len, frame_step, dither=1.0, preemph=0.97, remove_dc_offset=True, wintype='povey',
             stride_trick=True, dtype=np.float64):
    """Frame a signal into overlapping frames.

    :param sig: the audio signal to frame.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :param stride_trick: use stride trick to compute the rolling window and window multiplication faster
    :returns: an array of frames. Size is NUMFRAMES by frame_len.
    """
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + ((slen - frame_len) // frame_step)

    # check kaldi/src/feat_old/feature-window.h
    padsignal = sig[:(numframes - 1) * frame_step + frame_len]
    if wintype is 'povey':
        win = np.empty(frame_len)
        for i in range(frame_len):
            win[i] = (0.5 - 0.5 * np.cos(2 * np.pi / (frame_len - 1) * i)) ** 0.85
    else:  # the hamming window
        win = np.hamming(frame_len)

    if stride_trick:
        frames = rolling_window(padsignal, window=frame_len, step=frame_step)
    else:
        indices = np.tile(np.arange(0, frame_len), (numframes, 1)) + np.tile(
            np.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
        indices = np.array(indices, dtype=np.int32)
        frames = padsignal[indices]
        # TODO: defult float64
        win = np.tile(win, (numframes, 1))

    frames = frames.astype(dtype)
    raw_frames = np.zeros(frames.shape)
    for frm in range(frames.shape[0]):
        frames[frm, :] = do_dither(frames[frm, :], dither)  # dither
        if remove_dc_offset:
            frames[frm, :] = do_remove_dc_offset(frames[frm, :])  # remove dc offset
        raw_frames[frm, :] = frames[frm, :]
        frames[frm, :] = do_preemphasis(frames[frm, :], preemph)  # preemphasize

    return frames * win, raw_frames


def deframesig(frames, siglen, frame_len, frame_step, winfunc=lambda x: np.ones((x,))):
    """Does overlap-add procedure to undo the action of framesig.

    :param frames: the array of frames.
    :param siglen: the length of the desired signal, use 0 if unknown. Output will be truncated to siglen samples.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :returns: a 1-D signal.
    """
    frame_len = round_half_up(frame_len)
    frame_step = round_half_up(frame_step)
    numframes = np.shape(frames)[0]
    assert np.shape(frames)[1] == frame_len, '"frames" matrix is wrong size, 2nd dim is not equal to frame_len'

    indices = np.tile(np.arange(0, frame_len), (numframes, 1)) + np.tile(
        np.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
    indices = np.array(indices, dtype=np.int32)
    padlen = (numframes - 1) * frame_step + frame_len

    if siglen <= 0: siglen = padlen

    rec_signal = np.zeros((padlen,))
    window_correction = np.zeros((padlen,))
    win = winfunc(frame_len)

    for i in range(0, numframes):
        window_correction[indices[i, :]] = window_correction[
                                               indices[i, :]] + win + 1e-15  # add a little bit so it is never zero
        rec_signal[indices[i, :]] = rec_signal[indices[i, :]] + frames[i, :]

    rec_signal = rec_signal / window_correction
    return rec_signal[0:siglen]


def magspec(frames, NFFT):
    """Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the magnitude spectrum of the corresponding frame.
    """
    if np.shape(frames)[1] > NFFT:
        logging.warn(
            'frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.',
            np.shape(frames)[1], NFFT)
    complex_spec = np.fft.rfft(frames, NFFT)
    return np.absolute(complex_spec)


def powspec(frames, NFFT):
    """Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the power spectrum of the corresponding frame.
    """
    return np.square(magspec(frames, NFFT))


def logpowspec(frames, NFFT, norm=1):
    """Compute the log power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :param norm: If norm=1, the log power spectrum is normalised so that the max value (across all frames) is 0.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the log power spectrum of the corresponding frame.
    """
    ps = powspec(frames, NFFT);
    ps[ps <= 1e-30] = 1e-30
    lps = 10 * np.log10(ps)
    if norm:
        return lps - np.max(lps)
    else:
        return lps


def do_dither(signal, dither_value=1.0):
    signal += np.random.normal(size=signal.shape) * dither_value
    return signal


def do_remove_dc_offset(signal):
    signal -= np.mean(signal)
    return signal


def do_preemphasis(signal, coeff=0.97):
    """perform preemphasis on the input signal.

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
    :returns: the filtered signal.
    """
    return np.append((1 - coeff) * signal[0], signal[1:] - coeff * signal[:-1])


def hz2mel(hz):
    """ convert linear frequencies in Hertz to Mel frequencies

    inputs
    ---------------------------------
    hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.

    outputs
    ---------------------------------
    mels: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    mels = 1127 * np.log(1 + hz / 700.0)
    return mels


def mel2hz(mel):
    """Convert a value in Mels to Hertz

    inputs
    ---------------------------------
    mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.

    outputs
    ---------------------------------
    hz: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    hz = 700 * (np.exp(mel / 1127.0) - 1)
    return hz


def get_mel_filter(fs=16000, n_fft=512, n_mels=80, f_min=0, f_max=None):
    # TODO: different from librosa
    """ Compute a Mel-filterbank.
    The filters are stored in the rows, the columns correspond to fft bins. The filters are returned as an array of
    size nfilt * (nfft/2 + 1)

    inputs
    -----------------------------------------
    nfilt: the number of filters in the filterbank, default 20.
    nfft: the FFT size. Default is 512.
    samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    lowfreq: lowest band edge of mel filters, default 0 Hz
    highfreq: highest band edge of mel filters, default samplerate/2

    outputs
    -------------------------------------------
    fbank: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    f_max = f_max or fs / 2
    assert f_max <= fs / 2, "f_max is greater than fs/2"

    # compute points evenly spaced in mels
    lowmel = hz2mel(f_min)
    highmel = hz2mel(f_max)

    # check kaldi/src/feat_old/Mel-computations.h
    mel_basis = np.zeros([n_mels, n_fft // 2 + 1])
    mel_freq_delta = (highmel - lowmel) / (n_mels + 1)
    for j in range(0, n_mels):
        leftmel = lowmel + j * mel_freq_delta
        centermel = lowmel + (j + 1) * mel_freq_delta
        rightmel = lowmel + (j + 2) * mel_freq_delta
        for i in range(0, n_fft // 2):
            mel = hz2mel(i * fs / n_fft)
            if mel > leftmel and mel < rightmel:
                if mel < centermel:
                    mel_basis[j, i] = (mel - leftmel) / (centermel - leftmel)
                else:
                    mel_basis[j, i] = (rightmel - mel) / (rightmel - centermel)
    return mel_basis


def stft(signal, n_fft, hop_length, win_length, window="povey", center=False, dtype=np.float32, pad_mode="reflect", dither=0.0, preemph=0.97, remove_dc_offset=False):
    pad_len = win_length - hop_length // 2
    # signal = np.pad(signal, (pad_len, pad_len), 'constant', constant_values=(0, 0))
    frames, raw_frames = framesig(signal, win_length, hop_length, dither, preemph, remove_dc_offset, window)
    Y = np.fft.rfft(frames, n_fft)
    return Y


def istft():
    pass

if __name__ == '__main__':
    import matplotlib
    from matplotlib import pyplot as plt
    mel_filter = get_mel_filter()
    step = 5
    for i, m in enumerate(mel_filter):
        if (i+1) % step == 0 or i == 1:
            plt.plot(m)
    plt.ylim([0, 1])
    plt.xlim([0, 255])
    plt.xlabel("频率(f)")
    plt.ylabel("幅值")
    # plt.show()
    plt.savefig("./mel_filter.pdf")

