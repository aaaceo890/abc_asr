import logging
import numpy as np
import librosa


def load_audio(files, fs, soundtrack=None, speed=None, dtype=np.int16):
    """
    :param utt: name of this utterance, can extract speed info e.g. sp0.90-utt123
    :param files: list of channel file e.g. [ch1.wav, ch2.wav, ...],
    :param fs: sample rate
    :param soundtrack: 0 for left track, 1 for right track
    :param dtype: np.dtype
    :return: x: target audio -> shape(c, t)
    """

    # single channel case
    if len(files) == 1:
        x = _load_single_channel(files[0], fs=fs, speed=speed, soundtrack=soundtrack, dtype=dtype)
        # TODO: x = x[np.newaxis, :]
    # multi channel case
    else:
        x, lens = [], []
        for file in files:
            x.append(_load_single_channel(file, fs=fs, speed=speed, soundtrack=soundtrack, dtype=dtype))
        min_len = min(xx.shape[0] for xx in x)
        if not all([l == min_len for l in lens]):
            logging.warning('length is not match for {}, lens: {}, will pad to min_len: {}'.format(file, lens, min_len))
        x = np.stack([x_ch[:min_len] for x_ch in x], axis=0)

    return x


def _load_single_channel(file, fs, soundtrack=None, speed=None, dtype=np.int16):
    if soundtrack is not None:
        x = librosa.load(file, sr=fs, mono=False, dtype=np.float32)[0][soundtrack, :]
    else:
        # single channel case
        x = librosa.load(file, sr=fs, mono=True, dtype=np.float32)[0]

    # TODO: do resample
    if speed is not None:
        x = librosa.resample(x, fs, fs / speed)

    if dtype is np.int16:
        return (x * 2 ** 15).astype(dtype)
    else:
        raise ValueError('Not implement for read type: {}'.format(dtype))


if __name__ == "__main__":
    import matplotlib
    from matplotlib import pyplot as plt
    test_audio_path = "/home/Chenjq/Chenjq_utils/example/1.0_speed.wav"
    test_audio = librosa.load(test_audio_path, sr=16000)[0]

    sox_09 = librosa.load("/home/Chenjq/Chenjq_utils/example/0.9_speed.wav", sr=16000)[0]
    sox_11 = librosa.load("/home/Chenjq/Chenjq_utils/example/1.1_speed.wav", sr=16000)[0]

    speed = 0.9
    test_09sp = librosa.resample(test_audio, 16000, 16000 / speed)
    speed = 1.1
    test_11sp = librosa.resample(test_audio, 16000, int(16000 / speed))

    error1 = (test_09sp - sox_09)
    plt.plot(error1)
    plt.show()

    error2 = (test_11sp - sox_11)
    pass