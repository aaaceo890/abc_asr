import os
import os.path as path


def validate_audio_text(audio_files, text_files, channel_nums=1, channel_regex=None):
    # validate length
    assert len(audio_files) == len(text_files), \
        'length {} of audio_files and length {} of text_files is not match'.format(len(audio_files),
                                                                                   len(text_files))
    # sort and check key match
    sort_audio_files = dict()
    sort_text_files = dict()
    for k1, k2 in zip(sorted(audio_files), sorted(text_files)):
        # validate key match
        assert k1 == k2, '{} in audio_files and {} in text_fles is not match'.format(k1, k2)
        # validate path exist
        for i, _sound_path in enumerate(audio_files[k1]):
            if not path.exists(_sound_path):
                print('audio file {}[{}]: {} not exist'.format(k1, i, _sound_path))
            assert path.exists(_sound_path), 'audio file {}[{}]: {} not exist'.format(k1, i, _sound_path)
        _text_info = text_files[k2]
        assert type(_text_info) is str, 'text info {}: {} is not in str type'.format(k2, _text_info)

        sort_audio_files[k1] = audio_files[k1]
        sort_text_files[k2] = text_files[k2]

    # sort sound list and get utt nums
    utt_nums = 0
    for basename, audiolist in sort_audio_files.items():
        # validate channel nums
        if channel_nums > 0:
            assert len(audiolist) == channel_nums, 'list len {} is not match of channel_nums {}'.format(
                len(audiolist), channel_nums)
        if channel_regex is not None:
            sort_audio_files[basename] = sorted(audiolist, key=lambda x: '{:02d}'.format(int(x.split(channel_regex)[-1].split('.', 1)[0])))
        else:
            sort_audio_files[basename] = sorted(audiolist)
        utt_nums += 1

    return sort_audio_files, sort_text_files, utt_nums


def validate_feats_files(feats_files):
    # check feat path in recorded file exist
    msg = ""
    utt_nums = len(feats_files)
    flag = True
    for k, v in feats_files.items():
        if not path.exists(v['feat']):
            msg += 'Not exist {}\n'.format(v['feat'])
            flag = False

    return utt_nums, flag, msg
