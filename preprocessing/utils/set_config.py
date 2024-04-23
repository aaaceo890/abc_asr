import json

SPEED_REGEX = "sp[0,1].[0-9]-"
SPEED_FORMAT = "sp{:.1f}-{}"

fbank_conf = {'fs': 16000, 'n_fft': 512, 'n_mels': 80, 'hop_length': 160, 'win_length': 400, 'window': 'povey',
              'fmin': None, 'fmax': None, 'eps': 1e-10}

bpe_conf = {'bpemode': 'unigram', 'bpetag': 'bpe', 'nbpe': 5000}

cmvn_conf = {'norm_means': True, 'norm_vas': True, 'eps': 1.0e-20}


def set_config(default_conf, conf_file) -> dict:
    if type(default_conf) is str:
        return_conf = globals()['{}_conf'.format(default_conf)]
    else:
        raise ValueError

    if conf_file is not None:
        with open(conf_file, 'r') as fp:
            new_conf = json.load(fp)
        for k, v in new_conf.items():
            if k in return_conf:
                return_conf[k] = v

    return return_conf
