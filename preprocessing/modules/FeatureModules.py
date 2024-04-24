import os
import os.path as path
import json
import numpy as np

from feat import spectrom
from utils.io_utils import load_audio
from utils.decorators import multi_processing
from utils.validate import validate_feats_files

from utils.set_config import SPEED_FORMAT, SPEED_REGEX


def make_fbank(key, data, feat_dir, soundtrack=None, speed=None, **conf):
    """
    input Data path and Targe path
    Load Data
    Make fbank and save
    Return recorded path file
    :param key: utterance name
    :param data: path list
    :param feat_dir: the directory store feats(np.array)
    :param soundtrack: the soundtrack of audio e.g. soundtrack=0 will extract the left track of audio
    :param speed: perturb speed
    :param speed_regex: add a mark denote the speed
    :param conf: configuration of fbank making
    :return: feat_scp: a dict store utterance name and path of feat
    """
    # TODO: speed perturb here
    if speed is not None:
        key = SPEED_FORMAT.format(speed, key)
    # audio -> (len, ) or (c, len)
    audio = load_audio(data, conf['fs'], soundtrack, speed)
    # fbank -> (c, t, n_mels)
    fbank = spectrom.logmelspectrogram(audio, **conf)
    # save
    _feat_save_file = path.join(feat_dir, key + '_raw_feat.npy')
    np.save(_feat_save_file, fbank)
    feat_scp = {key: {'feat': _feat_save_file, 'shape': fbank.shape}}

    # return -> {key: path}
    return feat_scp


# @multi_processing(n_jobs=16, gather="sum")
def compute_cmvn(utt, feat, utt2spk=None):
    # feat file -> fname{feat.npy: path, shape: np.shape}
    # feat = np.load(data['feat.npy'])
    # shape = np.load(data['shape'])
    feat = np.load(feat["feat"])
    # shape = np.load(shape)
    # TODO: utt2spk?
    if utt2spk is not None:
        pass
    else:
        pass
    # single channel case
    if feat.ndim == 2:
        # fbank -> (t, f)
        # compute feat sum -> (f,)
        counts = feat.shape[0]
        feat_sums = feat.sum(axis=0)
        # compute feat square sum -> (f,)
        feat_square_sums = np.square(feat).sum(axis=0)
        # save kaldi format cmvn file
        # shape -> (2, n_feat + 1)
        # [0, :-1] -> sum of feat ; [0, :-1] -> count
        # [1, :-1] -> sum of feat square ; [1, :-1] -> count
        cmvn = np.zeros((2, feat_sums.shape[0] + 1), dtype=feat_sums.dtype)
        # TODO: wrong : len(feat_sum)=80 -> frame length
        cmvn[0, :] = np.append(feat_sums, counts)
        cmvn[1, :-1] = feat_square_sums
        return cmvn
    # multi channel case
    elif feat.ndim == 3:
        # TODO: check multi channel cmvn:
        #  in low frequency, max (std / mean) -> 1.5%,
        #  in high frequency, max (std / mean) -> 0.07%,
        #  so we just sum all channel together, ignore this small fluctuations
        # fbank -> (c, t, f)
        counts = feat.shape[0] * feat.shape[1]
        # feat -> (c*t, f)
        feat = feat.reshape(counts, -1)
        # compute feat sum -> (c*t, f) -> (f,)
        feat_sums = feat.sum(axis=0)
        # compute feat square sum -> (c*t, f) -> (f,)
        feat_square_sums = np.square(feat).sum(axis=0)
        cmvn = np.zeros((2, feat_sums.shape[0] + 1), dtype=feat_sums.dtype)
        cmvn[0, :] = np.append(feat_sums, counts)
        cmvn[1, :-1] = feat_square_sums
        return cmvn


# @multi_processing(n_jobs=16, gather="list")
def apply_utterance_cmvn(x, norm_means=True, norm_vas=True, eps=1.0e-20):
    x = np.load(x)
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


# @multi_processing(n_jobs=16, gather="dict")
def apply_global_cmvn(key, data, feat_dir, cmvn, norm_means=True, norm_vas=True, eps=1.0e-20):
    # cmvn -> (2, feat_dim + 1)
    # x -> (c, t, f) or (t, f)
    data = np.load(data)
    feat_file = path.join(feat_dir, key + '_feat.npy')
    feat_sums = cmvn[0, :-1]
    feat_square_sums = cmvn[1, :-1]
    counts = cmvn[0, -1]
    mean = feat_sums / counts
    if norm_means:
        data = np.subtract(data, mean)

    if norm_vas:
        var = feat_square_sums / counts - mean ** 2
        std = np.maximum(np.sqrt(var), eps)
        data = np.divide(data, std)
    np.save(feat_file, data)
    feat_scp = {key: {'feat': feat_file, 'shape': data.shape}}
    return feat_scp


class FeatureMaker(object):
    def get_feature(self, set_type):
        feat_files = dict()
        speeds = self.DT.speeds if set_type == "train" else [None]
        for speed in speeds:
            if speed is not None:
                self.logger.info("Speed: {}".format(speed))
            with open(self.DT.path[set_type]["audio_json"], "r") as fp:
                feat_files.update(
                    self.maker(
                        json.load(fp),
                        feat_dir=self.DT.path[set_type]["raw"],
                        soundtrack=self.DT.soundtrack,
                        speed=speed,
                        **self.conf,
                    )
                )

        # validate
        _utt, flag, msg = validate_feats_files(feat_files)
        assert flag is True, msg
        self.DT.prev_utt[set_type], self.DT.latest_utt[set_type] = self.DT.latest_utt[set_type], _utt
        self.logger.info('Successfully validate raw feats for set {}, number of utterance: {} -> {}, speeds x {}'.format(
            set_type, self.DT.prev_utt[set_type], self.DT.latest_utt[set_type], len(speeds)
        ))

        return {"raw_json": feat_files}

    def __init__(self, n_jobs, data_structure, logger, feature_type, **feature_conf):
        self.DT = data_structure
        self.feature_type = feature_type
        self.conf = feature_conf
        self.logger = logger

        if feature_type == "fbank":
            self.maker = multi_processing(func=make_fbank, n_jobs=n_jobs, gather="dict")
        elif feature_type == "mfcc":
            raise NotImplementedError
        elif feature_type == "fbank-pitch":
            raise NotImplementedError
        else:
            raise NotImplementedError

    def __call__(self):
        for set_type in self.DT.sets:
            if self.DT.check_resume(set_type, *["raw_json"]):
                self.logger.info(
                    "File -> raw-feat files exist in {} set, skip".format(set_type))
            else:
                self.logger.info(
                    "File -> raw-feat files files not exist in {} set, generating...".format(set_type))
                self.DT.save(self.get_feature(set_type=set_type), set_type)
                self.logger.info(
                    "File -> raw-feat files in {} set successful saving ".format(set_type))


class Remover(object):
    def remove_data(self, set_type):
        # feat file -> fname{feat.npy: path, shape: np.shape}
        # feat = np.load(data['feat.npy'])
        # shape = np.load(data['shape'])
        with open(self.DT.path[set_type]["raw_json"], "r") as fp:
            feat_files = json.load(fp)

        _feat_files_org = feat_files.copy()
        _utt = 0
        for k, v in _feat_files_org.items():
            # single channel
            if len(v['shape']) == 2:
                frame = v['shape'][0]
            elif len(v['shape']) == 3:
                frame = v['shape'][1]
            else:
                raise ValueError
            # remove long short data
            if frame < self.min_frame or frame > self.max_frame:
                del feat_files[k]

            else:
                _utt += 1

        self.DT.prev_utt[set_type], self.DT.latest_utt[set_type] = self.DT.latest_utt[set_type], _utt
        return {"raw_json": feat_files}

    def __init__(self, data_structure, logger, max_frame=3000, min_frame=400):
        self.DT = data_structure
        self.logger = logger
        self.max_frame = max_frame
        self.min_frame = min_frame
        # just process train and dev
        self.process_set = [s for s in self.DT.sets if s in ["train", "dev"]]

    def __call__(self):
        # not resume in removing operation
        for set_type in self.process_set:
            self.DT.save(self.remove_data(set_type=set_type), set_type)
            self.logger.info(
                "Removing raw-feat files in {} set and successful saving,  number of utterance: {} -> {}".format(
                    set_type, self.DT.prev_utt[set_type], self.DT.latest_utt[set_type])
            )


# TODO: reconstruct it,
#   def __call__ -> get cmvn and apply cmvn
class CMVNer(object):
    def __init__(self, n_jobs, data_structure, logger, apply_type="global", paste_cmvn_path=None, **conf):
        self.DT = data_structure
        self.logger = logger
        self.cmvn_conf = conf
        # validate cmvn geting
        if paste_cmvn_path is not None and path.exists(paste_cmvn_path):
            self.cmvn_path = paste_cmvn_path
            self.logger.info("Get cmvn from paste_cmvn_path: {}".format(paste_cmvn_path))
        elif "train" in self.DT.sets:
            cmvn_path = self.DT.path["train"]["cmvn_npy"]
            if path.exists(cmvn_path):
                self.cmvn_path = cmvn_path
                self.logger.info("Get cmvn from exist train set: {}".format(cmvn_path))
            else:
                self.cmvn_path = None
                self.cmvn_computer = multi_processing(func=compute_cmvn, n_jobs=n_jobs, gather="sum")
        else:
            raise ValueError("No train set and no paste_cmvn_path, can't get cmvn")

        # cmvn apply
        self.app_type = apply_type
        if apply_type == "global":
            self.cmvn_applier = multi_processing(func=apply_global_cmvn, n_jobs=n_jobs, gather="dict")
        elif apply_type == "utterance":
            self.cmvn_applier = multi_processing(func=apply_utterance_cmvn, n_jobs=n_jobs, gather="list")
            raise NotImplementedError

    def get_cmvn(self):
        if self.cmvn_path is not None:
            self.cmvn = np.load(self.cmvn_path)
        else:
            with open(self.DT.path["train"]["raw_json"], 'r') as fp:
                tr_raw_files = json.load(fp)
            self.cmvn = self.cmvn_computer(tr_raw_files)
            self.DT.save({"cmvn_npy": self.cmvn}, set_type="train")

    def apply_cmvn(self, set_type):
        with open(self.DT.path[set_type]["raw_json"], 'r') as fp:
            raw_feats = json.load(fp)
            if self.app_type == "global":
                dump_feats = self.cmvn_applier(
                    dict(zip(raw_feats.keys(), [v["feat"] for v in raw_feats.values()])),
                    feat_dir=self.DT.path[set_type]["dump"],
                    cmvn=self.cmvn,
                    **self.cmvn_conf
                )
            else:
                raise NotImplementedError

        return {"dump_json": dump_feats}

    def __call__(self):
        # 1. Get CMVN
        self.get_cmvn()
        # 2. Apply CMVN to all sets
        for set_type in self.DT.sets:
            if self.DT.check_resume(set_type, *["dump_json"]):
                self.logger.info(
                    "File -> dump feats files exist in set {}, skip".format(set_type))
            else:
                self.DT.save(self.apply_cmvn(set_type), set_type=set_type)
                self.logger.info(
                    "File -> dump feats json files in {} set successful saving ".format(set_type))
