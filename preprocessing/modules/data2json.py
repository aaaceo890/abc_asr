# from CDPR.utils.decorators import multi_processing
import json
from tqdm import tqdm
import re

from utils.set_config import SPEED_FORMAT, SPEED_REGEX


class Data2Json(object):
    def __init__(
            self,
            data_structure,
            logger,
            file_type="npy",
    ):
        self.DT = data_structure
        self.file_type = file_type
        self.logger = logger

    def _get_data2json(self, set_type):
        with open(self.DT.path[set_type]["dump_json"], "r") as fp:
            feat_files = json.load(fp)
        with open(self.DT.path[set_type]["text_json"], "r") as fp:
            text_files = json.load(fp)
        with open(self.DT.path[set_type]["utt2spk_json"], "r") as fp:
            utt2spk_files = json.load(fp)

        pbar = tqdm(total=len(feat_files))
        # read feat files
        json_info = dict()
        for utt, feat in feat_files.items():
            sp_prefix = re.match(SPEED_REGEX, utt)
            if sp_prefix is not None:
                org_utt = utt.split(sp_prefix.group())[-1]
            else:
                org_utt = utt
            assert org_utt in utt2spk_files, "Utterance {} not in utt2spk"
            _utt2spk = utt2spk_files[org_utt]
            assert org_utt in text_files, "Utterance {} not in text"
            _text = text_files[org_utt]
            _token = self.token_files[set_type][org_utt]
            _token_id = self.tokenid_files[set_type][org_utt]

            _input_info = [{
                "feat": feat['feat'],
                "name": 'input1',
                "shape": feat['shape'],
                "utt2spk": utt2spk_files[org_utt],
                "filetype": self.file_type,
            }]

            _output_info = [{
                "name": "target1",
                "shape": [len(_token.split(' ')), self.vocab_size],
                "text": _text,
                "token": _token,
                "tokenid": _token_id
            }]

            json_info[utt] = {"input": _input_info, "output": _output_info, "utt2spk": utt2spk_files[org_utt]}
            pbar.update()
        pbar.close()

        return {"utts": json_info}

    def __call__(
            self,
            token_files,
            tokenid_files,
            vocab_size,):
        self.token_files = token_files
        self.tokenid_files = tokenid_files
        # ctc blank and EOS
        self.vocab_size = vocab_size + 2

        for set_type in self.DT.sets:
            self.logger.info("Converting set {}...".format(set_type))
            self.DT.save(
                    {"data_json": self._get_data2json(set_type)},
                    set_type
            )
            self.logger.info('Successfully generate data.json for {} set'.format(set_type))
