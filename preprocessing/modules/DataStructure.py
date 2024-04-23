import os
import os.path as path
import yaml
import logging
import json
import numpy as np


class DataStructure(object):
    @staticmethod
    def get_attr(args, attr, default=None):
        if attr in args and args[attr] is not None:
            return args[attr]
        # elif default == "":
        #     raise ValueError("The attr {} is invalid and default is not provided".format(attr))
        else:
            return default

    def __init__(self, load_structure, tgt_dir, logger):
        # with open(filepath, 'r', encoding='utf-8') as fp:
        #     load_structure = yaml.load(fp)
        # sound src info
        self.audio_type = self.get_attr(load_structure, "audio_type", ".wav")
        self.audio_name_regex = self.get_attr(load_structure, "audio_name_regex", ".*" + self.audio_type)
        # text src info
        self.text_type = self.get_attr(load_structure, "text_type", ".txt")
        self.text_name_regex = self.get_attr(load_structure, "text_name_regex", ".*" + self.text_type)
        # speaker info
        self.spk_regex = self.get_attr(load_structure, "spk_regex")
        # channel info
        self.channel_nums = self.get_attr(load_structure, "channel_nums", 1)
        self.channel_regex = self.get_attr(load_structure, "channel_regex", None)
        if self.channel_nums > 1:
            assert self.channel_regex is not None, 'channel_nums > 1, but there is no regex for channel'
        # other info
        self.soundtrack = self.get_attr(load_structure, "soundtrack", None)
        self.speeds = self.get_attr(load_structure, "speeds", [None])
        # self.add_info = self.get_attr(load_structure, "add_info", dict())
        # # TODO: check add_info, speed_regex to global params
        # for _info in ["soundtrack", "speeds"]:
        #     if _info not in self.add_info:
        #         self.add_info[_info] = None

        # divide data set [file path]
        # path:
        #   train:
        #       audio:
        #       text:
        #   dev:
        self.path = load_structure["path"]
        # add directories
        self.lang = path.join(tgt_dir, "lang")
        os.makedirs(self.lang, exist_ok=True)
        self.sets = self.path.keys()
        # number of utterances recorded
        self.prev_utt = dict(zip(self.sets, [0]*len(self.sets)))
        self.latest_utt = dict(zip(self.sets, [0]*len(self.sets)))

        # TODO: check train or dev name match ?
        # for s in self.sets:
        #     assert s in ["train", "dev", "test"], "Unknown set name {}".format(s)

        # generate file related
        """
        File Structure:
        tgt_dir
            set_1
                data
                    audio.json -> store original audio file e.g. utt1 ["ch1", "ch2", ...]
                    text.json -> store text file e.g. utt1 "text"
                    utt2spk.json -> e.g. utt1 speaker
                    raw.json -> store raw feat file  [espnet like structure]
                    dump.json -> store processed feat file
                    data.json -> store finally data information file
                raw
                    ...
                dump
                    ...
            set_2
                ...
        """
        self.tgt_dir = tgt_dir
        for k, v in self.path.items():
            # k -> set_type
            # v -> all files in this set
            # Public path
            v.update(
                {
                    "audio_json": path.join(tgt_dir, k, "data", "audio.json"),
                    "text_json": path.join(tgt_dir, k, "data", "text.json"),
                    "utt2spk_json": path.join(tgt_dir, k, "data", "utt2spk.json"),
                    "raw_json": path.join(tgt_dir, k, "data", "raw.json"),
                    "dump_json": path.join(tgt_dir, k, "data", "dump.json"),
                    "data_json": path.join(tgt_dir, k, "data", "data.json"),
                    "raw": path.join(tgt_dir, k, "raw"),
                    "dump": path.join(tgt_dir, k, "dump"),
                }
            )
            # Make the directory to store files
            for ndir in ["data", "raw", "dump"]:
                os.makedirs(path.join(tgt_dir, k, ndir), exist_ok=True)
            # Special path
            if k == "train":
                v.update(
                    {
                        "cmvn_npy": path.join(tgt_dir, k, "data", "cmvn.npy"),
                    }
                )
            # After added ->
            # path:
            #   set:
            #       audio(org):
            #       text(org):
            #       audio_json
            #       text_json
            #       utt2spk_json
            #       raw_json
            #       dump_json
            #       data_json
            #       raw(dir)
            #       dump(dir)
            #       cmvn_npy(train only)

        self.logger = logger

    def check_resume(self, set_type, *file_list):
        resume = True
        for _file in file_list:
            if not path.exists(self.path[set_type][_file]):
                resume = False
                self.logger.debug("{} not exist".format(_file,))

        if resume:
            # calculate utt nums
            for _file in file_list:
                if _file in ["audio_json", "raw_json", "dump_json"]:
                    with open(self.path[set_type][_file], "r") as fp:
                        self.latest_utt[set_type] = len(json.load(fp))

        return resume

    def save(self, data_dict, set_type):
        for _file in data_dict.keys():
            if "json" in _file:
                with open(self.path[set_type][_file], 'w', encoding='utf-8') as wp:
                    json.dump(data_dict[_file], wp, ensure_ascii=False, indent=4)
                    self.logger.debug("saving {} in set {} to {}".format(_file, set_type, self.path[set_type][_file]))
            elif "npy" in _file:
                np.save(self.path[set_type][_file], data_dict[_file])
            else:
                raise ValueError("Unknown type of file {}".format(_file))


