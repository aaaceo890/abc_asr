import os
import re
import logging
from abc import abstractmethod
from CDPR.utils.validate import validate_audio_text


class DataGenerator(object):
    @staticmethod
    @abstractmethod
    def get_raw_text(text_files, audio_files, logger=None):
        raise NotImplementedError

    @staticmethod
    def format_name(name, regex=None):
        """
        Return a split name
        e.g.
        >>> name = "Utt1_ch01"
        >>> regex = "_ch"
        >>> name, ch = "Utt1", "01"
        """
        name = name.split('.')[0]
        if regex is None:
            return name
        else:
            try:
                name, _ = tuple(name.split(regex))
            except ValueError:
                print('Error split {} with regex {}'.format(name, regex))
            return name

    @staticmethod
    def walk_dir(target_dir, format_func, name_regex, channel_regex=None):
        """
        :param target_dir: root directory to walk
        :param format_func: file name to desire name, split channel number
        :param name_regex: constraint of file name [Regular expression: .*Name]
        :param channel_regex: divide the channel numbers from file name
        :return: A dict store target files,
        e.g. {
             "utt1": [/path/to/utt1_ch0, /path/to/utt1_ch1, ...].
             "utt2": [...],
             ...
             }
        """
        target_files = dict()
        for root, dirs, files in os.walk(target_dir, topdown=False):
            for f in files:
                # audio.wav -> ['audio', '.wav']
                if re.search(name_regex, f) is not None:
                    name = format_func(f, channel_regex)
                    if name not in target_files:
                        target_files[name] = ['{}/{}'.format(root, f)]
                    else:
                        target_files[name].append('{}/{}'.format(root, f))
        if len(target_files) == 0:
            raise ValueError("There is no file in dir {}".format(target_dir))
        return target_files

    @staticmethod
    def get_utt2spk(audio_files, spk_regex=None):
        utt2spk = dict()
        for k in audio_files.keys():
            if spk_regex is None:
                spk = 'default'
            else:
                spk_match = re.search(spk_regex, k)
                if spk_match is None:
                    raise ValueError('Wrong match with {} and {}'.format(spk_regex, k))
                spk = spk_match.group()
            assert type(spk) is str, 'not support speaker format like this {}'.format(spk)
            utt2spk[k] = spk
        return utt2spk

    def __init__(self, data_name, data_structure, logger=None):
        self.data_name = data_name
        self.DT = data_structure
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger()
            self.logger.addHandler(logging.StreamHandler())

    def __call__(self):
        # Generate all files (include audio, text, utt2spk)
        # Loop all sets
        for set_type in self.DT.sets:
            if self.DT.check_resume(set_type, *["audio_json", "text_json", "utt2spk_json"]):
                self.logger.info(
                    "File -> audio/text/utt2spk files exist in {} set, skip".format(set_type))
                # TODO: get utt nums
            else:
                self.logger.info(
                    "File -> audio/text/utt2spk files not exist in {} set, generating...".format(set_type))
                self.DT.save(self._get_data_set(set_type), set_type)
                self.logger.info(
                    "File -> audio/text/utt2spk json files in {} set successful saving ".format(set_type))

    def _get_data_set(self, set_type):
        # Generate files in a certain set
        # make sure is list
        # audio_path_list -> train: [train_dir1, train_dir2]
        audio_path_list = list(self.DT.path[set_type]["audio"])
        text_path_list = list(self.DT.path[set_type]["text"])

        # step 1. walk all dirs and find target files
        audio_files = dict()
        raw_text_files = dict()
        for speech_path in audio_path_list:
            audio_files.update(
                self.walk_dir(
                    target_dir=speech_path,
                    format_func=self.format_name,
                    name_regex=self.DT.audio_name_regex,
                    channel_regex=self.DT.channel_regex
                )
            )
        for text_path in text_path_list:
            raw_text_files.update(
                self.walk_dir(
                    target_dir=text_path,
                    format_func=self.format_name,
                    name_regex=self.DT.text_name_regex
                )
            )

        # step 2. Process raw text files (You should complete "get_raw_text" in your customized data-generator)
        # Pass in audio files to delete some utterances
        text_files, audio_files = self.get_raw_text(raw_text_files, audio_files, logger=self.logger)

        # step 3. Format and validate audio and text files
        audio_files, text_files, _utt = validate_audio_text(
            audio_files,
            text_files,
            channel_nums=self.DT.channel_nums,
            channel_regex=self.DT.channel_regex,
        )
        self.DT.prev_utt[set_type], self.DT.latest_utt[set_type] = self.DT.latest_utt[set_type], _utt

        self.logger.info('Successfully validate audio/text for {} set, number of utterance: {} -> {}'.format(
            set_type, self.DT.prev_utt[set_type], self.DT.latest_utt[set_type]
        ))

        # step 4. get utt2spk
        utt2spk = self.get_utt2spk(audio_files, self.DT.spk_regex)

        return {"audio_json": audio_files, "text_json": text_files, "utt2spk_json": utt2spk}
