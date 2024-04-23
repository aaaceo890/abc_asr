import os
import os.path as path
import json
import random
from utils.spm_utils import spm_train, spm_encode
from utils.decorators import multi_processing


def _make_dict(text, oov='<unk>'):
    """
    :param text: a list of text
    :param oov: oov symbol, default <unk>
    :return: dict
    """
    # TODO: if text is a file path
    # extract all char
    char_list = []
    for t in text:
        for c in t.split(' '):
            char_list.append(c)

    sort_char_list = sorted(char_list)

    # make dict
    char_dict = [oov]

    for sc in sort_char_list:
        if sc != char_dict[-1]:
            char_dict.append(sc)

    return char_dict


def get_normal_dict(input, logger=None, char=True):
    if type(input) is str and path.exists(input):
        with open(input, 'r') as wp:
            text = wp.readlines()
    elif type(input) is dict:
        text = list(input.values())
    elif type(input) is list:
        text = input
    # TODO: task dependent
    pro_text = []
    for line in text:
        pro_text.append(text2normal_token(key=None, data=line, char=char))

    normal_dict = _make_dict(pro_text)
    if logger is not None:
        logger.info('get dict of {} length'.format(len(normal_dict)))

    return normal_dict


# multi_processing(n_jobs=16, gather="dict")
def text2normal_token(key, data, char=True, space='<space>', upper=True):
    # space is to split english in chinese expression
    # e.g. 听 beat it -> '听' 'b' 'e' 'a' 't' '<space>' 'i' 't'
    token = []
    for group in data.split():
        if char:
            if len(token) != 0 and space is not None:
                token.append(space)
            for c in group:
                token.append(c.upper() if upper else c.lower())
        else:
            token.append(group.upper() if upper else group.lower())

    if key is not None:
        return {key: " ".join(token)}
    else:
        return " ".join(token)


# bpe related
def get_bpe_dict(input, out_dir, logger, bpemode, bpetag, nbpe):
    # convert input type
    if type(input) is str:
        text_file = input
    else:
        text_file = path.join(out_dir, '_tmp_input_{}.txt'.format(random.randint(1, 1000)))
        if path.exists(text_file):
            os.remove(text_file)
    if type(input) is list:
        for t in input:
            with open(text_file, 'a+') as wp:
                wp.write('{}\n'.format(t))
    elif type(input) is dict:
        for t in input.values():
            with open(text_file, 'a+') as wp:
                wp.write('{}\n'.format(t))

    # define output file
    bpeprefix = path.join(out_dir, '_'.join([bpemode, str(nbpe), bpetag]))
    bpemodel = bpeprefix + '.model'

    spm_train(input=text_file, vocab_size=nbpe, model_type=bpemode, model_prefix=bpeprefix,
              input_sentence_size=100000000)
    logger.info('get bpe model: {}'.format(bpemodel))
    bpe_out = spm_encode(model=bpemodel, inputs=[text_file], output_format='piece', logger=logger, log_level=1)
    bpe_dict = _make_dict(bpe_out)

    return bpemodel, bpe_dict


# @multi_processing(n_jobs=16, gather="dict")
def text2bpe_token(key, data, model=None, logger=None):
    token = spm_encode(inputs=data, model=model, output_format='piece', logger=logger, log_level=0)[0]
    return {key: token}


# @multi_processing(n_jobs=16, gather="dict")
def token2int(key, token, char_dict, oov='<unk>'):
    tokenid = [char_dict.index(c) + 1 if c in char_dict else char_dict.index(oov) + 1 for c in token.split(' ')]
    tokenid = ' '.join(map(str, tokenid))
    return {key: tokenid}


class TextProcessor(object):
    @staticmethod
    def load_dict(dict_path):
        out_dict = []
        with open(dict_path, 'r', encoding='utf-8') as fp:
            for line in fp.readlines():
                out_dict.append(line.split(' ')[0])
        return out_dict

    def __init__(self, n_jobs, data_structure, paste_dict_file, paste_model_file, mode, logger, **conf):
        # global params
        assert mode in ["char", "word", "bpe"]
        self.mode = mode
        self.DT = data_structure
        self.out_dir = self.DT.lang
        self.dict_file = path.join(self.out_dir, '{}_dict.txt'.format(self.mode))
        if path.exists(self.dict_file):
            os.remove(self.dict_file)
        self.logger = logger
        self.text_conf = conf
        # TODO: params
        self.oov = '<unk>'
        self.space = '<space>'
        self.upper = True

        # dynamic params, initalize here
        self.dict = None
        self.model = None
        self.token_dict = dict()
        self.id_dict = dict()

        # get exist dict
        if paste_dict_file is not None:
            if self.mode == "bpe":
                assert paste_model_file is not None, "must provide bpe model file"
                self.model = paste_model_file
            else:
                self.model = None
            logger.info('paste')
            self.dict = self.load_dict(paste_dict_file)

        else:
            # prepare train text
            assert 'train' in self.DT.sets, 'Required train text to generate dict'
            self.tr_text_json = self.DT.path["train"]["text_json"]

        # text transform related
        if self.mode in ["char", "word"]:
            self.text2token = multi_processing(func=text2normal_token, n_jobs=n_jobs, gather="dict")
        elif self.mode == "bpe":
            self.text2token = multi_processing(func=text2bpe_token, n_jobs=n_jobs, gather="dict")
        self.token2int = multi_processing(func=token2int, n_jobs=n_jobs, gather="dict")

    def get_dict(self):
        # step 1. generate dictionary if dictionary is not been defined
        if self.dict is None:
            # generate dict from train set
            with open(self.tr_text_json, "r") as fp:
                input_text = json.load(fp)
            if self.mode in ["char", "word"]:
                self.dict = get_normal_dict(input_text, self.logger, **self.text_conf)
                self.model = None
            else:
                self.model, self.dict = get_bpe_dict(input_text, self.out_dir, self.logger, **self.text_conf)

        # step 2. save dict file
        for i, c in enumerate(self.dict):
            with open(self.dict_file, 'a+') as wp:
                wp.write('{} {}\n'.format(c, i + 1))
        self.logger.info('dictionary save in {}'.format(self.dict_file))

    def text_transform(self):
        # Do text transform
        for set_type in self.DT.sets:
            self.logger.info("For {} set".format(set_type))
            # text2token
            with open(self.DT.path[set_type]["text_json"], "r") as fp:
                _input_text = json.load(fp)
            if self.mode in ["char", "word"]:
                _token_dict = self.text2token(_input_text, space=self.space, upper=self.upper, **self.text_conf)
            else:
                assert self.model is not None, "No bpe model is provided"
                _token_dict = self.text2token(_input_text, model=self.model, logger=self.logger)

            # token2id
            _id_dict = self.token2int(_token_dict, char_dict=self.dict, oov=self.oov)

            self.token_dict.update({set_type: _token_dict})
            self.id_dict.update({set_type: _id_dict})


if __name__ == '__main__':
    # # text_files = {'111': 'AS I APPROACHED THE CITY I HEARD BELLS RINGING AND A LITTLE LATER I FOUND THE STREETS ASTIR WITH THRONGS OF WELL DRESSED PEOPLE IN FAMILY GROUPS WENDING THEIR WAY HITHER AND THITHER',
    # # '222': 'AS I APPR'}
    # bpemodel = '/home/Chenjq/espnet2/egs/librispeech/asr1/data/lang_char/train_960_unigram5000.model'
    # # text_files = {'111': '/home/Chenjq/espnet2/egs/librispeech/asr1/data/dev/1'}
    # text_files = ['A', 'A', 'D', 'R', 'W', 'A', 'V', 'B', 'M', 'Q', 'WWW', 'Z', 'S S A B D']
    # _make_dict(text_files)
    import json
    # text_json = '/home/Chenjq/espnet2/egs/boneconduct/data_air_v2/data/test/text.json'
    text_json = '/home/Chenjq/espnet2/egs/aishell2/data_v2/data/train/text.json'
    with open(text_json, 'r') as fp:
        text = json.load(fp)

    text2normal_token(text, char=True, space='<space>', upper=True)
    my_dict = get_normal_dict(text, char=True)

    pass

