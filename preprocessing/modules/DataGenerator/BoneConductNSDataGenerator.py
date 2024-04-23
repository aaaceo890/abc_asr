from CDPR.modules.DataGenerator.data_generator import DataGenerator as InterFace
import re

special_symbol = '[\u0021-\u0026,\u0028-\u002f,\u003a-\u0040,\u005b-\u0060,\u2000-\u206f,\u3000-\u303f,\uff00-\uffef]'
lattins = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
chs = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']


class DataGenerator(InterFace):
    @staticmethod
    def get_raw_text(text_files: dict, audio_files: dict, logger=None):
        noise_pattern = ['<DEAF>', 'NOISE>', '']
        text_info = dict()
        for v in text_files.values():
            for file in v:
                with open(file, 'r') as fp:
                    for line in fp.readlines():
                        name, text = line.split(' ', 1)
                        name = name + '-ns'
                        # drp noise symbol
                        drp_flag = 0
                        for p in noise_pattern:
                            if p == text.strip():
                                del audio_files[name]
                                logger.warning('drop {}, cause its symbol [{}] match [{}]'.format(name, text.strip(), p))
                                drp_flag = 1
                                continue
                        if drp_flag:
                            continue
                        else:
                            # drp symbol
                            indent_text = re.sub(special_symbol, '', text.strip())
                            # replace num to chinese
                            for l, c in zip(lattins, chs):
                                indent_text = re.sub(l, c, indent_text)
                            text_info[name] = indent_text
                            # no indent
                            # text_info[name] = indent_text.replace(' ', '')
        return text_info, audio_files
