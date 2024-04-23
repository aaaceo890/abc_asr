#!/usr/bin/env python3
import os
import os.path as path
import argparse
import sys

from modules.DataStructure import DataStructure
from modules.FeatureModules import FeatureMaker, Remover, CMVNer
from modules.TextProcessor import TextProcessor
from modules.data2json import Data2Json
from modules.Logger import get_logger
from modules.DataGenerator import BoneConductDataGenerator, BoneConductNSDataGenerator
from utils.set_config import set_config

from cat_data_inch import cat_data

def get_data_structure(dataset_root, soundtrack="air", test=False):
    # soundtrack: 0 -> air, 1 -> bone
    ds = dict()
    ds.update({
        "audio_type": ".wav",
        "text_type": ".txt",
        "spk_regex": "Speaker[0-9]{1,3}",
        "channel_nums": 1,
        "speeds": [0.9, 1, 1.1],
        "path": {
            "test": {
                "audio": [path.join(dataset_root, "Audio/test")],
                "text": [path.join(dataset_root, "script/test")]
            }
        }
    })
    if not test:
        ds["path"].update(
            {"train": {
                "audio": [path.join(dataset_root, "Audio/train")],
                "text": [path.join(dataset_root, "script/train")]
            }})
        ds["path"].update(
            {"dev": {
                "audio": [path.join(dataset_root, "Audio/dev")],
                "text": [path.join(dataset_root, "script/dev")]
            }})
    if soundtrack == "air":
        ds["soundtrack"] = 0
    elif soundtrack == "air_ns":
        ds["path"] = {}
        # TODO:
        for ns_set in  ["test_-5db", "test_0db", "test_+5db", "test_+10db", "test_+15db", "test_+20db"]:
            ds["path"].update(
                {
                    ns_set: {
                        "audio": [path.join(dataset_root, f"Audio/{ns_set}")],
                        "text": [path.join(dataset_root, "script/test")]
                    }
                }
            )
    else:
        ds["soundtrack"] = 1
    return ds


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description='data generator config')

    base_group = parser.add_argument_group(title='base option')
    base_group.add_argument('--dataset_root', type=str, required=True)
    base_group.add_argument('--test', action="store_true")
    base_group.add_argument('--verbose', type=int, default=1,
                            help='0 for debug, 1 for info, 2 for warning, 3 for error, 4 for critical')
    base_group.add_argument('--n_jobs', type=int, default=16)

    config_group = parser.add_argument_group(title='config file path')
    config_group.add_argument('--feature_conf', type=str, default=None)

    feature_group = parser.add_argument_group(title='generate feature settings')
    feature_group.add_argument('--feature_type', type=str, default="fbank", choices=["fbank", "mfcc"])
    feature_group.add_argument('--cmvn_apply_type', type=str, default="global", choices=["global", "utterance"])
    feature_group.add_argument('--do_delta', type=bool, default=False)
    feature_group.add_argument('--min_frame', type=int, default=400)
    feature_group.add_argument('--max_frame', type=int, default=3000)

    dict_group = parser.add_argument_group(title='dictionary settings')
    dict_group.add_argument('--nbpe', type=int, default=5000)
    dict_group.add_argument('--bpemode', type=str, default='unigram')

    return parser


def prep(args, workdir, generator_model, soundtrack, min_frame=0,
         paste_dict_file=None, paste_cmvn_path=None):

    workdir = path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)

    # --- config ---
    feature_conf = set_config(args.feature_type, None)
    text_conf = {"char": True}
    cmvn_conf = set_config("cmvn", None)
    n_jobs = args.n_jobs
    """
    Initialize all modules
    """
    # Logger
    logdir = path.join(workdir, "log")
    os.makedirs(logdir, exist_ok=True)
    logger = get_logger(verbose=args.verbose, log_out_dir=logdir)

    # Data structure
    data_structure = DataStructure(get_data_structure(dataset_root=args.dataset_root, soundtrack=soundtrack, test=args.test), workdir, logger)

    data_generator = generator_model.DataGenerator(
        data_name=generator_model.__name__.split(".")[-1],
        data_structure=data_structure,
        logger=logger,
    )

    # Feature making related
    feature_maker = FeatureMaker(n_jobs, data_structure, logger, args.feature_type, **feature_conf)
    remover = Remover(
        data_structure,
        logger,
        min_frame=args.min_frame,
        max_frame=args.max_frame
    )
    cmvner = CMVNer(
        n_jobs,
        data_structure,
        logger,
        apply_type=args.cmvn_apply_type,
        paste_cmvn_path=paste_cmvn_path,
        **cmvn_conf,
    )

    # Text processor
    text_processor = TextProcessor(
        n_jobs,
        data_structure,
        paste_dict_file,
        None,
        mode="char",
        logger=logger,
        **text_conf,
    )

    # Data to Json convertor
    data2json = Data2Json(
            data_structure=data_structure,
            logger=logger,
            file_type="npy"
        )

    """
    1. Get Data files
        Automatically save audio.json, text.json and utt2spk.json
        The path is store in data_structure
    """
    logger.info('Generate data files...')
    data_generator()

    """
    2. Get Feature
        Read the path store in data structure (audio.json, text.json, utt2spk.json)
        Output numpy format raw feats (save in featdir/xxx)
        Save path-related json file in data structure (raw_json)
    """
    logger.info('Feature making...')
    feature_maker()

    logger.info('Removing data...')
    remover()

    """
    3. Processing of train set
        Read the path store cmvn_npy and the path store raw feat files
        Output cmvn.npy and the processing feats
        Save the path of cmvn and processing feats to data structure
    """
    logger.info('Computing CMVN...')
    cmvner()

    """
    4. Text processing
        Generate dictionary from train set or exist file
        Get token and token_id for all given sets
    """
    logger.info('Getting dictionary...')
    text_processor.get_dict()

    logger.info('Text Transforming...')
    text_processor.text_transform()

    """
    5. Data2json
        Convert all data to a json file, and
        this json file store all information 
    """
    logger.info('Converting data to json...')
    data2json(
        token_files=text_processor.token_dict,
        tokenid_files=text_processor.id_dict,
        vocab_size=len(text_processor.dict),
    )

    logger.info('Done.')


def main(cmd_args):
    # --- get parser ---
    parser = get_parser()
    args = parser.parse_args(cmd_args)
    # preprocess bone data
    bone_workdir = "../data/preprocess_bone"
    prep(args, bone_workdir, BoneConductDataGenerator, soundtrack="bone",
         paste_dict_file="../data/feature/char_dict.txt", paste_cmvn_path="../data/feature/cmvn_bone.npy"
         )

    # preprocess air data
    air_workdir = "../data/preprocess_air"
    prep(args, air_workdir, BoneConductDataGenerator, soundtrack="air",
         paste_dict_file="../data/feature/char_dict.txt", paste_cmvn_path="../data/feature/cmvn_air.npy",
         )

    # preprocess noisy air data
    prep(args, air_workdir, BoneConductNSDataGenerator, soundtrack="air_ns",
         paste_dict_file="../data/feature/char_dict.txt", paste_cmvn_path="../data/feature/cmvn_air_ns.npy"
         )

    # cat data
    out_root = path.abspath("../data/preprocess_stack")
    bone_tgt_set = "test"
    for air_tgt_set in ["test", "test_-5db", "test_0db", "test_+5db", "test_+10db", "test_+15db", "test_+20db"]:
        cat_data(out_root, air_workdir, bone_workdir, air_tgt_set, bone_tgt_set)

    print("Done.")


if __name__ == "__main__":
    main(sys.argv[1:])