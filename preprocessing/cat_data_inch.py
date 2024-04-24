import os
import os.path as path
import json
import numpy as np
import copy
from tqdm import tqdm

"""
This code is utilized to cat the air conducted and bone conducted data
e.g.
feat_a -> (T, D)
feat_b -> (T, D)
feat_output -> (C=2, T, D)
"""
def cat_data(out_root, air_root, bone_root, air_tgt_set, bone_tgt_set, tag=""):
    # cat data1, data2, ... dataN
    # generate -> data.json target_data_narray

    feat_out_dir = f'{out_root}/{air_tgt_set}/dump'
    os.makedirs(feat_out_dir, exist_ok=True)

    data_out_file = f'{out_root}/{air_tgt_set}/data/data.json'
    os.makedirs(path.split(data_out_file)[0], exist_ok=True)

    # read org json data
    # air
    data_air_json = f'{air_root}/{air_tgt_set}/data/data.json'

    with open(data_air_json, 'r') as fp:
        data_air = json.load(fp)
    # bone
    data_bone_json = f'{bone_root}/{bone_tgt_set}/data/data.json'
    with open(data_bone_json, 'r') as fp:
        data_bone = json.load(fp)

    air_dict = data_air['utts']
    bone_dict = data_bone['utts']

    # check match
    assert len(air_dict) == len(bone_dict), 'data len mismatch'

    # new_dict
    new_dict = copy.deepcopy(air_dict)

    pbar = tqdm(new_dict)
    for k, v in new_dict.items():
        pbar.update()
        feat_air = np.load(air_dict[k]['input'][0]['feat']) # (T, D)
        feat_bone = np.load(bone_dict[k.replace("-ns", "")]['input'][0]['feat']) # (T, D)

        feat_two_str = np.append(feat_air[np.newaxis, :], feat_bone[np.newaxis, :], axis=0)

        assert feat_two_str.ndim == 3, "feat ndim is not 3 but {}".format(feat_two_str.ndim)
        assert feat_two_str.shape[0] == 2, "feat channel axis is not 2 but {}".format(feat_two_str.size(0))

        # write new feat
        file_name = path.split(air_dict[k]['input'][0]['feat'])[-1]
        feat_file = path.join(feat_out_dir, file_name)
        np.save(feat_file, feat_two_str)

        # write data json
        new_dict[k]["input"][0]["feat"] = feat_file
        new_dict[k]["input"][0]["shape"] = feat_two_str.shape
        new_dict[k].update({"utt2spk": new_dict[k]["input"][0]["utt2spk"]})

    with open(data_out_file, 'w') as wp:
        json.dump({'utts': new_dict}, wp, ensure_ascii=False, indent=4)