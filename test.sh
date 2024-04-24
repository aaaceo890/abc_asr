#!/bin/bash

export ESPNETROOT=/home/Chenjq/Project/espnet
export PYTHONPATH=$PWD:$PYTHONPATH

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

ln -s ${KALDI_ROOT}/egs/wsj/s5/utils
ln -s ${KALDI_ROOT}/egs/wsj/s5/steps
. utils/parse_options.sh || exit 1;

set -e
set -u
set -o pipefail

export CUDA_VISIBLE_DEVICES=2
nj=2
# general configuration
# recog_set="test test_-5db test_0db test_+5db test_+10db test_+15db test_+20db"
recog_set="test_-5db test_0db test_+5db test_+10db test_+15db test_+20db"
dict=./data/feature/char_dict.txt

pids=() # initialize pids
for rtask in ${recog_set}; do
(
decode_dir=decode_${rtask}
feat_recog_dir=./data/preprocess_stack/${rtask}/data
# split data
splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

${decode_cmd} JOB=1:${nj} ./exp/${decode_dir}/log/decode.JOB.log \
    asr_recog.py \
    --config conf/decode.yaml \
    --ngpu 1 \
    --backend pytorch \
    --batchsize 0 \
    --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
    --result-label ./exp/${decode_dir}/data.JOB.json \
    --model ./results/model.acc.best  \
    --api v2

score_sclite.sh ./exp/${decode_dir} ${dict}

) &
pids+=($!) # store background pids
done
i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
[ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
echo "Finished"
