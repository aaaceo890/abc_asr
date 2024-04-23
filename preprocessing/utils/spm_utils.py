#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# https://github.com/pytorch/fairseq/blob/master/LICENSE

# see document
# https://github.com/google/sentencepiece

import sys
import sentencepiece as spm
import contextlib
import os.path as path
# TODO: spm_train
# TODO: if can direct run in python?
def spm_train(input, vocab_size, model_type, model_prefix, input_sentence_size):
    """
    :param input:
    :param vocab_size:
    :param model_type:
    :param model_prefix:
    :param input_sentence_size:
    :return:
    """
    # convert to string type
    run_args = '--input={} --vocab_size={} --model_type={} --model_prefix={} --input_sentence_size={}'.format(
        input, vocab_size, model_type, model_prefix, input_sentence_size
    )
    # org format -> spm.SentencePieceTrainer.Train(" ".join(sys.argv[1:]))
    spm.SentencePieceTrainer.Train(run_args)


def spm_encode(model, inputs: list = '-', outputs=None, pro_type='file', output_format='piece', min_len=None, max_len=None,
               logger=None, log_level=1):
    """
    :param model:
    :param inputs:
    :param outputs:
    :param pro_type:
    :param output_format:
    :param min_len:
    :param max_len:
    :return:
    """
    # check input
    if type(inputs) is not list:
        inputs = [inputs]
    in_type = 'file'
    for inp in inputs:
        if not path.exists(inp):
            in_type = 'python'
            break

    # check output
    if outputs is not None:
        out_type = 'file'
        if type(outputs) is not list:
            outputs = [outputs]
        if in_type == out_type:
            assert len(inputs) == len(outputs), \
                "number of input and output paths should match"
    else:
        out_type = 'python'

    sp = spm.SentencePieceProcessor()
    sp.Load(model)

    if output_format == "piece":
        def encode(l):
            return sp.EncodeAsPieces(l)
    elif output_format == "id":
        def encode(l):
            return list(map(str, sp.EncodeAsIds(l)))
    else:
        raise NotImplementedError

    if min_len is not None or max_len is not None:
        def valid(line):
            return (
                    (min_len is None or len(line) >= min_len) and
                    (max_len is None or len(line) <= max_len)
            )
    else:
        def valid(lines):
            return True

# TODO: redirect in/output
    with contextlib.ExitStack() as stack:
        if in_type == 'file':
            inputs = [
                stack.enter_context(open(input, "r", encoding="utf-8"))
                if input != "-" else sys.stdin
                for input in inputs
            ]
        elif in_type == 'python':
            inputs = [
                input.split('\n')
                for input in inputs
            ]
        else:
            raise ValueError

        if out_type == 'file':
            outputs = [
                stack.enter_context(open(output, "w", encoding="utf-8"))
                if output != "-" else sys.stdout
                for output in outputs
            ]

        elif out_type == 'python':
            outputs = [None] * len(inputs)
            re_output = []
        else:
            raise ValueError

        stats = {
            "num_empty": 0,
            "num_filtered": 0,
        }

        def encode_line(line):
            line = line.strip()
            if len(line) > 0:
                line = encode(line)
                if valid(line):
                    return line
                else:
                    stats["num_filtered"] += 1
            else:
                stats["num_empty"] += 1
            return None

        for i, lines in enumerate(zip(*inputs), start=1):
            enc_lines = list(map(encode_line, lines))
            if not any(enc_line is None for enc_line in enc_lines):
                for enc_line, output_h in zip(enc_lines, outputs):
                    if out_type == 'file':
                        print(" ".join(enc_line), file=output_h)
                    elif out_type == 'python':
                        re_output.append(" ".join(enc_line))
            if i % 10000 == 0:
                if logger is not None:
                    if log_level == 0:
                        logger.debug("processed {} lines".format(i))
                    else:
                        logger.info("processed {} lines".format(i))
                else:
                    print("processed {} lines".format(i), file=sys.stderr)
        if logger is not None:
            if log_level == 0:
                logger.debug("skipped {} empty lines".format(stats["num_empty"]))
                logger.debug("filtered {} lines".format(stats["num_filtered"]))
            else:
                logger.info("skipped {} empty lines".format(stats["num_empty"]))
                logger.info("filtered {} lines".format(stats["num_filtered"]))
        else:
            print("skipped {} empty lines".format(stats["num_empty"]), file=sys.stderr)
            print("filtered {} lines".format(stats["num_filtered"]), file=sys.stderr)
        if out_type == 'python':
            return re_output


# TODO: spm_decode
def spm_decode(model, input=None, input_format='piece'):
    """
    :param model:
    :param input:
    :param input_format:
    :return:
    """
    sp = spm.SentencePieceProcessor()
    sp.Load(model)

    if input_format == "piece":
        def decode(l):
            return "".join(sp.DecodePieces(l))
    elif input_format == "id":
        def decode(l):
            return "".join(sp.DecodeIds(l))
    else:
        raise NotImplementedError

    def tok2int(tok):
        # remap reference-side <unk> (represented as <<unk>>) to 0
        return int(tok) if tok != "<<unk>>" else 0

    if input is None:
        h = sys.stdin
    else:
        h = open(input, "r", encoding="utf-8")
    for line in h:
        print(decode(line.split()))
