#!/usr/bin/env python3
# Lint as: python3

# MIT License
#
# Copyright 2019 Google LLC
# Copyright (c) 2019 Franck Dernoncourt, Jenny Lee, Tom Pollard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from absl import flags

flags.DEFINE_boolean('shell', False, 'Start shell once ready.')

flags.DEFINE_boolean('fit', True, 'Start nn.fit() after maybe shell.')

flags.DEFINE_boolean('profile', False, 'Whether to use cProfile.')

flags.DEFINE_string(
    'profile_output', None,
    'If profile is True then cProfile output is written to the file path.')

flags.DEFINE_string(
    'dataset_text_folder', None,
    'Dataset text folder under ./data eg "i2b2_2014_deid". '
    'Results under this name are created under output_folder.')

flags.DEFINE_string(
    'output_folder', None,
    'Folder to place outputs of dataset_text_folder runs '
    'eg for "oct-30" and dataset "i2b2-14", '
    '"oct-30/i2b2-14/timestamp/..." results.')

flags.DEFINE_boolean('train', False, 'Whether to train model')

flags.DEFINE_boolean('eval', False, 'Whether to eval model')

flags.DEFINE_string('pretrained_model_folder', None,
                    'Model folder under ./trained_models if --eval is set')

flags.DEFINE_float('recall_inference_bias', None, 'Recall bias value')

flags.DEFINE_enum('token_embedding_dimension', None, ['100', '300'],
                  'Token embedding dimension size.')

flags.DEFINE_integer('threads_tf', 32, 'Num threads for any tf op.')

flags.DEFINE_integer(
    'threads_prediction', 100,
    'Num threads for eval prediction data partitioning '
    'into chucks for that amount of threads.')

flags.mark_flags_as_required(
    ['dataset_text_folder', 'output_folder', 'token_embedding_dimension'])

flags.mark_bool_flags_as_mutual_exclusive(['train', 'eval'], required=True)

flags.register_multi_flags_validator(
    flag_names=['train', 'recall_inference_bias'],
    multi_flags_checker=lambda flags: not flags['train'] or flags[
        'recall_inference_bias'] in [None, 0.],
    message='In train mode, recall_inference_bias must be unset or zero.')

flags.register_multi_flags_validator(
    flag_names=['eval', 'pretrained_model_folder'],
    multi_flags_checker=lambda flags: not flags['eval'] or all(flags.values()),
    message='In eval mode, all these flags must be set.')
