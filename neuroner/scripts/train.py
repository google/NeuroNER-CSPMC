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

import cProfile
import importlib
import os

from absl import app
from absl import flags
from neuroner import neuromodel
import IPython

_ = importlib.import_module('flags', '.')

FLAGS = flags.FLAGS


def real_main():
  if FLAGS.dataset_text_folder:
    model_flags = {
        'dataset_text_folder':
            FLAGS.dataset_text_folder,
        'output_folder':
            FLAGS.output_folder,
        'train_model':
            FLAGS.train and not FLAGS.eval,
        'use_pretrained_model':
            not FLAGS.train and FLAGS.eval,
        'pretrained_model_folder':
            FLAGS.pretrained_model_folder
            and os.path.join('./trained_models', FLAGS.pretrained_model_folder),
        'recall_inference_bias':
            FLAGS.recall_inference_bias,
        'token_pretrained_embedding_filepath':
            './data/word_vectors/glove.{dim_length}B.{dim_width}d.txt'.format(
                dim_length={
                    '100': '6',
                    '300': '840'
                }[FLAGS.token_embedding_dimension],
                dim_width=FLAGS.token_embedding_dimension),
        'token_embedding_dimension':
            int(FLAGS.token_embedding_dimension),
        'token_lstm_hidden_state_dimension':
            int(FLAGS.token_embedding_dimension),
        'number_of_cpu_threads':
            FLAGS.threads_tf,
        'number_of_cpu_threads_prediction':
            FLAGS.threads_prediction,
    }
    model_flags = {k: v for k, v in model_flags.items() if v is not None}
  else:
    model_flags = {}
  nn = neuromodel.NeuroNER(**model_flags)
  if FLAGS.fit:
    nn.fit()
  if FLAGS.shell:
    IPython.start_ipython(argv=[], user_ns=dict(globals(), **locals()))


def main(unused_argv):
  if FLAGS.profile:
    assert FLAGS.profile_output, ('profile_output must be specified when '
                                  'profile is True')
    print('################### PROFILING CODE ###################')
    cProfile.run('real_main()', filename=FLAGS.profile_output)
  else:
    real_main()


if __name__ == '__main__':
  app.run(main)
