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

import importlib
import os
import time

from absl import app
from absl import flags
from absl import logging

myflags = importlib.import_module('flags', '.')

FLAGS = flags.FLAGS


def run():
  timestamp = time.time()

  basedirpath = os.path.join('./output', FLAGS.output_folder,
                             FLAGS.dataset_text_folder)
  dirpath = os.path.join(basedirpath, str(timestamp))
  os.makedirs(dirpath, exist_ok=True)

  FLAGS.dataset_text_folder = './data/' + FLAGS.dataset_text_folder
  FLAGS.output_folder = dirpath
  FLAGS.profile_output = os.path.join(dirpath, 'prof')

  train_flags = ' '.join(
      '--{name}="{value}"'.format(name=flag.name, value=flag.value)
      for flag in [
          flag for flag in FLAGS.get_key_flags_for_module(myflags)
          if flag.value is not None
      ])
  cmd = 'stdbuf -o 0 -e 0 ./train.py {train_flags} 2>&1 | tee "{log}"'.format(
      train_flags=train_flags, log=os.path.join(dirpath, 'log'))
  logging.info('Running command: "%s"', cmd)
  assert 0 == os.system(cmd)


def main(unused_argv):
  run()


if __name__ == '__main__':
  app.run(main)
