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

import io
import json
import types

from absl import app
from absl import flags
import IPython
import pandas as pd

flags.DEFINE_boolean('csv', False, '')

flags.DEFINE_boolean('shell', False, 'Start shell once ready.')

flags.DEFINE_list('datasets', None, 'Dir names in output to compare.')

flags.DEFINE_list('metrics', ['token', 'binary'],
                  'comma separated list of metrics.')

flags.DEFINE_integer('epoch', None, 'Max epoch to compare.')

FLAGS = flags.FLAGS


def df_for_dataset(dataset):
  filename = './output/{dataset}/results.json'.format(dataset=dataset)
  try:
    ds = json.load(open(filename))
  except:
    print('Cannot read {filename}'.format(filename=filename))
    quit()
  epoch = max(map(int, ds['epoch'].keys()))
  epoch = min(FLAGS.epoch or epoch, epoch)
  df = {}
  for metric in FLAGS.metrics:
    data = ds['epoch'][str(epoch)][0]['test'][metric]['classification_report']
    df[metric] = pd.read_table(
        io.StringIO(data), delimiter=r'\s{2,}', engine='python')
  return types.SimpleNamespace(df=df, epoch=epoch)


def main(unused_argv):
  for dataset in FLAGS.datasets:
    res = df_for_dataset(dataset)
    print('dataset={dataset} epoch={epoch}'.format(
        dataset=dataset, epoch=res.epoch))
    for d in res.df.values():
      print(d.to_csv() if FLAGS.csv else d)
  if FLAGS.shell:
    IPython.start_ipython(argv=[], user_ns=dict(globals(), **locals()))


if __name__ == '__main__':
  flags.mark_flags_as_required(['datasets'])
  app.run(main)
