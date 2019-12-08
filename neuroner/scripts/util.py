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

import time
import threading

from absl import app
from absl import flags
from absl import logging
import more_itertools
import plumbum

flags.DEFINE_boolean('eval', None, 'Run evaluation.')

flags.DEFINE_string('pretrained_model_folder', None,
                    'pretrained_model_folder relative to trained_models')

flags.DEFINE_list(
    'rbias', [], 'In eval mode, pass recall_inference_bias '
    'values to each spawned evaluation run by them.')

flags.DEFINE_boolean('eval-view-test-bias', None, 'See eval on test by bias.')

flags.DEFINE_string('dataset_text_folder', 'i2b2-2014-paper',
                    'dataset to eval on')

flags.DEFINE_enum('edim', None, ['100', '300'], 'Token embedding dimension')

FLAGS = flags.FLAGS

flags.mark_flags_as_required(['edim'])

# Must specify only one verb.
flags.mark_bool_flags_as_mutual_exclusive(['eval', 'eval-view-test-bias'],
                                          required=True)

flags.register_multi_flags_validator(
    flag_names=['eval', 'rbias', 'pretrained_model_folder'],
    multi_flags_checker=lambda flags: not flags['eval'] or all(flags.values()),
    message='In eval mode, all these flags must be set.')


def start_and_wait_for_jobs(processes, descriptions):
  assert len(processes) == len(descriptions)
  jobs = []
  processes_descriptions = list(zip(processes, descriptions))
  for i, (process, description) in enumerate(processes_descriptions):
    logging.info('Starting %s', description)
    jobs.append(process.run_bg())
    if i != len(processes_descriptions) - 1:
      time.sleep(1)
  for running_jobs in more_itertools.repeatfunc(sum, None,
                                                (not j.ready() for j in jobs)):
    if running_jobs == 0:
      break
    logging.info('Still running ... {running_jobs} of {total_jobs}'.format(
        running_jobs=running_jobs, total_jobs=len(jobs)))
    time.sleep(60)
  logging.info('Done')


def main(argv):
  if FLAGS.eval:
    processes = []
    for bias in [float(e) for e in FLAGS.rbias]:
      processes.append(plumbum.local['./run.py'][
          '--eval', '--output_folder={pretrained_model_folder}-eval'.format(
              pretrained_model_folder=FLAGS.pretrained_model_folder),
          '--dataset_text_folder={dataset_text_folder}'.format(
              dataset_text_folder=FLAGS.dataset_text_folder),
          '--pretrained_model_folder={pretrained_model_folder}'.format(
              pretrained_model_folder=FLAGS.pretrained_model_folder),
          '--recall_inference_bias={bias}'.format(bias=bias),
          '--token_embedding_dimension={edim}'.format(edim=FLAGS.edim),])
    start_and_wait_for_jobs(processes, descriptions=FLAGS.rbias)
  elif FLAGS['eval-view-test-bias']:
    from plumbum.cmd import true, egrep, paste, expand, cut, sort
    cwd = plumbum.local.cwd
    command = (
        true
        | egrep['-h', 'bias|NAMED_ENTITY|Evaluate model',
                cwd // 'output/train-nov-07-eval/i2b2-2014-paper/1*/log']
        | egrep['-A1', 'bias|test']
        | egrep['NAMED|bias']
        | paste['-', '-']
        | expand['-30']
        | cut['-d:', '-f2-']
        | sort['-n'])
    print(command().rstrip())


if __name__ == '__main__':
  app.run(main)
