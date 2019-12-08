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

import os
from absl import app

from neuroner import prepare_pretrained_model


def share(dir_name, epoch, delete_token_mappings=True):
  prepare_pretrained_model.prepare_pretrained_model_for_restoring(
      output_folder_name=dir_name,
      epoch_number=epoch,
      model_name='{base}--epoch-{epoch}{emb}'.format(
          base=os.path.basename(dir_name),
          epoch=epoch,
          emb='' if delete_token_mappings else '-emb'),
      delete_token_mappings=delete_token_mappings)


def main(argv):
  pass
  # EXAMPLE: share('my train dir', my_epoch)


if __name__ == '__main__':
  app.run(main)
