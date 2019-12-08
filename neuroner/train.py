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

import codecs
import os
import pkg_resources
import pickle
import warnings
import threading
import time

from absl import logging
import numpy as np
import sklearn.metrics
import tensorflow as tf
# from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

from neuroner.evaluate import remap_labels
from neuroner import utils_tf
from neuroner import utils_nlp

def train_step(sess, dataset, sequence_number, model, parameters):
    """
    Train.
    """
    # Perform one iteration
    token_indices_sequence = dataset.token_indices['train'][sequence_number]

    for i, token_index in enumerate(token_indices_sequence):
        if token_index in dataset.infrequent_token_indices and np.random.uniform() < 0.5:
            token_indices_sequence[i] = dataset.UNK_TOKEN_INDEX

    feed_dict = {
      model.input_token_indices: token_indices_sequence,
      model.input_capitalization: model.InputCapitalization.create_columns(map(dataset.index_to_token.get, token_indices_sequence)),
      model.input_leading_spaces: dataset.leading_spaces['train'][sequence_number],
      model.input_label_indices_vector: dataset.label_vector_indices['train'][sequence_number],
      model.input_token_character_indices: dataset.character_indices_padded['train'][sequence_number],
      model.input_token_lengths: dataset.token_lengths['train'][sequence_number],
      model.input_label_indices_flat: dataset.label_indices['train'][sequence_number],
      model.dropout_keep_prob: 1-parameters['dropout_rate'],
      }

    _, _, loss, accuracy, transition_params_trained = sess.run(
                    [model.train_op, model.global_step, model.loss, model.accuracy,
                    model.transition_parameters],feed_dict)

    return transition_params_trained

def prediction_step(sess, dataset, dataset_type, model, transition_params_trained,
    stats_graph_folder, epoch_number, parameters, dataset_filepaths):
    """
    Predict.
    """
    if dataset_type == 'deploy':
        print('Predict labels for the {0} set'.format(dataset_type))
    else:
        print('Evaluate model on the {0} set'.format(dataset_type))

    all_predictions = []
    all_y_true = []
    output_filepath = os.path.join(stats_graph_folder, '{1:03d}_{0}.txt'.format(dataset_type,
        epoch_number))
    output_file = codecs.open(output_filepath, 'w', 'UTF-8')
    original_conll_file = codecs.open(dataset_filepaths[dataset_type], 'r', 'UTF-8')

    res = [None] * len(dataset.token_indices[dataset_type])

    def step(start, end):
        for i in range(start, end):
            token_indices_sequence = dataset.token_indices[dataset_type][i]
            feed_dict = {
              model.input_token_indices: token_indices_sequence,
              model.input_leading_spaces: dataset.leading_spaces[dataset_type][i],
              model.input_capitalization: model.InputCapitalization.create_columns(map(dataset.index_to_token.get, token_indices_sequence)),
              model.input_token_character_indices: dataset.character_indices_padded[dataset_type][i],
              model.input_token_lengths: dataset.token_lengths[dataset_type][i],
              model.input_label_indices_vector: dataset.label_vector_indices[dataset_type][i],
              model.dropout_keep_prob: 1.,
              model.recall_inference_bias: parameters['recall_inference_bias'] if not parameters['train_model'] else 0.0,
            }
            res[i] = unary_scores, predictions = sess.run([model.unary_scores,
                model.predictions], feed_dict)
            logging.log_every_n_seconds(logging.INFO, 'Predict... run model [{dataset}] {percent}%'.format(dataset=dataset_type ,percent=round(100*i/len(dataset.token_indices[dataset_type]), ndigits=1)), n_seconds=10)

    chunk_size = min(1, int(len(res) / parameters['number_of_cpu_threads_prediction']))
    threads = [threading.Thread(target=step, args=(i, min(len(res), i+chunk_size))) for i in range(0, len(res), chunk_size)]
    for t in threads:
        while True:
            try:
                t.start()
            except RuntimeError as e:
                logging.warning('RuntimeError: can\'t start new thread ######## threading.active_count=%d'% (threading.active_count(),) )
                time.sleep(3)
                continue
            else:
                break
    for t in threads:
        t.join()
        del t
    del threads

    for i in range(len(dataset.token_indices[dataset_type])):
        unary_scores, predictions = res[i]

        if parameters['use_crf']:
            predictions, _ = tf.contrib.crf.viterbi_decode(unary_scores,
                transition_params_trained)
            predictions = predictions[1:-1]
        else:
            predictions = predictions.tolist()

        assert(len(predictions) == len(dataset.tokens[dataset_type][i]))

        output_string = ''
        prediction_labels = [dataset.index_to_label[prediction] for prediction in predictions]
        unary_score_list = unary_scores.tolist()[1:-1]

        gold_labels = dataset.labels[dataset_type][i]

        if parameters['tagging_format'] == 'bioes':
            prediction_labels = utils_nlp.bioes_to_bio(prediction_labels)
            gold_labels = utils_nlp.bioes_to_bio(gold_labels)

        for prediction, token, gold_label, scores in zip(prediction_labels,
            dataset.tokens[dataset_type][i], gold_labels, unary_score_list):

            while True:
                line = original_conll_file.readline()
                split_line = line.strip().split(' ')

                if '-DOCSTART-' in split_line[0] or len(split_line) == 0 \
                or len(split_line[0]) == 0:
                    continue
                else:
                    token_original = split_line[0]

                    if parameters['tagging_format'] == 'bioes':
                        split_line.pop()

                    gold_label_original = split_line[-1]

                    assert(token == token_original and gold_label == gold_label_original)
                    break

            split_line.append(prediction)
            try:
                if parameters['output_scores']:
                    # space separated scores
                    scores = ' '.join([str(i) for i in scores])
                    split_line.append('{}'.format(scores))
            except KeyError:
                pass

            output_string += ' '.join(split_line) + '\n'

        output_file.write(output_string+'\n')

        all_predictions.extend(predictions)
        all_y_true.extend(dataset.label_indices[dataset_type][i])
        logging.log_every_n_seconds(logging.INFO, 'Predict... eval [{dataset}] {percent}%'.format(dataset=dataset_type ,percent=round(100*i/len(dataset.token_indices[dataset_type]), ndigits=1)), n_seconds=10)

    output_file.close()
    original_conll_file.close()

    if dataset_type != 'deploy':

        if parameters['main_evaluation_mode'] == 'conll':

            # run perl evaluation script in python package
            # conll_evaluation_script = os.path.join('.', 'conlleval')
            package_name = 'neuroner'
            root_dir = os.path.dirname(pkg_resources.resource_filename(package_name,
                '__init__.py'))
            conll_evaluation_script = os.path.join(root_dir, 'conlleval')

            conll_output_filepath = '{0}_conll_evaluation.txt'.format(output_filepath)
            shell_command = 'perl {0} < {1} > {2}'.format(conll_evaluation_script,
                output_filepath, conll_output_filepath)
            os.system(shell_command)

            with open(conll_output_filepath, 'r') as f:
                classification_report = f.read()
                print(classification_report)

        else:
            new_y_pred, new_y_true, new_label_indices, new_label_names, _, _ = remap_labels(all_predictions,
                all_y_true, dataset, parameters['main_evaluation_mode'])

            print(sklearn.metrics.classification_report(new_y_true, new_y_pred, 
                digits=4, labels=new_label_indices, target_names=new_label_names))

    return all_predictions, all_y_true, output_filepath


def predict_labels(sess, model, transition_params_trained, parameters, dataset,
    epoch_number, stats_graph_folder, dataset_filepaths):
    """
    Predict labels using trained model
    """
    seconds = int(time.time())
    y_pred = {}
    y_true = {}
    output_filepaths = {}

    for dataset_type in ['train', 'valid', 'test', 'deploy']:
        if dataset_type not in dataset_filepaths.keys():
            continue

        prediction_output = prediction_step(sess, dataset, dataset_type, model,
            transition_params_trained, stats_graph_folder, epoch_number,
            parameters, dataset_filepaths)
        y_pred[dataset_type], y_true[dataset_type], output_filepaths[dataset_type] = prediction_output

    seconds = int(time.time()) - seconds
    logging.info('Eval took %d seconds', seconds)

    return y_pred, y_true, output_filepaths
