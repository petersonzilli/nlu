# -*- coding: utf-8 -*-
#!/usr/bin/env python

# baseado no original em: https://github.com/dennybritz/cnn-text-classification-tf/blob/master/eval.py

import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
import csv
import sys


import data_helpers




# Parameters
# ==================================================

#reseting parameters
import argparse as _argparse
tf.flags._global_parser = _argparse.ArgumentParser()


# Data Parameters
tf.flags.DEFINE_string("data_folder", "./data", "Data source folder.")


# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")
tf.flags.DEFINE_string("x", "", "x (default: '')")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# A variável FLAGs deve ser definida nos parâmetros de entrada do app
# O comando executado deve conter o checkpoint desejado, como no exemplo:
# python eval.py --eval_train --checkpoint_dir="./runs/1499500000/checkpoints/"
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

# Caso FLAG não seja definido inicialmente
# checkpoint_dir = último timestamp
if FLAGS.checkpoint_dir == "":
    print("Flag não declarada.")
    this_file_path = os.path.dirname(os.path.realpath(__file__))
    runs = os.listdir(os.path.join(this_file_path, "runs"))
    runs = [os.path.join(this_file_path+"\\runs", f) for f in runs]
    runs.sort(key=lambda x: os.path.getmtime(x))
    last_run = runs[-1]
    print("Usando run '"+str(last_run)+"' no lugar.")
    checkpoint_file = os.path.join(this_file_path, "runs", last_run, "checkpoints")
    print(checkpoint_file)
    FLAGS.checkpoint_dir = checkpoint_file

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    categs_raw, x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.data_folder)
    y_test = np.argmax(y_test, axis=1)
else:
    #categs_raw = ['categ1', 'categ2']
    #x_raw = ["a masterpiece four years in the making", "everything is off."]
    #y_test = [1, 0]
    categs_raw, _, _ = data_helpers.load_data_and_labels(FLAGS.data_folder)
    x_raw = [FLAGS.x]
    y_test = [0]
    
categx2int = { c : i for (i, c) in enumerate(categs_raw)}


# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

# Extract word:id mapping from the object.
vocab_dict_word_2_int = vocab_processor.vocabulary_._mapping
vocab_dict_int_2_word = {i: w for w, i in vocab_dict_word_2_int.items()}

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(zip(x_test, y_test)), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for batch in batches:
            x_test_batch, y_test_batch = zip(*batch)
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])
            
            #print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            #print(x_test_batch)
            #print(y_test_batch)
            #print(batch_predictions)
            #print(x_batch.shape)
            #print(vocab_dict_int_2_word)
            #vocab_dict_word_2_int
            for i in range(len(x_test_batch[:10])):

                frase = ""
                categoria_predicao = ""
                for w in range(x_test_batch[0].shape[0]):
                    frase += (vocab_dict_int_2_word[x_test_batch[i][w]] if vocab_dict_int_2_word[x_test_batch[i][
                        w]] != "<UNK>" else "") + " "
                categoria_original = categs_raw[y_test_batch[i]]
                if FLAGS.eval_train:
                    if categx2int[categoria_original] != batch_predictions[i]:
                        print("FALSE MATCH: {}\t -- original: {}({})\t predicao: {}({})".format(frase.encode(sys.stdout.encoding, errors='replace'),
                            categx2int[categoria_original], categoria_original, batch_predictions[i], categs_raw[batch_predictions[i]]))
                else:
                    print("PREDICTION {}\t predicao: {}({})".format(frase.encode(sys.stdout.encoding, errors='replace'),
                            batch_predictions[i], categs_raw[batch_predictions[i]]))
                

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    if FLAGS.eval_train:
        print("Accuracy: {:g}".format(correct_predictions / float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))

import codecs
with codecs.open(out_path, 'w', "utf-8") as f:
    csv.writer(f).writerows(predictions_human_readable)