import numpy as np
import tensorflow as tf

import collections

scripts = open("data/scripts.txt", "r", encoding="utf-8")
corpus = scripts.read()

def create_tables(words):
    count = collections.Counter(words).most_common()
    dictionary = {}
    for word, k in count:
        dictionary[word] = len(dictionary) #word to key
    reverse = dict(zip(dictionary.values(), dictionary.keys())) #key to word
    return dictionary, reverse

def punctuations():
    return {
        '.': '||period||',
        ',': '||comma||',
        '"': '||quotes||',
        ';': '||semicolon||',
        '!': '||exclamation-mark||',
        '?': '||question-mark||',
        '(': '||left-parentheses||',
        ')': '||right-parentheses||',
        '--': '||emm-dash||',
        '\n': '||return||'  
    }

tokens = punctuations()
for token in tokens:
    corpus = corpus.replace(token, " " + tokens[token] + " ")
corpus = corpus.lower()
corpus = corpus.split()

dictionary, reverse = create_tables(corpus)

def make_minibatches(text, batch_size, sequence_length):
    words = batch_size * sequence_length
    num_batches = len(text) // words
    text = text[:num_batches*words]
    y = np.array(text[1:] + [text[0]])
    x = np.array(text)
    x_batches = np.split(x.reshape(batch_size, -1), num_batches, axis=1)
    y_batches = np.split(y.reshape(batch_size, -1), num_batches, axis=1)
    
    return np.array(list(zip(x_batches, y_batches)))

#Hyperparameters

epochs = 20000
batch_size = 512
rnn_size = 512
num_layers = 3
keep_prob = 0.7 #dropout rate
embed_dim = 512
sequence_length = 30
alpha = 0.001

save_dir = "./output"

training = tf.Graph()
with training.as_default():

    input_text = tf.placeholder(tf.int32, [None, None], name="input")
    targets = tf.placeholder(tf.int32, [None, None], name="targets")
    alpha = tf.placeholder(tf.float32, name='alpha')

    num_words = len(dictionary)
    input_shape = tf.shape(input_text)

    lstm = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_size)
    drop_cell = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell([drop_cell] * num_layers)

    initial_state = cell.zero_state(input_text_shape[0], tf.float32)
    initial_state = tf.identity(initial_state, name='initial_state')

    embed = tf.contrib.layers.embed_sequence(input_text, num_words, embed_dim)

    outputs, final_state = tf.nn.dynamic_rnn(cell, embed, dtype=tf.float32)
    final_state = tf.identity(final_state, name='final_state')

    logits = tf.contrib.layers.fully_connected(outputs, num_words, activation_fn=None)

    probs = tf.nn.softmax(logits, name='probs')

    cost = tf.contrib.seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_text_shape[0], input_text_shape[1]])
    )

    optimizer = tf.train.AdamOptimizer(alpha)

    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)

