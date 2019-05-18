import tensorflow as tf
import numpy as np
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

def pick_word(probabilities, reverse):
    return np.random.choice(list(reverse.values()), 1, p=probabilities.flatten())[0]

gen_length = 1000
prime_words = "intro"
seq_length = 30
loaded_graph = tf.Graph()

with tf.Session(graph=loaded_graph) as sess:
    load = tf.train.import_meta_graph('output.meta')
    load.restore(sess, "output")

    input_text = loaded_graph.get_tensor_by_name('input:0')
    initial_state = loaded_graph.get_tensor_by_name('initial_state:0')
    final_state = loaded_graph.get_tensor_by_name('final_state:0')
    probs = loaded_graph.get_tensor_by_name('probs:0')

    generated_sentences = prime_words.split()
    prev_state = sess.run(initial_state, {input_text: np.array([[1 for word in generated_sentences]])})

    for n in range(gen_length):
        dyn_input = [[dictionary[word] for word in generated_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])

        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state})

        next_word = pick_word(probabilities[0][dyn_seq_length-1], reverse)
        if n % 20 == 0:
            print(n)
        generated_sentences.append(next_word)

    episode = ' '.join(generated_sentences)
    for p, replacement in tokens.items():
        episode = episode.replace(" " + replacement, p)
    print(episode)
