import numpy as np
import tensorflow as tf
import pickle

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

