from __future__ import print_function

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
import pandas as pd
import re
import json
np.random.seed(1337)
import sys

INPUT_DIRECTORY = '/Users/shahriar/Documents/Research/Code/autoencoder/data/'
MODEL_DIRECTORY = '/Users/shahriar/Documents/Research/Code/autoencoder/models/'
PHISH_INPUT_FILE = INPUT_DIRECTORY + 'phish.txt'
LEGIT_INPUT_FILE = INPUT_DIRECTORY + 'legit.txt'
TOKENIZER_FILE = MODEL_DIRECTORY + 'tokenizer.pkl'

MAX_SEQUENCE_LENGTH = 100
MAX_NB_WORDS = 1000

def read_input_emails(input_file_path):
    with open(input_file_path, "r") as file:
        email_list = json.load(file)

    return email_list

def as_df(filename, phishyness):
    all_lines = process_lines(filename)
    phishy_col = [phishyness] * len(all_lines)
    df = pd.DataFrame(list(zip(all_lines, phishy_col)), columns = ['body', 'phishy'])
    return df

def load_data(filename):
    emails = read_input_emails(filename)
    emails = [email for email in emails if email['qualify'] == True]
    
    print("Num Emails:", len(emails))
    bodies = [email['body'] for email in emails]

    return bodies

def load_embeddings(embedding_file):
    embeddings_index = {}
    f = open(embedding_file, 'r')
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except ValueError:
            continue
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index

def get_norm_embeddings(embeddings_index, word, maximum, embedding_size):
    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:
        embedding_matrix = [float(x)/maximum for x in embedding_vector] #normalize the embedding_vector
    else:
        embedding_matrix = np.random.normal(-0.25, 0.25, embedding_size)

    return embedding_matrix



def main():
    input_file = sys.argv[1]
    embedding_file = sys.argv[2]

    print("Attemping to load input")
    sequences = load_data(input_file)
    print("Data loaded")
    embeddings_index = load_embeddings(embedding_file)
    print("Embeddings loaded")

    embedding_values = embeddings_index.values()
    embedding_size = len(list(embedding_values)[0])
    maximum = 5.0408 # max(max(v) for v in embedding_values)
    print("Maximum to Norm:", maximum)

    temp = np.zeros((len(sequences) , MAX_SEQUENCE_LENGTH, embedding_size))
    for i in range(len(sequences)):
        words = sequences[i].split()
        num_words = min(len(words), MAX_SEQUENCE_LENGTH)
        for j in range(num_words):
            temp[i][j] = get_norm_embeddings(embeddings_index, words[j], maximum, embedding_size)

    sequences = temp
    print(sequences.shape)

    print("Now dumping")
    pickle.dump(sequences, open(input_file + ".pkl", "wb"))

if __name__ == '__main__':
    main()
