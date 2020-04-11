# -*- encoding:utf-8 -*-
from __future__ import print_function

import numpy as np
import pickle
import keras
from keras.utils import *
from keras.layers import *
from keras.models import *
from keras.callbacks import ModelCheckpoint
import argparse
from keras.preprocessing.sequence import pad_sequences
import os
from attention_decoder import AttentionDecoder

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x, 2) * np.linalg.norm(y, 2))

def main():

    parser = argparse.ArgumentParser()
    parser.register('type','bool',str2bool)
    parser.add_argument('--emb_dim', type=str, default=300, help='Embeddings dimension')
    parser.add_argument('--hidden_size', type=int, default=512, help='Hidden size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=100, help='Num epochs')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
    parser.add_argument('--seq_length', type=int, default=100, help='Maximum sequence length')
    parser.add_argument('--input_data', type=str, default='data/input.pkl', help='Input data')
    parser.add_argument('--model_fname', type=str, default='models/autoencoder.h5', help='Model filename')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    args = parser.parse_args()
    print ('Model args: ', args)

    np.random.seed(args.seed)

    print("Starting...")

    print("Now building the autoencoder")
    n_features = args.emb_dim
    n_timesteps_in = args.seq_length
    n_timesteps_out = args.seq_length

    print((n_timesteps_in, n_features))

    model = Sequential()
    model.add(LSTM(args.hidden_size, input_shape=(n_timesteps_in, n_features), return_sequences=True))
    model.add(AttentionDecoder(args.hidden_size, n_features))
    model.compile(loss='mse', optimizer='adam')

    print(model.summary())

    print("Now loading data...")

    sequences = pickle.load(open(args.input_data, 'rb'))
    print('Found %s sequences.' % len(sequences))

    print("Now training the model...")

    checkpoint = ModelCheckpoint(filepath=args.model_fname, save_best_only=True)
    model.fit(sequences, sequences, epochs=args.n_epochs, verbose=2, validation_split=0.2, callbacks=[checkpoint])

    # xtest = sequences
    # ytest = model.predict(xtest)

    # cosims = np.zeros((xtest.shape[0]))

    for seq in sequences:
        seq = seq.reshape((1,seq.shape[0], seq.shape[1]))
        y = model.predict(seq)
        y = y.reshape((1, seq.shape[2], seq.shape[1]))
#        sim = cosine_similarity(seq[0], y[0])
#        print(np.mean(sim))


    # for rid in range(xtest.shape[0]):
    #     cosims[rid] = cosine_similarity(xtest[rid], ytest[rid])
    #     print(cosims[rid])


    # print("The average cosine similarities between all the sequences is ", np.mean(cosims))
    # print("The minimum cosine similarities between all the sequences is ", np.min(cosims))
    # print("The maximum cosine similarities between all the sequences is ", np.max(cosims))






if __name__ == "__main__":
    main()
