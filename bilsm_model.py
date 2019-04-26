from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense, TimeDistributed
from keras_contrib.layers import CRF
import keras
import process_data
import pickle

EMBED_DIM = 300
BiRNN_UNITS = 200


def create_model(train=True):
    if train:
        (train_x, train_y, train_length), (test_x, test_y, test_length), (
            vocab, chunk_tags, embedding_weights) = process_data.load_data()
    else:
        with open('model/config.pkl', 'rb') as inp:
            (vocab, chunk_tags, embedding_weights) = pickle.load(inp)
    model = Sequential()
    # model.add(Embedding(len(vocab) + 1, EMBED_DIM, weights=[embedding_weights], mask_zero=True))  # Random embedding
    model.add(Embedding(len(vocab) + 1, EMBED_DIM, mask_zero=True))  # Random embedding
    # model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, recurrent_dropout=0.2, return_sequences=True)))
    model.add(Dropout(0.7))
    model.add(TimeDistributed(Dense(len(chunk_tags) + 1, activation="softmax")))
    model.summary()
    model.compile('adam', loss="sparse_categorical_crossentropy", metrics=["acc"])
    if train:
        return model, (train_x, train_y, train_length), (test_x, test_y, test_length), (vocab, chunk_tags)
    else:
        return model, (vocab, chunk_tags)
