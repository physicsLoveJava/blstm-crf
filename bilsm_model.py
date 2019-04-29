from keras import Input, Model
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
        (train_x, train_y, train_max_len, train_length), (
            test_x, test_y, test_max_len, test_length), \
            (vocab, maxlen, chunk_tags, embedding_weights) = process_data.load_lstm_data()
    else:
        with open('model/chars-config.pkl', 'rb') as inp:
            (vocab, chunk_tags, embedding_weights) = pickle.load(inp)

    input = Input(shape=(train_max_len,))
    model = Embedding(len(vocab) + 1, EMBED_DIM, mask_zero=True)(input)
    model = Dropout(0.1)(model)
    model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
    model = Dropout(0.7)(model)
    out = TimeDistributed(Dense(len(chunk_tags) + 1, activation="softmax"))(model)
    model = Model(input, out)
    model.summary()
    model.compile('adam', loss="categorical_crossentropy", metrics=["accuracy"])
    if train:
        return model, (train_x, train_y, train_max_len), (test_x, test_y, test_max_len), (vocab, chunk_tags)
    else:
        return model, (vocab, chunk_tags)
