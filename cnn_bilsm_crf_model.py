import pickle

from keras.layers import Embedding, Bidirectional, LSTM, Dropout,  \
    Input, TimeDistributed, Conv1D, MaxPooling1D, Flatten, concatenate
from keras.models import Model
from keras_contrib.layers import CRF

import process_data

EMBED_DIM = 300
BiRNN_UNITS = 300


def create_model(train=True):
    if train:
        (train_x, chars_x, train_y, word_maxlen, char_maxlen), (test_x, test_chars_x, test_y, word_maxlen, char_maxlen), \
        (word_len, char_len, vocab, chars_vocab, chunk_tags, embedding_weights) = process_data.load_cnn_data()
    else:
        with open('model/chars-config.pkl', 'rb') as inp:
            (word_len, char_len, vocab, chars_vocab, chunk_tags, embedding_weights) = pickle.load(inp)
    # model = Sequential()
    word_in = Input(shape=(word_len,), name='word_in')
    # model.add(Embedding(len(vocab) + 1, EMBED_DIM, weights=[embedding_weights], mask_zero=True))  # Random embedding
    # words embedding
    embed_words = Embedding(len(vocab) + 1, EMBED_DIM, mask_zero=True, name='words_embedding')(word_in)

    # character embedding
    char_in = Input(shape=(word_len, 1,), name='char_in')
    embed_chars = TimeDistributed(Embedding(len(chars_vocab) + 2,
                                            100, mask_zero=False, name='char_embedding'))(char_in)
    # char_enc = TimeDistributed(LSTM(units=20, return_sequences=False, recurrent_dropout=0.5))(embed_chars)
    dropout = Dropout(0.5, name='char_dropout')(embed_chars)
    conv1d_out = TimeDistributed(Conv1D(kernel_size=3, filters=100, padding='same', activation='tanh', strides=1, name='cov1d'))(
        dropout)
    maxpool_out = TimeDistributed(MaxPooling1D(1, name='max_pooling'))(conv1d_out)
    char = TimeDistributed(Flatten(name='flatten'))(maxpool_out)
    char = Dropout(0.5)(char)

    x = concatenate([embed_words, char])
    x = Bidirectional(LSTM(BiRNN_UNITS // 2, recurrent_dropout=0.1, return_sequences=True, name='LSTM'))(x)
    crf = CRF(len(chunk_tags), sparse_target=True)
    out = crf(x)
    model = Model([word_in, char_in], out)
    model.summary()
    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
    if train:
        return model, (train_x, chars_x, train_y, word_len), (test_x, test_chars_x, test_y)
    else:
        return model, (vocab, chunk_tags)
