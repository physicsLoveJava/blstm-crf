import pickle

from keras.layers import Embedding, Bidirectional, LSTM, Dropout, \
    Input, TimeDistributed, Conv1D, MaxPooling1D, Flatten, concatenate
from keras.models import Model
from keras_contrib.layers import CRF

import process_data

EMBED_DIM = 300
BiRNN_UNITS = 300


def create_model(train=True):
    if train:
        (train_x, chars_x, train_y, word_maxlen, char_maxlen, x_length), (
            test_x, test_chars_x, test_y, word_maxlen, char_maxlen, y_length), \
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
    char_in_1 = Input(shape=(word_len, 1,), name='char_in_1')
    embed_chars_1 = TimeDistributed(Embedding(len(chars_vocab) + 2,
                                              100, mask_zero=False, name='char_embedding_1'))(char_in_1)
    # char_enc = TimeDistributed(LSTM(units=20, return_sequences=False, recurrent_dropout=0.5))(embed_chars)
    dropout_1 = Dropout(0.5, name='char_dropout_1')(embed_chars_1)
    conv1d_out_1 = TimeDistributed(
        Conv1D(kernel_size=3, filters=100, padding='same', activation='tanh', strides=1, name='cov1d_1'))(
        dropout_1)
    maxpool_out_1 = TimeDistributed(MaxPooling1D(1, name='max_pooling_1'))(conv1d_out_1)
    char_1 = TimeDistributed(Flatten(name='flatten_1'))(maxpool_out_1)
    char_1 = Dropout(0.5)(char_1)

    # character embedding
    char_in_2 = Input(shape=(word_len, 1,), name='char_in_2')
    embed_chars_2 = TimeDistributed(Embedding(len(chars_vocab) + 2,
                                              100, mask_zero=False, name='char_embedding_2'))(char_in_2)
    # char_enc = TimeDistributed(LSTM(units=20, return_sequences=False, recurrent_dropout=0.5))(embed_chars)
    dropout_2 = Dropout(0.5, name='char_dropout_2')(embed_chars_2)
    conv1d_out_2 = TimeDistributed(
        Conv1D(kernel_size=5, filters=100, padding='same', activation='tanh', strides=1, name='cov1d_2'))(
        dropout_2)
    maxpool_out_2 = TimeDistributed(MaxPooling1D(1, name='max_pooling_2'))(conv1d_out_2)
    char_2 = TimeDistributed(Flatten(name='flatten_2'))(maxpool_out_2)
    char_2 = Dropout(0.5)(char_2)

    # character embedding
    char_in_3 = Input(shape=(word_len, 1,), name='char_in_3')
    embed_chars_3 = TimeDistributed(Embedding(len(chars_vocab) + 2,
                                              100, mask_zero=False, name='char_embedding_3'))(char_in_3)
    # char_enc = TimeDistributed(LSTM(units=20, return_sequences=False, recurrent_dropout=0.5))(embed_chars)
    dropout_3 = Dropout(0.5, name='char_dropout_3')(embed_chars_3)
    conv1d_out_3 = TimeDistributed(
        Conv1D(kernel_size=7, filters=100, padding='same', activation='tanh', strides=1, name='cov1d_3'))(
        dropout_3)
    maxpool_out_3 = TimeDistributed(MaxPooling1D(1, name='max_pooling_3'))(conv1d_out_3)
    char_3 = TimeDistributed(Flatten(name='flatten_3'))(maxpool_out_3)
    char_3 = Dropout(0.5)(char_3)

    x = concatenate([embed_words, char_1, char_2, char_3])
    x = Bidirectional(LSTM(BiRNN_UNITS // 2, recurrent_dropout=0.1, return_sequences=True, name='LSTM'))(x)
    crf = CRF(len(chunk_tags), sparse_target=True)
    out = crf(x)
    model = Model([word_in, char_in_1, char_in_2, char_in_3], out)
    model.summary()
    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
    if train:
        return model, (train_x, chars_x, train_y, word_len), (test_x, test_chars_x, test_y, y_length), (
            vocab, chunk_tags)
    else:
        return model, (vocab, chunk_tags)
