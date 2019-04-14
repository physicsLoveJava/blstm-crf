from keras.models import Sequential, Model
from keras.layers import Embedding, Bidirectional, LSTM, Dropout, Concatenate, SpatialDropout1D, Input, TimeDistributed
from keras_contrib.layers import CRF
import process_data
import pickle

EMBED_DIM = 300
BiRNN_UNITS = 300


def create_model(train=True):
    if train:
        (train_x, chars_x, train_y), (test_x, test_chars_x, test_y), \
        (vocab, char_vocab, chunk_tags, embedding_weights) = process_data.load_cnn_data()
    else:
        with open('model/chars-config.pkl', 'rb') as inp:
            (vocab, char_vocab, chunk_tags, embedding_weights) = pickle.load(inp)
    # model = Sequential()
    word_in = Input(shape=(120,))
    # model.add(Embedding(len(vocab) + 1, EMBED_DIM, weights=[embedding_weights], mask_zero=True))  # Random embedding
    # words embedding
    embed_words = Embedding(len(vocab) + 1, EMBED_DIM, mask_zero=True)(word_in)

    # character embedding
    char_in = Input(shape=(79, 10,))
    embed_chars = TimeDistributed(Embedding(input_dim=len(char_vocab) + 2,
                                            output_dim=10, input_length=10, mask_zero=True))(char_in)
    char_enc = TimeDistributed(LSTM(units=20, return_sequences=False, recurrent_dropout=0.5))(embed_chars)

    x = Concatenate([embed_words, char_enc])
    x = TimeDistributed(SpatialDropout1D(0.3))(x)
    # model.add(Dropout(0.1))
    x = Bidirectional(LSTM(BiRNN_UNITS // 2, recurrent_dropout=0.1, return_sequences=True))(x)
    x = Dropout(0.1)(x)
    crf = CRF(len(chunk_tags), sparse_target=True)
    out = crf(x)
    model = Model([word_in, char_in], out)
    model.summary()
    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
    if train:
        return model, (train_x, chars_x, train_y), (test_x, test_y)
    else:
        return model, (vocab, chunk_tags)
