from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dropout
from keras_contrib.layers import CRF
import process_data
import pickle

EMBED_DIM = 300
BiRNN_UNITS = 200


def create_model(train=True):
    if train:
        (train_x, train_y), (test_x, test_y), (vocab, chunk_tags, embedding_weights) = process_data.load_data()
    else:
        with open('model/config.pkl', 'rb') as inp:
            (vocab, chunk_tags, embedding_weights) = pickle.load(inp)
    model = Sequential()
    # model.add(Embedding(len(vocab) + 1, EMBED_DIM, weights=[embedding_weights], mask_zero=True))  # Random embedding
    model.add(Embedding(len(vocab) + 1, EMBED_DIM, mask_zero=True))  # Random embedding
    # model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, recurrent_dropout=0.2, return_sequences=True)))
    model.add(Dropout(0.3))
    crf = CRF(len(chunk_tags), sparse_target=True)
    model.add(crf)
    model.summary()
    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
    if train:
        return model, (train_x, train_y), (test_x, test_y)
    else:
        return model, (vocab, chunk_tags)
