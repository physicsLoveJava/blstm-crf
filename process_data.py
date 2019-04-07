import os
import pickle
from collections import Counter

import numpy
from keras.preprocessing.sequence import pad_sequences


def load_data():
    train = _parse_data('data/train')
    test = _parse_data('data/test')

    word_counts = Counter(row[0].lower() for sample in train for row in sample)
    vocab = [w for w, f in iter(word_counts.items()) if f >= 2]
    chunk_tags = list(set([row[1] for sample in train for row in sample]))

    # save initial config data
    with open('model/config.pkl', 'wb') as outp:
        pickle.dump((vocab, chunk_tags), outp)

    train = _process_data(train, vocab, chunk_tags)
    test = _process_data(test, vocab, chunk_tags)
    return train, test, (vocab, chunk_tags)


def _parse_data(path):
    #  in windows the new line is '\r\n\r\n' the space is '\r\n' . so if you use windows system,
    #  you have to use recorsponding instructions
    path = os.path.normpath(path)
    files = os.listdir(path)
    file_list = []
    data = []
    for f in files:
        if os.path.isfile(os.path.join(path, f)):
            file_list.append(os.path.join(path, f))

    for f in file_list:
        with open(f, encoding='utf-8') as fd:
            lines = fd.readlines()
            first = [lines[i].split() for i in range(0, len(lines), 2)]
            second = [lines[i].split() for i in range(1, len(lines), 2)]
            data.append([[p, q] for i in first for j in second for (p, q) in zip(i, j)])
            fd.close()
    return data


def _process_data(data, vocab, chunk_tags, maxlen=None, onehot=False):
    if maxlen is None:
        maxlen = max(len(s) for s in data)
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]  # set to <unk> (index 1) if not in vocab

    y_chunk = [[chunk_tags.index(w[1]) for w in s] for s in data]

    x = pad_sequences(x, maxlen, padding='post')  # left padding

    y_chunk = pad_sequences(y_chunk, maxlen, padding='post', value=-1)

    if onehot:
        y_chunk = numpy.eye(len(chunk_tags), dtype='float32')[y_chunk]
    else:
        y_chunk = numpy.expand_dims(y_chunk, 2)
    return x, y_chunk


def process_data(data, vocab, maxlen=100):
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [word2idx.get(w[0].lower(), 1) for w in data]
    length = len(x)
    x = pad_sequences([x], maxlen)  # left padding
    return x, length

#
# if __name__ == '__main__':
#     # print(_parse_data('./data/train'))
#     load_data()
