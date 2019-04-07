import os
import pickle

import numpy
from gensim.models import KeyedVectors
from gensim.models import word2vec
from keras.preprocessing.sequence import pad_sequences

word2vec_path = 'model/word2vec.w2v'
train_path = 'data/train'
test_path = 'data/test'
dev_path = 'data/dev'
sentences_path = 'data/sentences/'
padding_letter = '<pad>'
embedding_size = 200


def transform_only_sentences(path, name):
    path = os.path.normpath(path)
    files = os.listdir(path)
    file_list = []
    for f in files:
        if os.path.isfile(os.path.join(path, f)):
            file_list.append(os.path.join(path, f))

    s_path = os.path.normpath(sentences_path)
    with open(os.path.join(s_path, name), 'wb') as wd:
        for f in file_list:
            with open(f, encoding='utf-8') as fd:
                lines = fd.readlines()
                first = [lines[i] for i in range(0, len(lines), 2)]
                for line in first:
                    wd.write(bytes(line, encoding='utf-8'))
                fd.close()
        wd.close()


def build_word2vec():
    tuples = [
        (train_path, 'train.txt'),
        (test_path, 'test.txt'),
        (dev_path, 'dev.txt'),
    ]
    for (path, name) in tuples:
        transform_only_sentences(path, name)
    sentences = word2vec.PathLineSentences(sentences_path)
    model = word2vec.Word2Vec(sentences, size=embedding_size, hs=1, min_count=5)
    print(len(model.wv.vocab))
    model.wv.add(padding_letter, numpy.zeros(model.wv.vector_size))
    print(len(model.wv.vocab))
    model.wv.save_word2vec_format(word2vec_path)
    return model.wv


def get_word2vec(rebuild=False):
    if rebuild is False and os.path.exists(word2vec_path):
        word_vec = KeyedVectors.load_word2vec_format(word2vec_path)
    else:
        word_vec = build_word2vec()
    return word_vec


def create_embedding(vocab, word_vec):
    embedding_weights = numpy.zeros((len(vocab) + 1, embedding_size))
    idx = 1
    for word in word_vec.vocab.items():
        if word in word_vec:
            embedding_weights[idx] = word_vec[word]
        else:
            embedding_weights[idx] = numpy.random.uniform(-0.25, 0.25, word_vec.vector_size)
        idx = idx + 1
    return embedding_weights


def load_data():
    word_vec = get_word2vec()
    train = _parse_data(train_path)
    test = _parse_data(test_path)

    vocab = [w for (w, i) in word_vec.vocab.items()]
    chunk_tags = list(set([row[1] for sample in train for row in sample]))

    embedding_weights = create_embedding(vocab, word_vec)

    # save initial config data
    with open('model/config.pkl', 'wb') as outp:
        pickle.dump((vocab, chunk_tags, embedding_weights), outp)

    train = _process_data(train, word_vec, vocab, chunk_tags)
    test = _process_data(test, word_vec, vocab, chunk_tags)
    return train, test, (vocab, chunk_tags, embedding_weights)


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
            idx = 0
            for line in lines:
                if idx % 2 == 0:
                    first = line.split()
                else:
                    second = line.split()
                    data.append([[first[i], second[i]] for i, w in enumerate(second)])
                idx = idx + 1
            fd.close()
    return data


def _process_data(data, word_vec, vocab, chunk_tags, maxlen=None, onehot=False):
    if maxlen is None:
        maxlen = max(len(s) for s in data)
    # idx2word = word_vec.index2word
    # word2idx = {w: i for i, w in enumerate(idx2word)}
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]  # set to <unk> (index 1) if not in sentences

    y_chunk = [[chunk_tags.index(w[1]) for w in s] for s in data]

    x = pad_sequences(x, maxlen)  # left padding

    y_chunk = pad_sequences(y_chunk, maxlen, value=-1)

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


# if __name__ == '__main__':
#     train = _parse_data(train_path)
