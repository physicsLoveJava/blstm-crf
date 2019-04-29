import gensim
import numpy as np


def getmatrix(data):
    datanum = len(data)
    matrix = []
    for i in data:
        v = getvec(i)
        matrix.append(v)
    matrix = np.array(matrix)
    matrix = matrix.reshape(-1, 300, 300)
    print('matrix', matrix.shape)
    return matrix


# 对长段文本
def getvec(data):
    model = get_model()
    wvec_t = []
    for word in data:
        try:
            wvec = model[word]
        except:
            wvec = np.random.uniform(-0.25, 0.25, 300)
        wvec_t.append(wvec)
    vecdata = np.array(wvec_t)
    v = vecdata.mean(axis=0);
    return v


# 对词
def getvec_word(data):
    model = get_model()
    wvec_t = []
    for word in data:
        try:
            wvec = model[word]
        except:
            wvec = np.random.uniform(-0.25, 0.25, 300)
        wvec_t.append(wvec)
    vecdata = np.array(wvec_t)
    for i in range(1, len(vecdata) - 1):
        vecdata[i] = (vecdata[i - 1] + vecdata[i] + vecdata[i + 1]) / 3
    # vecdata=np.mean(vecdata,axis=1)
    return vecdata


def get_model():
    model = gensim.models.Word2Vec.load('D:\\pycharm\\wordvec\\zh.bin')
    return model
