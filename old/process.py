import os
import numpy as np
from sklearn.linear_model import Perceptron, SGDClassifier

import vector_helpe
import sklearn_crfsuite
from sklearn.metrics import classification_report
from sklearn import svm, neighbors, naive_bayes, tree, metrics

permit_tag = [
    "B_Bear",
    "B_Time",
    "B_Name",
    "I_Time",
    "I_Document",
    "B_Know",
    "B_Gender",
    "B_Remarry",
    "I_Name",
    "B_Separation",
    "I_Court",
    "I_Judgment",
    "I_Duration",
    "B_Age",
    "B_Court",
    "I_Marry",
    "I_Age",
    "I_Price",
    "B_Price",
    "B_BeInLove",
    "B_Marry",
]


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
                    if has_pertmit_tag(second):
                        data.append([[first[i], second[i]] for i, w in enumerate(second)])
                idx = idx + 1
            fd.close()
    np.random.shuffle(data)
    return data


def has_pertmit_tag(second):
    for w in second:
        if w in permit_tag:
            return True
    return False


def get_data(filename):
    dataset = []
    data = np.array(_parse_data(filename))
    for sent in data:
        for j in sent:
            dataset.append(j);
    dataset = np.array(dataset)
    label = dataset[:, 1]
    data = dataset[:, 0]
    data = vector_helpe.getvec_word(data);
    return data, label


def crfmodel(train_filename, test_filename, dev_filename):
    train_data, train_label = get_data(train_filename)
    dev_data, dev_label = get_data(dev_filename)
    test_data, test_label = get_data(test_filename)
    print('-数据读取完成')
    # model=sklearn_crfsuite.CRF(algorithm='l2sgd');
    model = svm.SVC()
    model = SGDClassifier()
    model.fit(train_data, train_label)
    y_pre = model.predict(test_data)
    t = list(set(test_label))
    print(classification_report(test_label, y_pre, labels=t))


if __name__ == '__main__':
    crfmodel('./data/train', './data/test', './data/dev')
