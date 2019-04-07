import pickle
import pandas as pd

with open('model/history.pkl', 'rb') as wd:
    history = pickle.load(wd)
    wd.close()
    hist = pd.DataFrame(history)
    hist.loc[:, ['crf_viterbi_accuracy', 'val_crf_viterbi_accuracy']].plot(kind='line')
