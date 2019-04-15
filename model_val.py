import pickle
import pandas as pd

with open('model/history.pkl', 'rb') as wd:
    history = pickle.load(wd)
    wd.close()
    hist = pd.DataFrame(history)
    end = 7
    # hist.loc[:end, ['crf_viterbi_accuracy']].plot(kind='line')
    # hist.loc[:end, ['val_crf_viterbi_accuracy']].plot(kind='line')
    hist.loc[:, ['loss']].plot(kind='line')
    hist.loc[:, ['val_loss']].plot(kind='line')
    # hist.loc[:end, ['loss', 'val_loss']].plot(kind='line')


