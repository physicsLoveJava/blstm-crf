import pickle
import pandas as pd

# with open('model/history.pkl', 'rb') as wd:
#     history = pickle.load(wd)
#     wd.close()
#     hist = pd.DataFrame(history)
#     end = 10
#     # hist.loc[:end, ['crf_viterbi_accuracy']].plot(kind='line')
#     # hist.loc[:end, ['val_crf_viterbi_accuracy']].plot(kind='line')
#     hist.loc[:end, ['loss']].plot(kind='line')
#     hist.loc[:end, ['val_loss']].plot(kind='line')
#     # hist.loc[:end, ['loss', 'val_loss']].plot(kind='line')

with open('model/去掉O的模型参数/report-bgru-crf.pkl', 'rb') as wd:
    report = pickle.load(wd)
    print(report)

with open('model/去掉O的模型参数/report-blstm-crf.pkl', 'rb') as wd:
    report = pickle.load(wd)
    print(report)

with open('model/去掉O的模型参数/report-cnn-bigru-crf.pkl', 'rb') as wd:
    report = pickle.load(wd)
    print(report)
    
with open('model/去掉O的模型参数/report-cnn-blstm.pkl', 'rb') as wd:
    report = pickle.load(wd)
    print(report)

with open('model/去掉O的模型参数/report-cnns-blstm.pkl', 'rb') as wd:
    report = pickle.load(wd)
    print(report)


