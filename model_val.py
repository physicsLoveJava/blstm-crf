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

# base = "去掉O的模型参数"
# base = "20个参数"
base = "15个参数"


def parse_report():
    kv = {
        'report-bgru-crf.pkl': 'bigru-crf',
        'report-blstm-crf.pkl': 'blstm-crf',
        'report-cnn-bigru-crf.pkl': 'cnn-bigru-crf',
        'report-cnn-blstm.pkl': 'cnn-blstm-crf',
        'report-cnns-blstm.pkl': 'cnns-blstm-crf',
    }
    for k, v in kv.items():
        with open('model/%s/%s' % (base, k), 'rb') as wd:
            report = pickle.load(wd)
            print(v)
            print(report)
            print(v)


parse_report()


