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

base = "dropout"
# base = "25个参数"
# base = "25个参数dropout0.7"
# base = "20个参数"
# base = "20个参数dropout0.7"
# base = "15个参数"
# base = "15个参数dropout0.7"
# base = "10个参数"
# base = "10个参数dropout0.7"


def parse_report():
    kv = {
        'report-cnn-blstm-0.3.pkl': '0.3',
        'report-cnn-blstm-0.4.pkl': '0.4',
        'report-cnn-blstm-0.5.pkl': '0.5',
        'report-cnn-blstm-0.6.pkl': '0.6',
        'report-cnn-blstm-0.7.pkl': '0.7',
        'report-cnn-blstm-0.8.pkl': '0.8',
        'report-cnn-blstm-0.9.pkl': '0.9',
    }
    for i in range(3, 10):
        k = 'report-cnn-blstm-0.%s.pkl' % i
        with open('model/%s/%s' % (base, k), 'rb') as wd:
            report = pickle.load(wd)
            print(report)
            print(i)


parse_report()


