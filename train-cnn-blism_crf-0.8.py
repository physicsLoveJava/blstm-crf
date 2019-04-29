import pickle

import keras
import numpy as np
from sklearn_crfsuite.metrics import flat_classification_report
import process_data
import cnn_bilsm_crf_model_dp

EPOCHS = 10
model, (train_x, chars_x, train_y, word_len), (
    test_x, test_chars_x, test_y, length), (vocab, chunk_tags) = cnn_bilsm_crf_model_dp.create_model(0.8)
dev_x, dev_chars_x, dev_y, _, _, dev_length = process_data.load_cnn_data(use_dev=True)
# train model
# split = 7000

chars_x = np.array([[[ch] for ch in s] for s in chars_x])
test_chars_x = np.array([[[ch] for ch in s] for s in test_chars_x])
dev_chars_x = np.array([[[ch] for ch in s] for s in dev_chars_x])

#
# train_x = train_x[:100]
# chars_x = chars_x[:100]
# train_y = train_y[:100]
# test_x = test_x[:100]
# test_chars_x = test_chars_x[:100]
# test_y = test_y[:100]


history = model.fit([train_x, chars_x], train_y, batch_size=16, epochs=EPOCHS,
                    validation_data=[[test_x, test_chars_x], test_y],
                    callbacks=[
                        # keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='auto'),
                        keras.callbacks.TensorBoard(log_dir='./cnn-logs', histogram_freq=1, batch_size=128)
                    ])

# predict

pred_y = model.predict([dev_x, dev_chars_x])
pred_id = []
dev_id = []
for pred_one_y, one_length, y in zip(pred_y, length, dev_y):
    pred_id.append([np.argmax(x) for x in pred_one_y[-one_length:]])
    dev_id.append([yy[0] for yy in y[-one_length:]])

labels, tag_names = process_data.get_labels_tags(chunk_tags)

report = flat_classification_report(y_pred=pred_id, y_true=dev_id, labels=labels, target_names=tag_names, digits=4)

print(report)
model.save('model/crf.h5')

with open('model/report-cnn-blstm-0.8.pkl', 'wb') as wd:
    pickle.dump(report, wd)
    wd.close()

