import pickle

import keras
import matplotlib.pyplot as plt
import numpy as np
import cnn_bilsm_crf_model
from sklearn_crfsuite.metrics import flat_classification_report

EPOCHS = 1
model, (train_x, chars_x, train_y, word_len), (test_x, test_y) = cnn_bilsm_crf_model.create_model()
# train model
split = 6000

chars_x = np.array([[[ch] for ch in s] for s in chars_x])
history = model.fit([train_x[:split], chars_x[:split]], train_y[:split], batch_size=16, epochs=EPOCHS,
                    validation_data=[[train_x[split:], chars_x[split:]], train_y[split:]],
                    callbacks=[
                        # keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='auto'),
                        keras.callbacks.TensorBoard(log_dir='./cnn-logs', histogram_freq=1, batch_size=128)
                    ])

pred_y = model.predict(test_x)
print(pred_y)
pred_id = []
dev_id = []
for pred_one_y, one_length, y in zip(pred_y, word_len, test_y):
    pred_id.append([np.argmax(x) for x in pred_one_y[-one_length:]])
    dev_id.append([yy[0] for yy in y[-one_length:]])

report = flat_classification_report(y_pred=pred_id, y_true=dev_id)

print(report)
model.save('model/crf.h5')

with open('model/report.pkl', 'wb') as wd:
    pickle.dump(report, wd)
    wd.close()
