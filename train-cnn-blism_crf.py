import pickle

import keras
import matplotlib.pyplot as plt

import cnn_bilsm_crf_model

EPOCHS = 1
model, (train_x, chars_x, train_y), (test_x, test_y) = cnn_bilsm_crf_model.create_model()
# train model
split = 6000
history = model.fit(train_x[:split], train_y[:split], batch_size=16, epochs=EPOCHS,
                    validation_data=[train_x[split:], train_y[split:]],
                    callbacks=[
                        # keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='auto'),
                        keras.callbacks.TensorBoard(log_dir='./cnn-logs', histogram_freq=1, batch_size=128)
                    ])
model.save('model/cnn-crf.h5')

with open('model/cnn-history.pkl', 'wb') as wd:
    pickle.dump(history.history, wd)
    wd.close()

plt.plot(history.history['crf_viterbi_accuracy'], 'b--')
plt.plot(history.history['val_crf_viterbi_accuracy'], 'y-')
plt.savefig('results/result_acc.png')
plt.show()
