import pickle
import keras
import bilsm_crf_model

EPOCHS = 10
model, (train_x, train_y), (test_x, test_y) = bilsm_crf_model.create_model()
# train model
split = 6000
history = model.fit(train_x[:split], train_y[:split], batch_size=16, epochs=EPOCHS,
                    validation_data=[train_x[split:], train_y[split:]],
                    callbacks=[
                        # keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='auto'),
                        keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, batch_size=128)
                    ])
model.save('model/crf.h5')

with open('model/history.pkl', 'wb') as wd:
    pickle.dump(history.history, wd)
    wd.close()
