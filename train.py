import bilsm_crf_model
import pickle

EPOCHS = 5
model, (train_x, train_y), (test_x, test_y) = bilsm_crf_model.create_model()
# train model
history = model.fit(train_x[:1000], train_y[:1000], batch_size=16, epochs=EPOCHS, validation_data=[test_x, test_y])
model.save('model/crf.h5')

with open('model/history.pkl', 'wb') as wd:
    pickle.dump(history, wd)
    wd.close()

