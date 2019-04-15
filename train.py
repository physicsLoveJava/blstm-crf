import pickle

import keras
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

import bilsm_crf_model

EPOCHS = 10
model, (train_x, train_y), (test_x, test_y) = bilsm_crf_model.create_model()
# train model
split = 7000

# define the grid search parameters
# batch_size = [10, 20, 40, 60, 80, 100]
# epochs = [16, 32, 64, 100]
# param_grid = dict(batch_size=batch_size, nb_epoch=epochs)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
# grid_result = grid.fit(train_x[:split], train_y[:split], validation_data=[train_x[split:], train_y[split:]])
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# for params, mean_score, scores in grid_result.grid_scores_:
#     print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))

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

# plt.plot(history.history['crf_viterbi_accuracy'], 'b--')
# plt.plot(history.history['val_crf_viterbi_accuracy'], 'y-')
# plt.savefig('results/result_acc.png')
# plt.show()
