import pickle

import keras
import numpy as np
from sklearn_crfsuite.metrics import flat_classification_report

import BiGRU_crf_model

EPOCHS = 10
model, (train_x, train_y, _), (test_x, test_y, length) = BiGRU_crf_model.create_model()
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
pred_y = model.predict(test_x)
print(pred_y)
pred_id = []
dev_id = []
for pred_one_y, one_length, y in zip(pred_y, length, test_y):
    pred_id.append([np.argmax(x) for x in pred_one_y[-one_length:]])
    dev_id.append([yy[0] for yy in y[-one_length:]])

report = flat_classification_report(y_pred=pred_id, y_true=dev_id)

print(report)
model.save('model/crf.h5')

with open('model/report-gru.pkl', 'wb') as wd:
    pickle.dump(report, wd)
    wd.close()

# plt.plot(history.history['crf_viterbi_accuracy'], 'b--')
# plt.plot(history.history['val_crf_viterbi_accuracy'], 'y-')
# plt.savefig('results/result_acc.png')
# plt.show()
