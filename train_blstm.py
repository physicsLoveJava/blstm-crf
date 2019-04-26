import pickle

import keras
import numpy as np
from sklearn_crfsuite.metrics import flat_classification_report

import bilsm_model
import process_data

EPOCHS = 10
model, (train_x, train_y, _), (test_x, test_y, length), (vocab, chunk_tags) = bilsm_model.create_model()
dev_x, dev_y, dev_length = process_data.load_data(use_dev=True)
# train model
# split = 7000

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

history = model.fit(train_x, train_y, batch_size=16, epochs=EPOCHS,
                    validation_data=[test_x, test_y],
                    callbacks=[
                        # keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='auto'),
                        keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, batch_size=128)
                    ])

pred_y = model.predict(dev_x)
pred_id = []
dev_id = []
for pred_one_y, one_length, y in zip(pred_y, dev_length, dev_y):
    pred_id.append([np.argmax(x) for x in pred_one_y[-one_length:]])
    dev_id.append([yy[0] for yy in y[-one_length:]])

labels, tag_names = process_data.get_labels_tags(chunk_tags)

report = flat_classification_report(y_pred=pred_id, y_true=dev_id, labels=labels, target_names=tag_names)

print(report)
model.save('model/crf.h5')

with open('model/report-blstm-crf.pkl', 'wb') as wd:
    pickle.dump(report, wd)
    wd.close()

# plt.plot(history.history['crf_viterbi_accuracy'], 'b--')
# plt.plot(history.history['val_crf_viterbi_accuracy'], 'y-')
# plt.savefig('results/result_acc.png')
# plt.show()
