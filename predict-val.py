from sklearn_crfsuite.metrics import flat_classification_report

import bilsm_crf_model
import process_data
import numpy as np

model, (vocab, chunk_tags) = bilsm_crf_model.create_model(train=False)
dev_x, dev_y, length = process_data.load_dev_data()
model.load_weights('model/crf.h5')

pred_y = model.predict(dev_x)
print(pred_y)
pred_id=[]
dev_id = []
for pred_one_y,one_length,y in zip(pred_y,length,dev_y):
    pred_id.append([np.argmax(x) for x in pred_one_y[-one_length:]])
    dev_id.append([yy[0] for yy in y[-one_length:]])

report = flat_classification_report(y_pred=pred_id, y_true=dev_id)

print(report)