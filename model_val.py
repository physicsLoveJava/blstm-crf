import pickle

import matplotlib.pyplot as plt
import pandas as pd

with open('model/history.pkl', 'rb') as wd:
    history = pickle.load(wd)
    wd.close()
    hist = pd.DataFrame(history)
    plt.style.use("ggplot")
    plt.figure(figsize=(12, 12))
    plt.plot(hist["acc"])
    plt.plot(hist["val_acc"])
    plt.show()
