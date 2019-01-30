

"""
Do not use
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('data_raw/Kickstarter_light_utd.csv', index_col=0)
data.set_index(data['id'], drop=True, inplace=True)

data.drop_duplicates(['name'], inplace=True)

labels = data['achieved (%)']
labels[labels < 100] = 0
labels[(labels >= 100) & (labels < 120)] = 1
labels[labels >= 100] = 2

labels = labels.astype(np.int32)

data.drop(['usd_pledged', 'id', 'state', 'pledged', 'achieved (%)'], axis=1, inplace=True)

data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.3)

data_train.to_csv('data/data_train.csv', index=True, header=True)
data_test.to_csv('data/data_test.csv', index=True, header=True)
labels_train.to_csv('data/labels_train.csv', index=True, header=False)
labels_test.to_csv('data/labels_test.csv', index=True, header=False)