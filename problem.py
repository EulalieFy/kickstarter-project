


from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score

from math import sqrt

import numpy as np


def metric_report(X_test, y_true, y_pred):

    ## Regression
    print('-------- REGRESSION METRICS --------\n')
    # Root mean squared error
    print('RMSE: %.2f' %sqrt(mean_squared_error(y_true, y_pred)))

    # Mean absolute error
    print('MAE: %.2f' %mean_absolute_error(y_true, y_pred))

    # Penalized root mean squared error


    # Penalized mean absolute error


    ## Classification
    print('\n-------- CLASSIFICATION METRICS --------\n')

    y_true_binary = (X_test['goal'] < y_true).astype(np.int32)
    y_pred_binary = (X_test['goal'] < y_pred).astype(np.int32)

    # Accuracy on predicting success
    print('Accuracy: %.2f' %accuracy_score(y_true_binary, y_pred_binary))

    # Precision 
    print('Precision: %.2f' %precision_score(y_true_binary, y_pred_binary))

    # Recall 
    print('Recall: %.2f' %recall_score(y_true_binary, y_pred_binary))






def get_train_data():
    pass


def get_test_data():
    pass
