import xgboost as xgb
import numpy as np

from sklearn.base import BaseEstimator

class Classifier(BaseEstimator):
    
    def __init__(self):
        self.model = xgb.XGBClassifier(booster = 'gbtree',
                                       objective = 'multi:softmax',
                                       colsample_bytree = 0.9, 
                                       learning_rate = 0.1,
                                       max_depth = 5, 
                                       alpha =10,
                                       n_estimators = 50)

    def fit(self, X, y):
        print('FIT', X.shape, y.shape)
        self.model.fit(X, y)

    def predict(self, X):
        prediction = self.model.predict(X)
        return prediction
    
    def predict_proba(self,X):
        prediction = self.model.predict(X)
        probas = np.zeros((X.shape[0],3))
        for ind,c in enumerate(prediction) : 
            probas[ind,c]=1
        return probas