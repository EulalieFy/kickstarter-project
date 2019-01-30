from __future__ import division, print_function

############# Packages #############

import os
import numpy as np
from math import sqrt

import warnings
import pandas as pd
import itertools

from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, confusion_matrix

############# RAMP Packages #############

import rampwf as rw
#from rampwf.score_types.base import BaseScoreType
from rampwf.score_types.classifier_base import ClassifierBaseScoreType
from rampwf.workflows.feature_extractor import FeatureExtractor
from rampwf.workflows.classifier import Classifier
warnings.filterwarnings("ignore")

####################################################

problem_title = 'KickStarter FundRaising  Classification'

_prediction_label_names = [0, 1, 2]

# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)


# An object implementing the workflow
class FeatureExtractorClassifier(object):
    """
    Difference with the FeatureExtractorClassifier from ramp-workflow:
    `test_submission` wraps the y_proba in a DataFrame with the original
    index.
    """

    def __init__(self):
        self.element_names = ['feature_extractor', 'classifier']
        self.feature_extractor_workflow = FeatureExtractor(
            [self.element_names[0]])
        self.classifier_workflow = Classifier([self.element_names[1]])

    def train_submission(self, module_path, X_df, y_array, train_is=None):
        if train_is is None:
            train_is = slice(None, None, None)
        fe = self.feature_extractor_workflow.train_submission(
            module_path, X_df, y_array, train_is)
        X_train_array = self.feature_extractor_workflow.test_submission(
            fe, X_df.iloc[train_is])
        clf = self.classifier_workflow.train_submission(
            module_path, X_train_array, y_array[train_is])
        return fe, clf

    def test_submission(self, trained_model, X_df):
        print('TEST SUBMIT')
        fe, clf = trained_model
        X_test_array = self.feature_extractor_workflow.test_submission(
            fe, X_df)
        y_proba = self.classifier_workflow.test_submission(clf, X_test_array)
        
        #arr = X_df.index.values.astype('datetime64[m]').astype(int)
        #y = np.hstack((arr[:, np.newaxis], y_proba))
        return y_proba


workflow = FeatureExtractorClassifier()
    
  
class Precision(ClassifierBaseScoreType):

    def __init__(self, label,name='precision', precision=2):
        self.name = name + '_' + label
        self.precision = precision
        self.label = label

    def __call__(self, y_true, y_pred):
        return precision_score(y_true, y_pred, average=None)[self.label]
    
class Recall(ClassifierBaseScoreType):

    def __init__(self, label, name='recall', precision=2):
        self.name = name + '_' + label
        self.precision = precision
        self.label = label

    def __call__(self, y_true, y_pred):
        recall = recall_score(y_true, y_pred, average=None)[self.label]
        return recall

class Shortfall(ClassifierBaseScoreType):

    def __init__(self, name='shortfall', precision=5):
        self.name = name 
        self.precision = precision

    def __call__(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        #i (true), j (pred)
        type1 = cm[0,1] +cm[0,2]
        type2_s = cm[1,0] 
        type2_hs = cm[2,0]
        type3 = cm[1,2] 
        type4 = cm[1,2]
        
        loss = 0*type1 + 0.05*0.2* type2_s +\
            0.05*0.4*type2_hs + 0.05*type3 +\
             0.05*0.2*type4
        
        return loss

    
# Scores Types to print

score_types = [
        
    #Recall(name='mae', precision=2)
    #Precision(name='mae', precision=2)
    #F1_score w_FScore(),
    Recall(label='Failure'),
    Recall(label='Success'),
    Recall(label='Higher Sucess'),
    Precision(label='Failure'),
    Precision(label='Success'),
    Precision(label='Higher Success'),
    Shortfall()

]
        
############# Cross Validation function used in the workflow #############
def get_cv(X, y):
    # using 5 folds as default
    k = 5
    # up to 10 fold cross-validation based on 5 splits, using two parts for
    # testing in each fold
    n_splits = 5
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(cv.split(X, y))
    
    # get all combinaison of folds
    combinaison = list(itertools.permutations([i for i in range(n_splits)]))
    combinaison = [tuple(sorted(l[:int(n_splits*0.667)])+sorted(l[int(n_splits*0.667):])) for l in combinaison]
    combinaison = list(set(combinaison))
    combinaison = [(l[:int(n_splits*0.667)], l[int(n_splits*0.667):]) for l in combinaison]
    for ps in combinaison[:k]:
         
        print(y[np.hstack([splits[p][1] for p in ps[0]])].shape)
        print(y[np.hstack([splits[p][1] for p in ps[1]])].shape)

        yield (np.hstack([splits[p][1] for p in ps[0]]),
               np.hstack([splits[p][1] for p in ps[1]]))


############# Load and Read the Data #############

def _read_data(path, type_, threshold = 1000):

    ''' 
        From path leading to the data, return a dataset of features X with 
    it corresponding label vector y. Using a Threshold value, one can
    limit itself to a smaller number of rows to do some quick tests 
    '''
    
    assert type_ in ['test', 'train']
    
    X = pd.read_csv(os.path.join(path, 'data_{}.csv'.format(type_)))
    y = pd.read_csv(os.path.join(path, 'labels_{}.csv'.format(type_)), header=None)
    
    X.drop('id', axis=1, inplace =True)
    y.drop(0,axis=1,  inplace=True)
    test = os.getenv('RAMP_TEST_MODE', 0)
    
    y = np.array(y).reshape((y.shape[0]))
    
    if test:
        X = X[:threshold]
        y = y[:threshold]
    return X, y

# Return Train dataa and labels
def get_train_data(path='./data'):
    #path='./data'
    return _read_data(path, 'train')

# Return Test data and labels
def get_test_data(path='./data'):
    #path='./data'
    return _read_data(path, 'test')
