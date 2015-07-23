import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler




from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet




K_FOLDS = 10
SEED = 92126

def logloss(y_true, y_pred):
  '''
    Parameters:
    ----------
      y_true: {array-like} shape {n_samples}
      y_pred: {array-like} shape {n_samples, n_classes}

    Return: logloss
    ------
  '''
  epsilon=1e-15
    
  # get probabilities
  y = [np.maximum(epsilon, np.minimum(1-epsilon, y_pred[i, j])) 
                             for (i, j) in enumerate(y_true)]

  logloss = - np.mean(np.log(y))
  return logloss

def load_train_data(path):
  df = pd.read_csv(path)
  X = df.values.copy()
  np.random.shuffle(X)
  ids, X, labels = X[:, 0], X[:, 1:-1].astype(np.float32), X[:, -1]
  encoder = LabelEncoder()
  y = encoder.fit_transform(labels).astype(np.int32)
  scaler = StandardScaler()
  X = scaler.fit_transform(X)
  return X, y, encoder, scaler, ids

def load_test_data(path, scaler):
  df = pd.read_csv(path)
  X = df.values.copy()
  X, idx = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
  X = scaler.transform(X)
  return X, idx

def cross_validation(nnet, X, y, X_test):
  
  cv = KFold(num_samples, n_folds=K_FOLDS, shuffle=True, random_state=SEED)
  y_pred = np.zeros( (num_samples, num_classes) )
  preds_k = np.zeros( (num_samples_test, num_classes, K_FOLDS) )
  
  for k, (tr_idx, cv_idx) in enumerate(cv):
    f = open('net0.pickle', 'rb')
    # new classifier for each fold
    clf = pickle.load(f)
    x_train, x_vali = X[tr_idx], X[cv_idx]
    y_train = y[tr_idx]
    
    # predicting on validation set
    clf.fit( x_train, y_train )
    y_pred[cv_idx, :] = clf.predict_proba( x_vali )
      
    # predicting on test set
    preds_k[:,:,k] = clf.predict_proba( X_test )
    print 'Complete {}th fold'.format(k+1)
    f.close()
    
  preds = preds_k.mean(2)
  
  return y_pred, preds




X, y, encoder, scaler, ids = load_train_data('train.csv')
X_test, idx = load_test_data('test.csv', scaler)

num_classes = len(encoder.classes_)
num_samples, num_features = X.shape
num_samples_test = X_test.shape[0]

layers0 = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout', DropoutLayer),
           ('dense1', DenseLayer),
           ('output', DenseLayer)]
#"""
epoch0 = 500

net0 = NeuralNet(layers=layers0,
                 
                 input_shape=(None, num_features),
                 dense0_num_units=300,
                 dropout_p=0.5,
                 dense1_num_units=300,
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 
                 update=nesterov_momentum,
                 update_learning_rate=0.001,
                 update_momentum=0.9,
                 
                 eval_size=0.2,
                 verbose=1,
                 max_epochs=epoch0)
#"""

import cPickle as pickle
with open('net0.pickle', 'wb') as f:
  pickle.dump(net0, f, -1)




def make_submission(y, encoder, y_pred, preds):
  cv = pd.DataFrame(y_pred, index=ids, columns = encoder.classes_ )
  cv['target'] = y
  cv.sort_index(axis=0, ascending=True, inplace=True)
  cv.to_csv('nn_model/pred_cv.csv', index=True, index_label='id')
  
  rs = pd.DataFrame(preds, index=idx, #np.arange(1,num_samples_test+1),
                    columns=encoder.classes_)
  
  rs = rs.div(rs.sum(axis=1), axis=0)
  rs.to_csv( 'nn_model/nn_submission.csv', index=True, index_label='id' )

  print("Wrote submission to file")
  print 'Accuracy of cross validation: {}'.format(logloss(y, y_pred))
  return None




y_pred, preds = cross_validation(net0, X, y, X_test)
make_submission(y, encoder, y_pred, preds)



