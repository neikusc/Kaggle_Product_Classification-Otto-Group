import numpy as np
import pandas as pd
import graphlab as gl
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder


BASE_DIR = "/Users/kien/Documents/otto_group/"


class Otto_Group():
  def __init__(self):
    '''
      Parameters:
      ----------
      self.y_true: {array-like} shape {n_samples}
      self.y_pred: {array-like} shape {n_samples, n_classes}
    '''
    self.seed = 92127
    self.k_folds = 2
    self.train = pd.DataFrame()
    self.test = gl.SFrame()
    self.y_true = []
    self.y_pred = []
    self.t_samples = 0
    self.n_samples = 0
    self.n_classes = 0
  
  def logloss(self, y_true, y_pred):
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
                               for (i, j) in enumerate(self.y_true)]
    
    logloss = - np.mean(np.log(y))
    return logloss
  
  
  def load_data(self):
    # loading test set
    self.test = gl.SFrame.read_csv('test.csv')
    del self.test['id']
    self.t_samples = self.test.shape[0]
    
    # loading train set
    self.train = pd.read_csv('train.csv')
    self.train.drop('id', axis=1, inplace=True)
    # encoding labels
    self.encoder = LabelEncoder()
    self.train['target'] = self.encoder.fit_transform(self.train['target'])

    self.n_classes = len(self.encoder.classes_)
    self.y_true = [int(x) for x in self.train['target']]
    self.n_samples = self.train.shape[0]
    self.y_pred = np.zeros( (self.n_samples, self.n_classes) )
    self.y_test = np.zeros( (self.t_samples, self.n_classes) )
  
  
  def prediction(self, kwargs):
    # 10-fold cross validation for single outcome
    cv = KFold(self.n_samples, n_folds=self.k_folds, shuffle=True, random_state=self.seed)
    preds_k = np.zeros( (self.t_samples, self.n_classes, self.k_folds) )
    
    for k, (tr_idx, cv_idx) in enumerate(cv):
      x_train, x_vali = self.train.ix[tr_idx], self.train.ix[cv_idx]
      x_train.reset_index(drop=True, inplace=True)
      x_vali.reset_index(drop=True, inplace=True)
      x_train = gl.SFrame(x_train);
      x_vali = gl.SFrame( x_vali.drop('target', axis=1) )
      
      # neural networks model
      net = gl.deeplearning.create(x_train, target='target')
      net.layers[0].num_hidden_units = 100
      model = gl.neuralnet_classifier.create(x_train, target =  'target', network = net, **kwargs)

      # predicting on validation set
      preds = model.predict_topk(x_vali, output_type='score', k=self.n_classes)
      self.y_pred[cv_idx] = self.clean_proba(preds)

      # predicting on test set
      preds = model.predict_topk(self.test, output_type='score', k=9)
      preds_k[:,:,k] = self.clean_proba(preds)
      print 'Complete {}th fold.'.format(k+1)
    
    self.y_test = preds_k.mean(2)
    logloss = self.logloss(self.y_true, self.y_pred)

    print 'Model accuracy from the cross validation: {}'.format(logloss)
    return None
  
  
  def clean_proba(self, proba):
    proba = proba.unstack(['class', 'score'], 'probs').unpack('probs', '')
    proba['row_id'] = proba['row_id'].astype(int) + 1
    proba = proba.sort('row_id')
    del proba['row_id']

    proba = proba.to_dataframe().as_matrix()
    return proba
  
  
  def merge_test_pred(self):
    cv = pd.DataFrame(self.y_pred, index=np.arange(1,self.n_samples+1), columns = self.encoder.classes_ )
    cv['target'] = self.y_true
    cv.to_csv(BASE_DIR+'nn_model/nn_pred_cv.csv', index=True, index_label = 'id')

    rs = pd.DataFrame(self.y_test, index=np.arange(1,self.t_samples+1), columns = self.encoder.classes_ )
    rs.to_csv(BASE_DIR+'nn_model/resl_neuralnet_avg.csv', index=True, index_label = 'id')
    return None


params = {'max_iterations': 3,
          'metric': ['accuracy'],
          'validation_set': None}

og = Otto_Group()
og.load_data()
og.prediction(params)
#og.merge_test_pred()
