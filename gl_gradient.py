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
    self.seed = 92127 # 2015
    self.k_folds = 2
    self.train = pd.DataFrame()
    self.test = gl.SFrame()
    self.y_true = []
    self.y_pred = []
    self.n_samples = 0
    self.t_samples = 0
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
    self.t_samples = self.test.shape[0]
    
    # loading train set
    self.train = pd.read_csv('train.csv')
    self.train.drop('id', axis=1, inplace=True)
    # encoding labels
    self.encoder = LabelEncoder()
    self.train['target'] = self.encoder.fit_transform(self.train['target'])
    
    self.n_classes = len(self.encoder.classes_)
    self.y_true = self.train['target'].astype(np.int32)
    self.n_samples = len(self.y_true)
    self.y_pred = np.zeros( (self.n_samples, self.n_classes) )
    self.y_test = np.zeros( (self.t_samples, self.n_classes) )
  
  def prediction(self, kwargs):
    
    # 10-fold cross validation for single outcome
    cv = KFold(self.n_samples, n_folds=self.k_folds, shuffle=True, random_state=self.seed)
    preds_k = np.zeros( (self.t_samples, self.n_classes, self.k_folds) )
    
    for k, (tr_idx, cv_idx) in enumerate(cv):
      x_train, x_vali = self.train.ix[tr_idx], self.train.ix[cv_idx]
      x_train = gl.SFrame(x_train);
      x_vali = gl.SFrame( x_vali.drop('target', axis=1) )
      
      # training a boosted trees model
      model = gl.boosted_trees_classifier.create(x_train, **kwargs)
      preds = model.predict_topk(x_vali, output_type='probability', k=self.n_classes)
      self.y_pred[cv_idx] = self.clean_proba(preds)

      # predicting on test set
      preds = model.predict_topk(self.test, output_type='probability', k=self.n_classes)
      preds_k[:,:,k] = self.clean_proba(preds)
      print 'Complete {}th fold.'.format(k+1)
    
    self.y_test = preds_k.mean(2)
    
    logloss = self.logloss(self.y_true, self.y_pred)
    print logloss
    return None
  
  def clean_proba(self, proba):
    proba = proba.unstack(['class', 'probability'], 'probs').unpack('probs', '')
    proba['id'] = proba['id'].astype(int) + 1
    proba = proba.sort('id')
    del proba['id']
    
    proba = proba.to_dataframe().as_matrix()
    return proba
  
  def save_results(self):
    cv = pd.DataFrame(self.y_pred, index=np.arange(1,self.n_samples+1), columns = self.encoder.classes_)
    cv['target'] = self.y_true
    cv.to_csv(BASE_DIR+'gb_model/gb_pred_cv.csv', index=True, index_label = 'id')
    
    rs = pd.DataFrame(self.y_test, index=np.arange(1,self.t_samples+1), columns = self.encoder.classes_ )
    rs.to_csv(BASE_DIR+'gb_model/resl_graphlab_avg.csv', index=True, index_label = 'id')
    return None



params = {'target': 'target',
          'max_iterations': 10,
          'max_depth': 10,
          'min_child_weight': 4,
          'row_subsample': .9,
          'min_loss_reduction': 1,
          'column_subsample': .8,
          'validation_set': None}

og = Otto_Group()
og.load_data()
og.prediction(params)
og.save_results()
