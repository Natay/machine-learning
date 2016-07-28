# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 21:27:26 2016

@author: Natay
"""

import importlib
import inspect
import types
import time

from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV


def param_tuner(clf, score, cv, xtr, ytr):

    def make_pipeline_grid(*args, **kwargs):
        
        names = {type(x[1]).__name__:x[0] for x in args}
        pipeline_grid = {}
        
        for cl in kwargs:
            if cl in names.keys():
                for param in kwargs[cl][0]:
                    pipeline_grid[names[cl]+'__'+param] = kwargs[cl][0][param]
        
        return [pipeline_grid]
    
    nys_grid = [{'n_components': [2500],
                     'gamma': [1e-1],
                     'degree': [1, 2],
                     'coef0': [1, 2]}],'Nystroem'
                     
    sgd_grid = [{'loss': [ 'hinge', 'squared_hinge'],
                      'warm_start': [True, False],
                      'penalty': ['l2'],
                      'alpha': [1e-3],
                      'class_weight': ['balanced'],
                      'n_iter': [100]}],'SGDClassifier'
                      
    bagging_grid = [{'n_estimators': [150, 180],
                         'max_samples': [0.61, 0.67],
                         'max_features': [0.72, 0.83],
                         'bootstrap_features': [True],
                         'oob_score': [False]}],'BaggingClassifier'
    # for future use                    
    adaboost_grid = [{'n_estimators':[50, 100, 150],
                      'learning_rate': [1]
                      }],'AdaBoostClassifier'
    # for future use                 
    linsvc_grid = [{'C': [1, 2, 5, 10],
                    'max_iter': [1000, 5000, 6000]}],'LinearSVC'
    
    indiv_grids = [nys_grid, sgd_grid, bagging_grid, adaboost_grid, linsvc_grid]
    all_grid_types = [x[1] for x in indiv_grids]+['Pipeline']

    if type(clf).__name__ in all_grid_types:         
            
        if type(clf).__name__ in all_grid_types[:-1]:
            for used in indiv_grids:
                if used[1] == type(clf).__name__:
                    estimer = GridSearchCV(clf, used[0], cv=cv,
                                       scoring =score, verbose=1, n_jobs=2)
                    estimer.fit(xtr, ytr)                     
                    return estimer.best_params_
    
        if type(clf).__name__ is 'Pipeline':
            
            grids ={x[1]:x[0] for x in indiv_grids}            
            estimer = GridSearchCV(clf, make_pipeline_grid(*clf.steps, **grids), 
                                   cv=cv, scoring =score, verbose=1, n_jobs=2)
            estimer.fit(xtr, ytr)
            clf.set_params(**estimer.best_params_)
            return clf 
    else:
        raise ValueError ('{} not an option: {}'.format(type(clf).__name__,
                                                        all_grid_types))

def report(*args,**kwargs):
    
    try:
        y_true, y_pred = kwargs['y_true'], kwargs['y_pred']  
    except:
        raise KeyError('{} not param options.'.format(kwargs.keys()))
    full_reports= {}
    local_import = importlib.import_module('sklearn.metrics')

    for report_type in args:
        if report_type in dir(local_import):
            score_type = getattr(local_import, report_type)
            if type(score_type) is types.FunctionType:
                if ('y_true' and 'y_pred') in inspect.getargspec(score_type).args:
                        full_reports[score_type.__name__] = score_type(y_true,
                                                                        y_pred)
        else:
            raise TypeError('{} not a scoring type'.format(report_type))
            
    return full_reports 
        
class SupervisedEstimators:
    
    def __init__(self, xtrain, ytrain, xtest, ytest):
        
        self.xtr = xtrain
        self.ytr = ytrain
        self.xte = xtest
        self.yte = ytest
        
    def reduce_dim(self):
        # use PCA
        pass
                
    def classifier(self, scoring, cv, eval_using):
        
        pipe = Pipeline([('feature_map', Nystroem()),
                         ('clf', SGDClassifier())])
                         

        feature_used = type(pipe.named_steps['feature_map']).__name__
        clf_used = type(pipe.named_steps['clf']).__name__
         
        for score in scoring:
            
            print('Tuning parameters of Pipline...')
            tuned_pipe = param_tuner(pipe, score=score,
                                   cv=cv, xtr=self.xtr, ytr=self.ytr)
    
            print('\n'+'Transforming X with {} for {}...'.format(feature_used, 
                                                              clf_used))
            X_feat = tuned_pipe.named_steps['feature_map'].fit_transform(self.xtr)
            sgd_clf = tuned_pipe.named_steps['clf'].fit(X_feat, self.ytr)
 
            print('Fitting bagging classifer with SGDclf then predicting...') 
            bgclf = BaggingClassifier(sgd_clf)
            bagging_params = param_tuner(bgclf, score=score,
                                         cv=cv, xtr=self.xtr, ytr=self.ytr)  
            
            bgclf.set_params(**bagging_params)            
            bgclf = bgclf.fit(self.xtr,self.ytr)
            print(bgclf.get_params())
            s= time.time()
            y_true, y_pred = self.yte, bgclf.predict(self.xte)
            print('\n'+'{0:.4f}'.format(time.time()-s)+'--prediction time')
            
            print('\n' + '-'*5, 'FINAL PREDICTION RESULTS','-'*5)
            clf_evaluation = report(*eval_using, 
                                    y_true=y_true, y_pred=y_pred)
            for reports in clf_evaluation:
                print('---',reports)
                print(clf_evaluation[reports])


    def regression(self):
        pass
    def neural_net(self):
        # use rbm or theano net made in other module
        pass


class UnsupervisedEstimators:
    
    def __init__(self):
        pass
    def mean_shift_clustering(self):
        pass
    def kmeans_clustering(self):
        pass


                
if __name__ == '__main__':
    
    # local imports
    import data
    import warnings
    
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    data.options()

    hiv1 = data.load(mode='hiv-1', select_tr ='schilling', select_te='impens')
    scores_to_max = ['accuracy']
     
    
    for alldata in hiv1:                                                    

        trainx, trainy = alldata[0][0]
        testx, testy = alldata[0][1]
        
        print('\t\t'+'{} as training, {} as testing'.format(len(trainx),
                                                     len(testx))+'\n')
        used_to_eval = ['accuracy_score', 'f1_score', 'brier_score_loss', 
                        'classification_report','pairwise']                                         
        
        machine = SupervisedEstimators(trainx, trainy, testx, testy)
        machine.classifier(scores_to_max, 3, used_to_eval)
        
        print(('-'*50) +'\n')
        














