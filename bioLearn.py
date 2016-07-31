# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 21:27:26 2016

@author: Natay
"""

import importlib
import inspect
import types
import time

import numpy as np
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV


def param_tuner(clf, score, cv, xtr, ytr):
    
    # Combine existing grids to make pipeline grid
    def make_pipeline_grid(*args, **kwargs):
        
        names = {type(x[1]).__name__:x[0] for x in args}
        pipeline_grid = {}
        
        for cl in kwargs:
            if cl in names.keys():
                for param in kwargs[cl][0]:
                    pipeline_grid[names[cl]+'__'+param] = kwargs[cl][0][param]
        
        return [pipeline_grid]

    sgd_grid = [{'loss': [ 'hinge', 'squared_hinge'],
                      'warm_start': [True],
                      'penalty': ['l2'],
                      'alpha': [1e-3],
                      'class_weight': ['balanced'],
                      'n_iter': [100],
                      'l1_ratio': np.arange(0.36, 0.9, 0.091)
                      }],'SGDClassifier'
                      
    bagging_grid = [{'n_estimators': [x for x in range(70, 100, 5)],
                         'max_samples':  np.arange(0.051, 0.1, 0.009),
                         'max_features': np.arange(0.52, 0.92, 0.08),
                         'bootstrap_features': [True],
                         'oob_score': [False]
                         }],'BaggingClassifier'
                  
    adaboost_grid = [{'n_estimators':[x for x in range(60, 90, 1)],
                      'learning_rate': [1]
                      }],'AdaBoostClassifier'
                
    svc_grid = [{'C': [1, 2, 3],
                 'degree': [1, 2, 3],
                 'gamma': ['auto','rbf', 'poly','sigmoid'],
                 'probability': [True, False],
                 'class_weight': ['balanced'],
                 }],'SVC'
                 
    passive_grid = [{'C': np.arange(0.09, 0.7, 0.1),
                     'fit_intercept': [True],
                     'n_iter': [x for x in range(5, 30, 3)],
                     'loss': ['hinge', 'squared_hinge'],
                     'warm_start': [True],
                     'class_weight': ['balanced']
                     }],'PassiveAggressiveClassifier'
    
    indiv_grids = [sgd_grid, bagging_grid, adaboost_grid, svc_grid,
                   passive_grid]
    all_grid_types = [x[1] for x in indiv_grids]+['Pipeline']

    if type(clf).__name__ in all_grid_types:         
            
        if type(clf).__name__ in all_grid_types[:-1]:
            for used in indiv_grids:
                if used[1] == type(clf).__name__:
                    estimer = GridSearchCV(clf, used[0], cv=cv,
                                       scoring =score, verbose=1)
                    estimer.fit(xtr, ytr)                     
                    return estimer.best_params_
    
        if type(clf).__name__ is 'Pipeline':
            
            grids ={x[1]:x[0] for x in indiv_grids}            
            estimer = GridSearchCV(clf, make_pipeline_grid(*clf.steps, **grids), 
                                   cv=cv, scoring =score, verbose=1)
            estimer.fit(xtr, ytr)
            clf.set_params(**estimer.best_params_)
            return clf 
    else:
        raise ValueError ('{} not an option: {}'.format(type(clf).__name__,
                                                        all_grid_types))
                                                        
# Too many scoring choices to import all and use them so
# dynamically import whats requested and return report dict. 
                                                                                                       
def report(*args,**kwargs):
    
    try:
        y_true, y_pred = kwargs['y_true'], kwargs['y_pred'] 
        full_reports= {}
        local_import = importlib.import_module('sklearn.metrics')
        
    except:
        raise KeyError('y_true and y_pred not set.')

    for report_type in args:
        if report_type in dir(local_import):
            score_type = getattr(local_import, report_type)
            
            if type(score_type) is types.FunctionType:
                if ('y_true' and 'y_pred') in inspect.getargspec(score_type).args:
                        full_reports[score_type.__name__] = score_type(y_true,
                                                                        y_pred)
                if ('y_true' and 'y_score') in inspect.getargspec(score_type).args:
                        full_reports[score_type.__name__] = score_type(y_true,
                                                                        y_score=y_pred)
        else:
            raise TypeError('{} not a scoring type'.format(report_type))
            
    return full_reports 
        
class SupervisedEstimators:
    
    def __init__(self, xtrain, ytrain, xtest, ytest):
        
        self.xtr = xtrain
        self.ytr = ytrain
        self.xte = xtest
        self.yte = ytest
            
    def reduce_dim_plot(self):
        # use PCA
        pass
    
    def classifier(self, scoring, cv, eval_using):
        
        adaclf = AdaBoostClassifier(algorithm='SAMME')
        xtr = StandardScaler().fit_transform(self.xtr)
        xte = StandardScaler().fit_transform(self.xte)
        
        # iterate over each grid score for param tuner
        for score in scoring:
            
            print('Tuning parameters of inital classifiers...')
            passive_params = param_tuner(PassiveAggressiveClassifier(), 
                                         score=score, cv=cv, xtr=xtr, 
                                         ytr=self.ytr)
            passclf = PassiveAggressiveClassifier().set_params(**passive_params)  
            sgd_params = param_tuner(SGDClassifier(), score=score, cv=cv,
                                     xtr=xtr, ytr=self.ytr)
            sgdclf = SGDClassifier().set_params(**sgd_params)
            
            # cant use resampling/bagging with passive aggressive classifier
            # will raise ValueError: The number of class labels must be > 1
            # since resampling may results in training sets with 1 class. 
            
            print('\n'+'Tuning meta-classifiers with tuned classifier/s...') 
            bagsgd_params = param_tuner(BaggingClassifier(sgdclf), 
                                         score=score, cv=cv, xtr=xtr, 
                                         ytr=self.ytr)
            bg_sgdclf = BaggingClassifier(sgdclf).set_params(**bagsgd_params)
            
            adasgd_params = param_tuner(adaclf.set_params(base_estimator=sgdclf), 
                                        score =score, cv=cv, xtr=xtr, 
                                        ytr=self.ytr)
            ada_sgdclf = adaclf.set_params(**adasgd_params)
            
            print('Voting on meta-classifiers/classifiers then predicting...')
            vote = VotingClassifier(estimators=[('BagSGD', bg_sgdclf),
                                                ('adaboostSGD', ada_sgdclf),
                                                ('Passive', passclf)],
                                    voting='hard').fit(xtr, self.ytr)
            
            print(adasgd_params)
            print(bagsgd_params)
            print(passive_params)
            
            s= time.time()
            y_true, y_pred = self.yte, vote.predict(xte)
            print('\n' + '-'*5, 'FINAL PREDICTION RESULTS','-'*5 +'\n', 
                  '{0:.4f}'.format(time.time()-s)+'--prediction time(secs)')
                  
            clf_evaluation = report(*eval_using, y_true=y_true, y_pred=y_pred)
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
    
    # local module
    import data
    
    data.options()

    hiv1 = data.load(mode='hiv-1', select_tr ='schilling', select_te='impens')
    grid_score = ['accuracy']
    used_to_eval = ['accuracy_score', 'f1_score', 'brier_score_loss', 
                    'classification_report','pairwise', 'roc_auc_score'] 
    
    # iterate over data
    for alldata in hiv1:                                                    

        trainx, trainy = alldata[0][0]
        testx, testy = alldata[0][1]
        
        print('\t\t'+'{} as training, {} as testing'.format(len(trainx),
                                                     len(testx))+'\n')
        machine = SupervisedEstimators(trainx, trainy, testx, testy)
        machine.classifier(grid_score, 3, used_to_eval)
        
        print(('-'*50) +'\n')
        
        














