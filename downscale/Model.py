import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn import svm
import joblib
from joblib import dump, load
import pandas
import numpy
import xarray

class Model:
    
    def __init__ (self, name):
        self.stats=""
        if name == "RandomForest":
            self.model= RandomForestRegressor()
            self.name="RandomForest"
        elif name == "SupportVectorMachine":
            self.model=svm.SVR()
            self.name="SVM"
        else:
            raise TypeError("model not supported")
            
    def modify_model(self,C=None, cache_size=None, coef0=None, degree=None, epsilon=None,
                 gamma=None, kernel=None, max_iter=None,
                 shrinking=None, tol=None,
                 bootstrap=None, ccp_alpha=None, criterion=None,
                 max_depth=None,
                 max_features=None, max_leaf_nodes=None, max_samples=None,
                 min_impurity_decrease=None, min_impurity_split=None,
                 min_samples_leaf=None, min_samples_split=None,
                 min_weight_fraction_leaf=None, n_estimators=None, n_jobs=None,
                 oob_score=None, random_state=None, verbose=None,
                 warm_start=None):
        if type(self.model) == type(svm.SVR()):
            if C is not None:
                self.model.C=C
            if cache_size is not None:
                self.model.cache_size=cache_size
            if coef0 is not None:
                self.model.coef0=coef0
            if degree is not None:
                self.model.degree=degree
            if epsilon is not None:
                self.model.epsilon=epsilon
            if gamma is not None:
                self.model.gamma=gamma
            if kernel is not None:
                self.model.kernel=kernel
            if max_iter is not None:
                self.model.max_iter=max_iter
            if shrinking is not None:
                self.model.shrinking=shrinking
            if tol is not None:
                self.model.tol=tol
            if verbose is not None:
                self.model.verbose=verbose
        elif type(self.model) == type(RandomForestRegressor()):
            if bootstrap is not None:
                self.model.bootstrap=bootstrap
            if ccp_alpha is not None:
                self.model.ccp_alpha=ccp_alpha
            if criterion is not None:
                self.model.criterion=criterion
            if max_depth is not None:
                self.model.max_depth=max_depth
            if max_features is not None:
                self.model.max_features=max_features
            if max_leaf_nodes is not None:
                self.model.max_leaf_nodes=max_leaf_nodes
            if max_samples is not None:
                self.model.max_samples=max_samples
            if min_impurity_decrease is not None:
                self.model.min_impurity_decrease=min_impurity_decrease
            if min_impurity_split is not None:
                self.model.min_impurity_split=min_impurity_split
            if min_samples_leaf is not None:
                self.model.min_samples_leaf=min_samples_leaf
            if min_samples_split is not None:
                self.model.min_samples_split=min_samples_split 
            if min_weight_fraction_leaf is not None:
                self.model.min_weight_fraction_leaf=min_weight_fraction_leaf
            if n_estimators is not None:
                self.model.n_estimators=n_estimators
            if n_jobs is not None:
                self.model.n_jobs=n_jobs
            if oob_score is not None:
                self.model.oob_score=oob_score
            if random_state is not None:
                self.model.random_state=random_state
            if verbose is not None:
                self.model.verbose=verbose
            if warm_start is not None:
                self.model.warm_start=warm_start 
                
    def train(self, X_train, y_train):
        self.model.fit(X_train,y_train) 
   

    def train_test_split(self,data,features,target,train_start,train_end,test_start,test_end):
        s="lat{}_lon{}".format(data.longitude.values[0],data.latitude.values[0]).split(".")
        for string in s:
            self.name=self.name+"_"+string
        train=data.sel(time=slice(train_start,train_end))
        test=data.sel(time=slice(test_start,test_end))
        tabellatrain=(train*1.).to_dataframe().dropna()
        tabellatest=(test*1.).to_dataframe().dropna()
        del train
        del test
        X_train=tabellatrain.drop(columns=target)
        y_train=tabellatrain.drop(columns=features).to_numpy().flatten()
        X_test=tabellatest.drop(columns=target)
        y_test=tabellatest.drop(columns=features).to_numpy().flatten()
        return (X_train,y_train,X_test,y_test)
    
    def accuracy(self,X_test,y_test):
        y_pred=self.model.predict(X_test)
        if type(self.model) == type(svm.SVR()):
             s="""score :{} 
            explained variance score: {} 
            max error: {} 
            mean absolute error: {} 
            mean squared error: {} 
            median absolute error: {} 
            r2_score: {}""".format(self.model.score(X_test,y_test),explained_variance_score(y_test, y_pred),max_error(y_test, y_pred),mean_absolute_error(y_test, y_pred),
                                     mean_squared_error(y_test, y_pred),median_absolute_error(y_test, y_pred),r2_score(y_test, y_pred))
        else:
            s="""score :{} 
            explained variance score: {} 
            max error: {} 
            mean absolute error: {} 
            mean squared error: {} 
            mean squared log error: {} 
            median absolute error: {} 
            r2_score: {}""".format(self.model.score(X_test,y_test),explained_variance_score(y_test, y_pred),max_error(y_test, y_pred),mean_absolute_error(y_test, y_pred),
                                     mean_squared_error(y_test, y_pred),mean_squared_log_error(y_test, y_pred),median_absolute_error(y_test, y_pred),r2_score(y_test, y_pred))
        print(s)
        self.stats=s
        
            
    def save(self,path):
        filenamemodello=path+self.name+".joblib"
        filenametesto=path+self.name+".txt"
        dump(self.model, filenamemodello) 
        fileTesto = open(filenametesto, "w")
        fileTesto.write(self.stats)
        fileTesto.close()
            
        
            