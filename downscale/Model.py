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
from abc import ABC,abstractmethod

class Model:


    """
    class that creates a scikit-learn estimator and provides methods to recive divide and use a xarray dataset to train evaluate and save the estimator
    """
    def __init__ (self,name,estimator):
        self.stats=""
        self.name=name
        self.model=estimator

      
    @abstractmethod       
    def modify_model(self):
        pass


    def train(self, X_train, y_train):
        """trains the estimator

        Args:
            X_train (pandas.dataframe): data to use as features
            y_train (numpy.array): data to use as target
        """        
        self.model.fit(X_train,y_train) 
   
 
    def train_test_split(self,data,features,target,train_start,train_end,test_start,test_end):
        """splits the dataset in train and test based on date, and splits both the train and test in
        features and target

        Args:
            data (xarray.dataset): dataset with dimensions longitude,latitude and time, with data relative to only 1 longitude and 1 latitude
            features (string): the names of the datavariables used as features
            target (string): the name of the datavariable used as target
            train_start (datetime64[ns]): first date of the training set
            train_end (datetime64[ns]): last date of the training set
            test_start (datetime64[ns]): first date of the test set
            test_end (datetime64[ns]): last date of the test set

        Returns:
            (pandas.dataframe,numpy.array,pandas.dataframe,numpy.array):  pandas dataframe with the features and a numpy array with the target respecctively for the training and test sets
        """        
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
        y_train=tabellatrain[target].to_numpy()
        X_test=tabellatest.drop(columns=target)
        y_test=tabellatest[target].to_numpy()
        return (X_train,y_train,X_test,y_test)
    

    @abstractmethod
    def accuracy(self):
        pass
       
                   
    def save(self,path):
        """saves the estimator in a file Model.name.joblib and the relative accuracy stats in a file Model.name.txt

        Args:
            path (string): he path where to save the file on disk
        """        
        filenamemodello=path+self.name+".joblib"
        filenametesto=path+self.name+".txt"
        dump(self.model, filenamemodello) 
        fileTesto = open(filenametesto, "w")
        fileTesto.write(self.stats)
        fileTesto.close()
            
        
class RandomForestModel(Model):
    
    def __init__ (self):
        super().__init__("RandomForest",RandomForestRegressor())
        
    def modify_model(self, bootstrap=None, ccp_alpha=None, criterion=None,
                 max_depth=None,
                 max_features=None, max_leaf_nodes=None, max_samples=None,
                 min_impurity_decrease=None, min_impurity_split=None,
                 min_samples_leaf=None, min_samples_split=None,
                 min_weight_fraction_leaf=None, n_estimators=None, n_jobs=None,
                 oob_score=None, random_state=None, verbose=None,
                 warm_start=None):
        """set some features to the estimator

        Args:
            bootstrap (bool, optional): Whether bootstrap samples are used when building trees. If False, the
                                        whole dataset is used to build each tree. Defaults to True.
            ccp_alpha (non-negative float, optional):  Complexity parameter used for Minimal Cost-Complexity Pruning. Defaults to 0.0.
            criterion (string, optional):  {"squared_error", "absolute_error", "poisson"},The function to measure the quality of a split. Defaults to "squared_error".
            max_depth (int, optional):  The maximum depth of the tree. If None, then nodes are expanded until
                                        all leaves are pure or until all leaves contain less than
                                        min_samples_split samples. Defaults to None.
            max_features (int or float, optional):  The number of features to consider when looking for the best split:. Defaults to 1.0.
            max_leaf_nodes (int, optional): Grow trees with ``max_leaf_nodes`` in best-first fashion.
                                            Best nodes are defined as relative reduction in impurity.
                                            If None then unlimited number of leaf nodes.. Defaults to None.
            max_samples (int or float, optional):   If bootstrap is True, the number of samples to draw from X
                                                    to train each base estimator. Defaults to None.
            min_impurity_decrease (_type_, optional): _description_. Defaults to None.
            min_impurity_split (float, optional):A node will be split if this split induces a decrease of the impurity
                                                greater than or equal to this value.. Defaults to 0.
            min_samples_leaf (int or float, optional): The minimum number of samples required to be at a leaf node. Defaults to 1.
            min_samples_split (int or float, optional): The minimum number of samples required to split an internal node. Defaults to 2.
            min_weight_fraction_leaf (float, optional): The minimum weighted fraction of the sum total of weights (of all
                                                        the input samples) required to be at a leaf node. Samples have
                                                        equal weight when sample_weight is not provided.. Defaults to 0.0.
            n_estimators (int, optional): The number of trees in the forest. Defaults to 100.
            n_jobs (int, optional): The number of jobs to run in parallel. Defaults to None.
            oob_score (bool, optional): Whether to use out-of-bag samples to estimate the generalization score. Defaults to False.
            random_state (int, optional):  Controls both the randomness of the bootstrapping of the samples used
                                        when building trees (if ``bootstrap=True``) and the sampling of the
                                        features to consider when looking for the best split at each node. Defaults to None.
            verbose (int, optional): Controls the verbosity when fitting and predicting. Defaults to 0.
            warm_start (bool, optional): When set to ``True``, reuse the solution of the previous call to fit
                                        and add more estimators to the ensemble, otherwise, just fit a whole
                                        new forest.. Defaults to False.
        """
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
            
        
    def accuracy(self,X_test,y_test):
        """makes prediction and confront them to the recived numpyarray to calculate the score,explained variance, max error,mean absolute error, mean squared error, median absolute error and r2 score and saves them as a string in the class attribute Model.stats

        Args:
            X_test (pandas.dataframe): Pandas dataframe used to make predictions
            y_test (numpy.array): numpy array used to evaluate the accuracy of the predictions
        """        
        y_pred=self.model.predict(X_test)
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
        

        
        
        
class SVMModel(Model):
    
    def __init__ (self):
        super().__init__("SVM",svm.SVR())
        
    def modify_model(self,C=None, cache_size=None, coef0=None, degree=None, epsilon=None,
                 gamma=None, kernel=None, max_iter=None,
                 shrinking=None, tol=None,verbose=None):
        """set some features to the estimator

        Args:
            C (float, optional):  Regularization parameter. The strength of the regularization is
                                inversely proportional to C. Must be strictly positive.. Defaults to 1.
            cache_size (float, optional): Specify the size of the kernel cache (in MB).. Defaults to 200.
            coef0 (float, optional):  Independent term in kernel function. Defaults to 1.0.
            degree (int, optional):  Degree of the polynomial kernel function ('poly'). Defaults to 3.
            epsilon (float, optional):  Epsilon in the epsilon-SVR model. Defaults to 0.1.
            gamma ({'scale', 'auto'} or float, optional): Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. Defaults to 'scale'.
            kernel (string, optional):  Specifies the kernel type to be used in the algorithm. Defaults to rbf.
            max_iter (int, optional):  Hard limit on iterations within solver, or -1 for no limit. Defaults to -1.
            shrinking (bool, optional):  Whether to use the shrinking heuristic.. Defaults to True.
            tol (float, optional): Tolerance for stopping criterion. Defaults to 1e-3.
            verbose (int, optional): Controls the verbosity when fitting and predicting. Defaults to 0.
        """
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
            
        
    def accuracy(self,X_test,y_test):
        """makes prediction and confront them to the recived numpyarray to calculate the score,explained variance, max error,mean absolute error, mean squared error, median absolute error and r2 score and saves them as a string in the class attribute Model.stats

        Args:
            X_test (pandas.dataframe): Pandas dataframe used to make predictions
            y_test (numpy.array): numpy array used to evaluate the accuracy of the predictions
        """        
        y_pred=self.model.predict(X_test)
        s="""score :{} 
            explained variance score: {} 
            max error: {} 
            mean absolute error: {} 
            mean squared error: {} 
            median absolute error: {} 
            r2_score: {}""".format(self.model.score(X_test,y_test),explained_variance_score(y_test, y_pred),max_error(y_test, y_pred),mean_absolute_error(y_test, y_pred),
                                     mean_squared_error(y_test, y_pred),median_absolute_error(y_test, y_pred),r2_score(y_test, y_pred))
        print(s)
        self.stats=s            