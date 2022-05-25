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


    """
    class that creates a scikit-learn estimator and provides methods to recive divide and use a xarray dataset to train evaluate and save the estimator
    """
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


    """
    set some features to the estimator
    """        
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
                self.model.cache_size=cache_size                                #Specify the size of the kernel cache (in MB).
            if coef0 is not None:
                self.model.coef0=coef0                                          #Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
            if degree is not None:
                self.model.degree=degree                                        #Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
            if epsilon is not None:
                self.model.epsilon=epsilon                                      #Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value.
            if gamma is not None:
                self.model.gamma=gamma                                          #Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma, if ‘auto’, uses 1 / n_features.
            if kernel is not None:
                self.model.kernel=kernel                                        #Specifies the kernel type to be used in the algorithm. If none is given, ‘rbf’ will be used. If a callable is given it is used to precompute the kernel matrix.
            if max_iter is not None:
                self.model.max_iter=max_iter                                    #Hard limit on iterations within solver, or -1 for no limit.
            if shrinking is not None:
                self.model.shrinking=shrinking                                  #Whether to use the shrinking heuristic.
            if tol is not None:
                self.model.tol=tol                                              #Tolerance for stopping criterion.
            if verbose is not None:
                self.model.verbose=verbose                                      
        elif type(self.model) == type(RandomForestRegressor()):
            if bootstrap is not None:
                self.model.bootstrap=bootstrap                                  #Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.
            if ccp_alpha is not None:
                self.model.ccp_alpha=ccp_alpha                                  #Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen. By default, no pruning is performed
            if criterion is not None:
                self.model.criterion=criterion                                  #The function to measure the quality of a split. Supported criteria are “squared_error” for the mean squared error, which is equal to variance reduction as feature selection criterion, “absolute_error” for the mean absolute error, and “poisson” which uses reduction in Poisson deviance to find splits. Training using “absolute_error” is significantly slower than when using “squared_error”.
            if max_depth is not None:
                self.model.max_depth=max_depth                                  #The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples
            if max_features is not None:
                self.model.max_features=max_features                            #he number of features to consider when looking for the best split: If int, then consider max_features features at each split. If float, then max_features is a fraction and round(max_features * n_features) features are considered at each split. If “auto”, then max_features=n_features. If “sqrt”, then max_features=sqrt(n_features). If “log2”, then max_features=log2(n_features). If None or 1.0, then max_features=n_features.
            if max_leaf_nodes is not None:
                self.model.max_leaf_nodes=max_leaf_nodes                        #Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
            if max_samples is not None:
                self.model.max_samples=max_samples                              #If bootstrap is True, the number of samples to draw from X to train each base estimator. If None (default), then draw X.shape[0] samples. If int, then draw max_samples samples. If float, then draw max_samples * X.shape[0] samples. Thus, max_samples should be in the interval (0.0, 1.0].
            if min_impurity_decrease is not None:
                self.model.min_impurity_decrease=min_impurity_decrease          #A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
            if min_impurity_split is not None:
                self.model.min_impurity_split=min_impurity_split                
            if min_samples_leaf is not None:
                self.model.min_samples_leaf=min_samples_leaf                    #The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.
            if min_samples_split is not None:
                self.model.min_samples_split=min_samples_split                  #The minimum number of samples required to split an internal node.
            if min_weight_fraction_leaf is not None:
                self.model.min_weight_fraction_leaf=min_weight_fraction_leaf    #The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.
            if n_estimators is not None:                                        
                self.model.n_estimators=n_estimators                            #The number of trees in the forest.
            if n_jobs is not None:
                self.model.n_jobs=n_jobs                                        #The number of jobs to run in parallel. fit, predict, decision_path and apply are all parallelized over the trees. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.
            if oob_score is not None:
                self.model.oob_score=oob_score                                  #Whether to use out-of-bag samples to estimate the generalization score. Only available if bootstrap=True.
            if random_state is not None:
                self.model.random_state=random_state                            #Controls both the randomness of the bootstrapping of the samples used when building trees (if bootstrap=True) and the sampling of the features to consider when looking for the best split at each node (if max_features is less n_features)
            if verbose is not None:
                self.model.verbose=verbose                                      #Controls the verbosity when fitting and predicting
            if warm_start is not None:
                self.model.warm_start=warm_start                                # When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest.

    """
    recives a pandas dataframe with the features and a numpy array target and trains the estimator
    """   

    def train(self, X_train, y_train):
        self.model.fit(X_train,y_train) 
   
    """
    recives a xarray dataset, the name of the features, the name of the target, 
    the first date of the training set,the last date of the training set,the first date of the test set and the last date of the test set.
    Generates the name of the model based on the longitude and latitude of the dataset, selects the training set
    from the dataset and returns a pandas dataframe with the features and a numpy array with the target respecctively for the training and test sets
    """   
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
        y_train=tabellatrain[target].to_numpy()
        X_test=tabellatest.drop(columns=target)
        y_test=tabellatest[target].to_numpy()
        return (X_train,y_train,X_test,y_test)
    
    
    """
    recives pandas dataframe, uses it to make prediction and confront them to the recived numpyarray to calculate the score,
    explained variance, max error,mean absolute error, mean squared error, median absolute error and r2 score
    and saves them as a string in the class attribute Model.stats
    """     
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
        
    """
    saves the estimator in a file Model.name.joblib and the relative accuracy stats in a file Model.name.txt
    """             
    def save(self,path):
        filenamemodello=path+self.name+".joblib"
        filenametesto=path+self.name+".txt"
        dump(self.model, filenamemodello) 
        fileTesto = open(filenametesto, "w")
        fileTesto.write(self.stats)
        fileTesto.close()
            
        
            