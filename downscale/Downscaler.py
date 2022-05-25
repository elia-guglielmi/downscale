import joblib
from joblib import dump, load
import sklearn
import pandas
import numpy
import xarray

class Downscaler:
    
    """
    creates the object and reads the sklearn estimator from file if passed
    """
    def __init__(self,modelfile=None):
        if modelfile is not None:
            self.model=load(modelfile)
        else:
            self.model=None
        
    """
     reads the sklearn estimator from file
    """ 
    def read_model(self,modelfile):
        self.model=load(modelfile)
    

    """
     recives a xarray dataset saves it in a class attribute(scaledData), 
     use the estimator to apply a machine learning algorithm and make a prediction using the dataset, 
     returns a new dataset with an extra datavariable containing the scaled data
    """ 
    def scale(self,data,target_name):
        x=data.dataset.to_dataframe().dropna()
        y=self.model.predict(x)
        x[target_name]=y
        scaledData=x.to_xarray()
        return scaledData
   
    """
    recives a xarray dataset splits it based on year and month, assigns to them a name and saves them of disk
    """    
        
    def save_by_month(self,data,path):
        dsets1=list(data.groupby("time.year"))
        for y in dsets1:
            dsets2=list(y[1].groupby("time.month"))
            for x in dsets2:
                nomefile="{}_{}.nc".format(y[0],x[0])
                x[1].to_netcdf(path+nomefile)

    """
    recives a xarray dataset saves it of disk
    """              
    def save(self,data,path):
        data.to_netcdf(path)