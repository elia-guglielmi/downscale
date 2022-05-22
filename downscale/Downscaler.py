import joblib
from joblib import dump, load
import sklearn
from downscale.Preprocessing import Preprocessing
import pandas
import numpy
import xarray
import Model

class Downscaler:
    
    def __init__(self,modelfile):
        self.model=load(modelfile)
        self.scaledData=None
        
    def scale(self,data,target_name):
        x=data.dataset.to_dataframe().dropna()
        y=self.model.predict(x)
        x[target_name]=y
        self.scaledData=x.to_xarray()
        
    def save(self,path):
        self.scaledData.to_netcdf(path)