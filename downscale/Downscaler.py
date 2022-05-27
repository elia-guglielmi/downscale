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
        
    def read_model(self,modelfile):
        """reads the sklearn estimator from file

        Args:
            modelfile (string): name of the .joblib file to read
        """        
        self.model=load(modelfile)
    

    def scale(self,data,target_name):
        """use the estimator saved in the model attribute to apply a machine learning algorithm and make a prediction on a dataset

        Args:
            data (xarray.dataset): dataset (with 3 dimenstions:longitude,latitude,time, with data relative to only 1 longitude and 1 latitude) containing the data to downscale
            target_name (string): the name of the datavariable where to put the scaled data

        Returns:
            xarray.dataset: new dataset with an extra datavariable containing the old data and the scaled data
        """        
        x=data.dataset.to_dataframe().dropna()
        y=self.model.predict(x)
        x[target_name]=y
        scaledData=x.to_xarray()
        return scaledData
   
    """
    recives a xarray dataset splits it based on year and month, assigns to them a name and saves them of disk
    """    
        
    def save_by_month(self,data,path):
        """splits a dataset based on year and month, assigns to them a name and saves them of disk

        Args:
            data (xarray.dataset): dataset to save on disk
            path (string): the path where to save the netCDF file
        """        
        dsets1=list(data.groupby("time.year"))
        for y in dsets1:
            dsets2=list(y[1].groupby("time.month"))
            for x in dsets2:
                nomefile="{}_{}.nc".format(y[0],x[0])
                x[1].to_netcdf(path+nomefile)
           
    def save(self,data,path):
        """recives a xarray dataset saves it of disk

        Args:
            data (xarray.dataset): dataset to save on disk
            path (string): the path where to save the netCDF file
        """        
        data.to_netcdf(path)