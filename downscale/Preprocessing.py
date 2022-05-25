import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
import regionmask
import warnings
import xesmf as xe  


class Preprocessing:
    
    """
    creates a new isctance.
    if a path is passed reads a xarray dataset from the file 
    and saves it on the instance variable ds
    """
    def __init__(self,nomefile=None):
        if nomefile is not None:
            self.ds=xr.open_mfdataset(nomefile,combine='by_coords')
        else:    
            self.ds=None
        
    """
    reads a xarray dataset from a file given the path 
    and saves it on the instance variable ds
    """    
    def open_dataset(self,path):
        self.ds=xr.open_mfdataset(path,combine='by_coords')
        
    """
    given the name of a country selects only the data relative of that country 
    from the dataset
    """
     
    def select_region(self, name, shapefile=None, column_name=None, scale=None):
        self.convert_longitude_range()
        if shapefile is not None:
            if scale is not None:
                self.ds["longitude"]=self.ds.longitude*scale
                self.ds["latitude"]=self.ds.latitude*scale
    
            regioni=gpd.read_file(shapefile)
            region_index= regioni[regioni[column_name] == name].index
            region_geom = regioni.loc[ region_index, 'geometry']
        else: 
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            region_index= world[world.name == name].index
            region_geom = world.loc[ region_index, 'geometry']
        bounds=region_geom.total_bounds
        mask = regionmask.mask_3D_geopandas(region_geom,
                                         self.ds.longitude,
                                         self.ds.latitude,wrap_lon=False)
        if self.latitude[0]>self.latitude[-1]:
            self.ds = self.ds.sel(longitude=slice(bounds[0],bounds[2]),latitude=slice(bounds[3],bounds[1])).where(mask)
        else:
            self.ds = self.ds.sel(longitude=slice(bounds[0],bounds[2]),latitude=slice(bounds[1],bounds[3])).where(mask)
        self.ds = self.ds.squeeze(dim="region",drop=True)
    
        if scale is not None:
            self.ds["longitude"]=self.ds.longitude/scale
            self.ds["latitude"]=self.ds.latitude/scale


    
   
    """
    groups by time based on frequency:
    "Y"=annual,"M"=montly,"W"=weekly,"D"=daily,"h"=hourly,"m"=by minutes
    value of grouping="min","max","sum","median","median","prod","std","var"
    """
    def select_time_frequence(self,freq,tipo_raggruppamento=None):
        if tipo_raggruppamento is None:
            self.ds=self.ds.resample(time=freq).mean()
        elif tipo_raggruppamento =="min":
            self.ds=self.ds.resample(time=freq).min()
        elif tipo_raggruppamento =="max":
            self.ds=self.ds.resample(time=freq).max()
        elif tipo_raggruppamento =="sum":
            self.ds=self.ds.resample(time=freq).sum()
        elif tipo_raggruppamento =="median":
            self.ds=self.ds.resample(time=freq).median()
        elif tipo_raggruppamento =="prod":
            self.ds=self.ds.resample(time=freq).prod()
        elif tipo_raggruppamento =="std":
            self.ds=self.ds.resample(time=freq).std()
        elif tipo_raggruppamento =="var":
            self.ds=self.ds.resample(time=freq).var()
        else:
            raise TypeError("il tipo_raggruppamento non è valido scegliere uno tra: min,max,sum,median,prod,std,var")
            
    
    def unify_chunks(self):
        self.ds=self.ds.unify_chunks()
        
    """
    converts the latitude range from (0,360) to (-180,180)
    """   
    def convert_longitude_range(self):
        self.ds = self.ds.assign_coords(longitude=(((self.ds.longitude + 180) % 360) - 180)).sortby('longitude')
    
    
    """
    selects the data relative to a specific time,longitude or latitude

    """
    def select(self,time=None,lon=None,lat=None):
        if time is not None:
            if type(time) is tuple:
                self.select_time(time[0],time[1])
            else:
                self.select_time(time)
        if lon is not None:
            if type(lon) is tuple:
                self.select_lon(lon[0],lon[1])
            else:
                self.select_lon(lon)
            
        if lat is not None:
            if type(lat) is tuple:
                self.select_lat(lat[0],lat[1])
            else:
                self.select_lat(lat)
            
            
    """
    select data from the dataset relative to a specific date, list of dates or 
    time span(passed as a tuple of (firstdate,lastdate))
    """
    def select_time(self,data,fineperiodo=None):
        if fineperiodo is None:
            data=np.array(data,dtype='datetime64[ns]')
            data=self.time.isin(data)
            self.ds=self.ds.sel(time=data)
        elif type(data) is not list:
                self.ds=self.ds.sel(time=slice(data,fineperiodo))
        else:
            raise TypeError ("too many attributes")
    """
    select data from the dataset relative to a specific longitude, list of longitudes or 
    longitude span(passed as a tuple of (firstExtreme,lastExtreme))
    """          
    def select_lon(self,lon,estremolon=None):
        if estremolon is None:
            lon=np.array(lon,dtype='float32')
            mask=np.isin(lon,np.around(self.longitude,3))
            self.ds=self.ds.sel(longitude=lon[mask],method="nearest")
        elif type(lon) is not list:
            self.ds=self.ds.sel(longitude=slice(lon,estremolon))
    """
    select data from the dataset relative to a specific longitude, list of longitudes or 
    longitude span(passed as a tuple of (firstExtreme,lastExtreme))
    """ 
    def select_lat(self,lat,estremolat=None):
        if estremolat is None:
            lat=np.array(lat,dtype='float32')
            mask=np.isin(lat,np.around(self.latitude,3))
            self.ds=self.ds.sel(latitude=lat[mask],method="nearest")
        elif type(lat) is not list:
            self.ds=self.ds.sel(latitude=slice(lat,estremolat))

    
    
    """
    returns a copy of this object
    """
    def copy(self):
        copia=Preprocessing()
        copia.ds=self.ds.copy()
        return copia
    
    """
    drops 1 or more datavars form the dataset
    """
    def dropvars(self,name):
        self.ds=self.ds.dropvars(name)
    
    """
    recives 1 or more datasets and combines them toghether based on longitude latitude and time
    """
    def combine_dataset(self,datasets):
        if type(datasets) is list:
            datasets.append(self.ds)
            self.ds = xr.combine_by_coords(datasets, coords=['latitude', 'longitude', 'time'], join="inner")
        else:
            raise TypeError("datasets is not type list")
    
    """
    recives a dataset with a lower lon/lat resolution and upscales this.ds to the same resolution 
    """

    def regrid_lon_lat(self,dataset):
        warnings.filterwarnings("ignore",  '.*Using.*', )
        regridder = xe.Regridder(self.ds.rename({ 'latitude': 'lat','longitude': 'lon'}), dataset.rename({'latitude': 'lat','longitude': 'lon'}),'nearest_s2d')
        self.ds= regridder(self.ds)
        self.ds= self.ds.rename({'lat' : 'latitude','lon': 'longitude'}) 


    """
    forcefully putes the dataset in main memory
    """
    def load(self):
        self.ds=self.ds.load()
    
    """
    Recives the conversion rate and convert the unito of mesure of the data contained in a data variable of the dataset
    """
    def convert_unit_of_measure(self,var,conversion_value):
        self.ds[var]=self.ds[var]*conversion_value
    
    def __str__ (self):
        return repr(self.ds)
                  
    @property
    def time (self):
        return self.coordinates["time"]
                  
    @property
    def longitude (self):
        return self.coordinates["longitude"]
    
    @property
    def latitude (self):
        return self.coordinates["latitude"]
    
    @property
    def coordinates (self):
        return self.ds.coords

    @property
    def dims (self):
            return self.ds.dims
    @property
    def attributes (self):
        return self.ds.attrs

    @property
    def data_vars (self):
        return self.ds.data_vars
    
    @property
    def dataset (self):
        return self.ds
