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
           
    def open_dataset(self,path):
        """reads a xarray dataset from a file given the path 
    and saves it on the instance variable ds

        Args:
            path (string): the netCDF file path
        """        
        self.ds=xr.open_mfdataset(path,combine='by_coords')
        
     
    def select_region(self, name, shapefile=None, column_name=None, scale=None):
        """selects data of a specific geographic area from a dataset

        Args:
            name (string): name of the geographic area
            shapefile (string, optional): file path to the shapefile (.shp) containing the poligon of the desired geographic area.if None is passed pandas 'naturalearth_lowres' is used. Defaults to None.
            column_name (string, optional): the name of the column containing the region name in the shapefile. Defaults to None.
            scale (int, optional): the scale applied in the shapefile. Defaults to 1.
        """        
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
        """groups by time based on frequency:

        Args:
            freq (string{"Y"=annual,"M"=montly,"W"=weekly,"D"=daily,"h"=by hour,"m"=by minute}): the freqency of the data present in the dataset
            tipo_raggruppamento (string{"min","max","sum","median","median","prod","std","var"}, optional): the value returned by the grouping. Defaults to Mean.

        """
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
            
    
    def unify_chunks(self):
        """unify the chincks of the ds attribute
        """        
        self.ds=self.ds.unify_chunks()
        
    def convert_longitude_range(self):
        """converts the latitude range from (0,360) to (-180,180)
        """        
        self.ds = self.ds.assign_coords(longitude=(((self.ds.longitude + 180) % 360) - 180)).sortby('longitude')
    
    
    def select(self,time=None,lon=None,lat=None):
        """selects the data relative to a specific time,longitude or latitude

        Args:
            time (datetime64[ns] or list or tuple, optional): the dates to select along the time dimension, it accepts a single date with dtype=datetime64[ns],a list of dates or a period rapresented by tuple of 2 elements,the first biing the start and the second the end. Defaults to None.
            lon (float or list or tuple, optional): the longitude to select along the longitude dimension, it accepts a single float value, a list of floats, or a tuple of 2 elements to select all the value from the firs elemento fo the tuple to the second. Defaults to None.
            lat (float or list or tuple, optional): the latitude to select along the latitude dimension, it accepts a single float value, a list of floats, or a tuple of 2 elements to select all the value from the firs elemento fo the tuple to the second. Defaults to None.
        """        
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
            
            
    def select_time(self,data,fineperiodo=None):
        """selects the data relative to a specific time

        Args:
            data (datetime64[ns] or list): a date or a list of dates to select, if fineperiodo is passed this is considered the first date of the period to select
            fineperiodo (datetime64[ns], optional): the last date of the period to select. Defaults to None.

        Raises:
            TypeError: _description_
        """        
        if fineperiodo is None:
            data=np.array(data,dtype='datetime64[ns]')
            data=self.time.isin(data)
            self.ds=self.ds.sel(time=data)
        elif type(data) is not list:
                self.ds=self.ds.sel(time=slice(data,fineperiodo))
        else:
            raise TypeError ("too many attributes")
       
    def select_lon(self,lon,estremolon=None):
        """selects the data relative to a specific longitude

        Args:
            lon (float or list): a float or a list of floats  to select, if estremolon is passed this is considered the first longitude of the range to select
            estremolon (float, optional): the last longitude of the range to select. Defaults to None.
        """        
        if estremolon is None:
            lon=np.array(lon,dtype='float32')
            mask=np.isin(lon,np.around(self.longitude,3))
            self.ds=self.ds.sel(longitude=lon[mask],method="nearest")
        elif type(lon) is not list:
            self.ds=self.ds.sel(longitude=slice(lon,estremolon))


    def select_lat(self,lat,estremolat=None):
        """selects the data relative to a specific latitude

        Args:
            lat (float or list): a float or a list of floats  to select, if estremolon is passed this is considered the first latitude of the range to select
            estremolat (float, optional):  the last latitude of the range to select. Defaults to None.
        """        
        if estremolat is None:
            lat=np.array(lat,dtype='float32')
            mask=np.isin(lat,np.around(self.latitude,3))
            self.ds=self.ds.sel(latitude=lat[mask],method="nearest")
        elif type(lat) is not list:
            self.ds=self.ds.sel(latitude=slice(lat,estremolat))

    

    def copy(self):
        """returns a copy of this object

        Returns:
            downscaling.Preprocessing: a copy of this object
        """        
        copia=Preprocessing()
        copia.ds=self.ds.copy()
        return copia
    
    def dropvars(self,name):
        """drops 1 or more datavars form the dataset

        Args:
            name (string): the name of the datavariable to drop
        """        
        self.ds=self.ds.dropvars(name)
    

    def combine_dataset(self,datasets):
        """recives 1 or more datasets and combines them toghether based on longitude latitude and time

        Args:
            datasets (xarray.dataset): datasets to combine

        Raises:
            TypeError: _description_
        """        
        if type(datasets) is list:
            datasets.append(self.ds)
            self.ds = xr.combine_by_coords(datasets, coords=['latitude', 'longitude', 'time'], join="inner")
        else:
            raise TypeError("datasets is not type list")
    

    def regrid_lon_lat(self,dataset):
        """recives a dataset with a lower lon/lat resolution and regridds this.ds to the same resolution 

        Args:
            dataset (xarray.dataset): dataset with lower resolution
        """        
        warnings.filterwarnings("ignore",  '.*Using.*', )
        regridder = xe.Regridder(self.ds.rename({ 'latitude': 'lat','longitude': 'lon'}), dataset.rename({'latitude': 'lat','longitude': 'lon'}),'nearest_s2d')
        self.ds= regridder(self.ds)
        self.ds= self.ds.rename({'lat' : 'latitude','lon': 'longitude'}) 


    def load(self):
        """forcefully putes the dataset in main memory
        """        
        self.ds=self.ds.load()
    
    """
    Recives the conversion rate and convert the unito of mesure of the data contained in a data variable of the dataset
    """
    def convert_unit_of_measure(self,var,conversion_value):
        """converts the unit of mesure of the data contained in a data variable of the dataset

        Args:
            var (string): the data variable to convert
            conversion_value (int or float): the value to apply
        """        
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
