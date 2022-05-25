import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
from cartopy.util import add_cyclic_point
import cartopy.mpl.ticker as cticker


class Plotter:
    

    """
    recives a dataset with the unscaled and scaled data and plots them
    """
    def plot(self,dataset,unscaled_name, scaled_name, colormap=None):
        models=[unscaled_name,scaled_name]
        
        if dataset.latitude[0]>dataset.latitude[-1]:
            dataset=dataset.reindex(latitude=list(reversed(dataset.latitude)))
            
        # Define the figure and each axis for the 3 rows and 3 columns
        fig, axs = plt.subplots(nrows=1,ncols=2,
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        figsize=(11,8.5))

        # axs is a 2 dimensional array of `GeoAxes`.  We will flatten it into a 1-D array
        axs=axs.flatten()
        #Loop over all of the model
        
        for i,model in enumerate(models):

            # Select the week 1 forecast from the specified model
            data=dataset[model].isel(time=0)

            # Add the cyclic point
            data,lons=add_cyclic_point(data,coord=dataset['longitude'])

            # Contour plot
            if colormap is None:
                cs=axs[i].imshow(data,
                          transform = ccrs.PlateCarree(),
                          extent=(dataset.longitude[0],dataset.longitude[-1],dataset.latitude[0],dataset.latitude[-1]),
                         cmap="coolwarm")
            else:
                cs=axs[i].imshow(data,
                          transform = ccrs.PlateCarree(),
                          extent=(dataset.longitude[0],dataset.longitude[-1],dataset.latitude[-1],dataset.latitude[0]),
                         cmap=colormap)

            # Title each subplot with the name of the model
            axs[i].set_title(model)

            # Draw the coastines for each subplot
            axs[i].coastlines()
            gl=axs[i].gridlines(draw_labels=True)
            gl.ylabels_right = False
            gl.xlabels_top = False

        cbar_ax = fig.add_axes([0.2, 0.2, 0.6, 0.02])
        cbar=fig.colorbar(cs, cax=cbar_ax,orientation='horizontal')
        plt.suptitle('Before and after downscaling')

