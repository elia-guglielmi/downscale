import unittest
import numpy as np
import pandas as pd
import xarray as xr
from downscale.Preprocessing import Preprocessing

class TestPreprocessing(unittest.TestCase):
    
    def setUp(self):
        data =  np.arange(64, dtype=np.int64).reshape(4, 4,4)
        lon = ([1., 2., 3.,4.])
        lon=np.array(lon,dtype='float32')
        lat = ([1.,2., 3., 4.])
        lat=np.array(lat,dtype='float32')
        times = pd.date_range("2000-01-01", periods=4)
        x = xr.Dataset(
        data_vars=dict(
            foo=(["time", "latitude", "longitude"], data),
            ),
            coords=dict(
            time=times,
            latitude=("latitude",lat),
            longitude=("longitude",lon),
            ),
        )
        y = xr.Dataset(
        data_vars=dict(
            boo=(["time", "latitude", "longitude"], data),
            ),
            coords=dict(
            time=times,
            latitude=("latitude",lat),
            longitude=("longitude",([350.,351., 352., 353.])),
            ),
        )
        data3 =  np.arange(256, dtype=np.int64).reshape(4, 8,8)
        lon3 = ([1.,1.5, 2.,2.5, 3.,3.5,4.,4.5])
        lon3=np.array(lon3,dtype='float32')
        lat3 = ([1.,1.5, 2.,2.5, 3.,3.5,4.,4.5])
        lat3=np.array(lat3,dtype='float32')
        z = xr.Dataset(
        data_vars=dict(
            coo=(["time", "latitude", "longitude"], data3),
            ),
            coords=dict(
            time=times,
            latitude=("latitude",lat3),
            longitude=("longitude",lon3),
            ),
        )
        self.classedati1=Preprocessing()
        self.classedati1.ds=x
        self.classedati2=Preprocessing()
        self.classedati2.ds=y
        self.classedati3=Preprocessing()
        self.classedati3.ds=z
        self.x=x
        
    def tearDown(self):
        self.classedati1.ds=self.x
        
    def test_select_time_frequence(self):
        self.classedati1.select_time_frequence("MS")
        np.testing.assert_array_equal(self.classedati1.time,np.array(["2000-01-01"],dtype='datetime64[ns]'))
        self.assertEqual(self.classedati1.ds.foo[0,0,0].values,24)
        
        self.classedati1.ds=self.x
        
        self.classedati1.select_time_frequence("MS","min")
        np.testing.assert_array_equal(self.classedati1.time,np.array(["2000-01-01"],dtype='datetime64[ns]'))
        self.assertEqual(self.classedati1.ds.foo[0,0,0].values,0)
        
        self.classedati1.ds=self.x
        
        self.classedati1.select_time_frequence("MS","max")
        np.testing.assert_array_equal(self.classedati1.time,np.array(["2000-01-01"],dtype='datetime64[ns]'))
        self.assertEqual(self.classedati1.ds.foo[0,0,0].values,48)
        
        self.classedati1.ds=self.x
        
        self.classedati1.select_time_frequence("MS","sum")
        np.testing.assert_array_equal(self.classedati1.time,np.array(["2000-01-01"],dtype='datetime64[ns]'))
        self.assertEqual(self.classedati1.ds.foo[0,0,0].values,96)
        
        
    def test_select_time(self):
        self.classedati1.select_time("2000-01-01","2000-01-03")
        np.testing.assert_array_equal(self.classedati1.time,np.array(["2000-01-01","2000-01-02","2000-01-03"],dtype='datetime64[ns]'))
        self.classedati1.select_time(["2000-01-01","2000-01-02"])
        np.testing.assert_array_equal(self.classedati1.time,np.array(["2000-01-01","2000-01-02"],dtype='datetime64[ns]'))
        self.classedati1.select_time(["2000-01-01","2000-01-02","2001-01-01"])
        np.testing.assert_array_equal(self.classedati1.time,np.array(["2000-01-01","2000-01-02"],dtype='datetime64[ns]'))
        self.classedati1.select_time("2000-01-01")
        self.assertEqual(self.classedati1.time,np.array(["2000-01-01"],dtype='datetime64[ns]'))
        
        
    def test_select_lon(self):
        self.classedati1.select_lon(1,3)
        np.testing.assert_array_equal(self.classedati1.longitude,np.array([1.,2.,3.],dtype='float32'))
        self.classedati1.select_lon([1.,2.])
        np.testing.assert_array_equal(self.classedati1.longitude,np.array([1.,2.],dtype='float32'))
        self.classedati1.select_lon([1.,2.,5.])
        np.testing.assert_array_equal(self.classedati1.longitude,np.array([1.,2.],dtype='float32'))
        self.classedati1.select_lon(1.)
        self.assertEqual(self.classedati1.longitude,np.array([1.],dtype='float32'))
        
    def test_select_lat(self):
        self.classedati1.select_lat(1,3)
        np.testing.assert_array_equal(self.classedati1.latitude,np.array([1.,2.,3.],dtype='float32'))
        self.classedati1.select_lat([1.,2.])
        np.testing.assert_array_equal(self.classedati1.latitude,np.array([1.,2.],dtype='float32'))
        self.classedati1.select_lat([1.,2.,5.])
        np.testing.assert_array_equal(self.classedati1.latitude,np.array([1.,2.],dtype='float32'))
        self.classedati1.select_lat(1.)
        self.assertEqual(self.classedati1.latitude,np.array([1.],dtype='float32'))
        
    def test_copy(self):
        copia=self.classedati1.copy()
        np.testing.assert_array_equal(self.classedati1.latitude,copia.latitude)
        np.testing.assert_array_equal(self.classedati1.longitude,copia.longitude)
        np.testing.assert_array_equal(self.classedati1.time,copia.time)
        np.testing.assert_array_equal(self.classedati1.data_vars["foo"],copia.data_vars["foo"])
        
    def test_convert_longitude_range(self):
        self.classedati1.convert_longitude_range()
        np.testing.assert_array_equal(self.classedati1.longitude,([1., 2., 3.,4.]))
        self.classedati2.convert_longitude_range()
        np.testing.assert_array_equal(self.classedati2.longitude,([-10., -9., -8.,-7.]))
        
    def test_combine_dataset(self):
        self.classedati2.combine_dataset([self.classedati1.dataset])
        np.testing.assert_array_equal(self.classedati2.latitude,self.classedati1.latitude)
        np.testing.assert_array_equal(self.classedati2.longitude,([]))
        np.testing.assert_array_equal(self.classedati2.time,self.classedati1.time)
        
    def test_convert_unit_of_measure(self):
        copia=self.classedati1.copy()
        self.classedati1.convert_unit_of_measure("foo",10)
        np.testing.assert_array_equal(self.classedati1.data_vars["foo"],copia.data_vars["foo"]*10)
        
    def test_upscale_lon_lat(self):
        self.classedati3.upscale_lon_lat(self.classedati1.dataset)
        np.testing.assert_array_equal(self.classedati3.longitude,([1., 2., 3.,4.]))
        np.testing.assert_array_equal(self.classedati3.latitude,([1., 2., 3.,4.]))

unittest.main(argv=[''], verbosity=2, exit=False)