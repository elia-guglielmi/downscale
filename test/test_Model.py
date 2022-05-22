import unittest
import numpy as np
import pandas as pd
import xarray as xr
from downscale.Model import Model
from pandas.testing import assert_frame_equal

class TestModel(unittest.TestCase):
    def setUp(self):
        data =  np.arange(4, dtype=np.int64).reshape(4, 1,1)
        lon = ([1.])
        lon=np.array(lon,dtype='float32')
        lat = ([1.])
        lat=np.array(lat,dtype='float32')
        times = pd.date_range("2000-01-01", periods=4)
        x = xr.Dataset(
        data_vars=dict(
            foo=(["time", "latitude", "longitude"], data),
            boo=(["time", "latitude", "longitude"], data),
            ),
            coords=dict(
            time=times,
            latitude=("latitude",lat),
            longitude=("longitude",lon),
            ),
        )
        self.ds=x
        self.rf=Model("RandomForest")
        self.sm=Model("SupportVectorMachine")
        
    def test_train_test_split(self):
        X_train,y_train,X_test,y_test=self.rf.train_test_split(self.ds,"foo","boo","2000-01-01","2000-01-03","2000-01-04","2000-01-04")
        self.sm.train_test_split(self.ds,"foo","boo","2000-01-01","2000-01-03","2000-01-04","2000-01-04")
        self.assertEqual(self.rf.name,"RandomForest_lat1_0_lon1_0")
        self.assertEqual(self.sm.name,"SVM_lat1_0_lon1_0")
        times = pd.date_range("2000-01-01", periods=4)
        xdf = pd.DataFrame(data={'foo': [0.,1.,2.]},index=(pd.MultiIndex.from_tuples([(times[0],1.0, 1.0),
                    ( times[1],1.0, 1.0),
                    ( times[2],1.0, 1.0)],
                   names=["time", "latitude", "longitude"])))
        pd.testing.assert_frame_equal(xdf, X_train)
        np.testing.assert_array_equal(y_train,([0.,1., 2.]))
        ydf = pd.DataFrame(data={'foo': [3.]},index=(pd.MultiIndex.from_tuples([(times[3],1.0, 1.0),],
                   names=["time", "latitude", "longitude"])))
        self.assertTrue(X_test.equals(ydf))
        np.testing.assert_array_equal(y_test,([3]))
        
        
    def  test_accuracy(self):
        X_train,y_train,X_test,y_test=self.rf.train_test_split(self.ds,"foo","boo","2000-01-01","2000-01-03","2000-01-04","2000-01-04")
        self.rf.train(X_train,y_train)
        self.rf.accuracy(X_test,y_test)
        self.assertNotEqual(self.rf.stats,"")
        
        
unittest.main(argv=[''], verbosity=2, exit=False)   