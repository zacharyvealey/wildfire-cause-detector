import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

import sys, os.path
sys.path.append(os.path.abspath('../'))
from settings import Settings

ml_settings_copy = Settings()
cycle_ranges = cycle_ranges = ml_settings_copy.cycle_ranges
cycle_offsets = ml_settings_copy.cycle_offsets

class ConvertToCyclical(BaseEstimator, TransformerMixin):
    """
    A class to convert features into a cyclical time format (i.e.,
    the days of the week are all adjacent to each other instead of
    labeled 0-6).
    """
    def __init__(self, ml_settings): 
        pass
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_cycle = None
        ind = 0
        
        for c_range, c_off in zip(cycle_ranges, cycle_offsets):
            sin_vals = np.sin((X[:, ind] - c_off) * 2 * np.pi / c_range)
            cos_vals = np.cos((X[:, ind] - c_off) * 2 * np.pi / c_range)
            
            if X_cycle is not None:
                X_cycle = np.c_[X_cycle, sin_vals, cos_vals]
            else:
                X_cycle = np.c_[sin_vals, cos_vals]
            
            ind += 1
        
        return X_cycle