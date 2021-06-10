# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 17:51:20 2021

@author: Pierre
"""

import pandas as pd
from imblearn.over_sampling import RandomOverSampler

import warnings

warnings.filterwarnings("ignore")

def load_data(split = True, resample = True, random_state = 42):
    X = pd.read_csv("Data/amf_train_x.csv", index_col = 'Index')
    y = pd.read_csv("Data/amf_train_y.csv")
    
    y = X.merge(y, on = 'Trader')['type'].replace(['NON HFT', 'MIX', 'HFT'], [0, 1 , 2])
    y.index = X.index
    
    if split:
        test_traders = X['Trader'].value_counts().sample(15, random_state = random_state)
        X_train = X.reset_index().set_index("Trader").drop(test_traders.index).reset_index().set_index('Index')
        X_test = X.reset_index().set_index("Trader").loc[test_traders.index].reset_index().set_index('Index')
        
        y_train = y.loc[X_train.index]
        y_test = y.loc[X_test.index]
        
        if resample:
            ros = RandomOverSampler(random_state=42)
            X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
            
            return X_train_resampled, y_train_resampled, X_test, y_test
        
        return X_train, y_train, X_test, y_test
    
    return X, y, None, None


