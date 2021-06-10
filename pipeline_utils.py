# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 17:51:20 2021

@author: Pierre
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import f1_score

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import QuantileTransformer, FunctionTransformer
from sklearn.impute import SimpleImputer, MissingIndicator

from category_encoders import TargetEncoder, CountEncoder

from lightgbm.sklearn import LGBMClassifier

import warnings

warnings.filterwarnings("ignore")

def na_counter(x):
    return x.isna().sum(axis = 1).values.reshape((-1, 1))

class MultiClassTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None
    
    def fit(self, X, y):
        y = pd.get_dummies(y)
        self.encoders = [TargetEncoder().fit(X, y[y_i]) for y_i in y.columns]
        return self
    
    def transform(self, X):
        return pd.concat([encoder.transform(X) for encoder in self.encoders], axis = 1).values
    

def make_pipeline(params):
    
    NACounter = FunctionTransformer(na_counter)
    imputer = SimpleImputer(strategy = "constant", fill_value = -1)
    na_indicator = MissingIndicator()
    
    preprocessing = Pipeline([
        ("cleaning", FeatureUnion([
            ("na_counter", NACounter),
            ("impute", imputer),
            ("missing_indicator", na_indicator),
            ("category_encoding", ColumnTransformer([
                ('day_target_encoding', MultiClassTargetEncoder(), 'Day'),
                ('share_target_encoding', MultiClassTargetEncoder(), 'Share'),
                ('day_count_encoding', CountEncoder(handle_unknown = 0, min_group_size = 0, handle_missing = 0), 'Day'),
                ('share_count_encoding', CountEncoder(handle_unknown = 0, min_group_size = 0, handle_missing = 0), 'Share')],
                remainder = "drop"))
        ])),
        ("drop_trader_date_share", ColumnTransformer([
            ("drop_columns", "drop", [1, 2, 3])
        ], remainder = "passthrough")),
        ("scaling", QuantileTransformer())
    ])

    pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("model", LGBMClassifier(**params))
    ])
    
    return pipeline

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    