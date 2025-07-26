import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib

class FeatureProcessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.pipeline = None
        self.feature_columns = [
            'step', 'type', 'amount',
            'oldbalanceOrg', 'newbalanceOrig',
            'oldbalanceDest', 'newbalanceDest'
        ]

    def build_pipeline(self):
        numeric_features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        cat_features = ['type']

        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('scaler', StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])

        self.pipeline = ColumnTransformer([
            ('num', numeric_pipeline, numeric_features),
            ('cat', cat_pipeline, cat_features)
        ])

    def fit(self, X, y=None):
        self.build_pipeline()
        X_proc = X[self.feature_columns].copy()
        self.pipeline.fit(X_proc)
        return self

    def transform(self, X):
        X_proc = X[self.feature_columns].copy()
        return self.pipeline.transform(X_proc)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
