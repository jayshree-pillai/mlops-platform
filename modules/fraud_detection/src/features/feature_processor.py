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
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.pipeline = ColumnTransformer([
            ('num', numeric_pipeline, numeric_features),
            ('cat', cat_pipeline, cat_features)
        ])

    def fit(self, X, y=None):
        self.build_pipeline()
        if isinstance(X, np.ndarray):
            raw_names = [
                'step', 'type', 'amount',
                'oldbalanceOrg', 'newbalanceOrig',
                'oldbalanceDest', 'newbalanceDest'
            ]
            total_cols = X.shape[1]

            if total_cols <= len(raw_names):
                self.feature_columns = raw_names[:total_cols]
            else:
                extra_cols = [f"feature_{i}" for i in range(len(raw_names), total_cols)]
                self.feature_columns = raw_names + extra_cols

            X = pd.DataFrame(X, columns=self.feature_columns)
        else:
            self.feature_columns = X.columns.tolist()
        X_proc = X[self.feature_columns].copy()
        self.pipeline.fit(X_proc)
        self.schema_hash_value = self.hash_schema()

        return self

    def transform(self, X):
        X_proc = X[self.feature_columns].copy()
        return self.pipeline.transform(X_proc)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def save(self, path):
        joblib.dump(self, path)

    @property
    def schema_hash(self):
        return self.schema_hash_value

    @staticmethod
    def load(path):
        return joblib.load(path)

    def dump_schema(self, path: str):
        schema = {
            "feature_columns": self.feature_columns,
            "transformers": str(self.pipeline)
        }
        with open(path, "w") as f:
            json.dump(schema, f, indent=2)

    def hash_schema(self) -> str:
        schema = {
            "feature_columns": self.feature_columns,
            "transformers": str(self.pipeline)
        }
        schema_str = json.dumps(schema, sort_keys=True)
        return hashlib.sha256(schema_str.encode()).hexdigest()
