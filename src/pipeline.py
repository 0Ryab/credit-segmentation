import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib


class LogTransformer(BaseEstimator, TransformerMixin):
    """Aplica transformação log1p em colunas financeiras."""
    
    def __init__(self, cols):
        self.cols = cols
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            if col in X.columns:
                X[col] = np.log1p(X[col])
        return X


class DropColumns(BaseEstimator, TransformerMixin):
    """Remove colunas desnecessárias."""
    
    def __init__(self, cols):
        self.cols = cols
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop(columns=self.cols, errors='ignore')


def build_pipeline(n_components=6, n_clusters=4):
    """Constrói o pipeline completo de segmentação."""
    
    cols_log = [
        'BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES',
        'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
        'PAYMENTS', 'MINIMUM_PAYMENTS', 'CREDIT_LIMIT'
    ]
    
    cols_drop = ['CUST_ID']
    
    pipeline = Pipeline([
        ('drop', DropColumns(cols=cols_drop)),
        ('log', LogTransformer(cols=cols_log)),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components)),
        ('kmeans', KMeans(n_clusters=n_clusters, 
                         random_state=42, n_init=10))
    ])
    
    return pipeline


def train_pipeline(df: pd.DataFrame, 
                   n_components: int = 6,
                   n_clusters: int = 4) -> tuple:
    """Treina o pipeline e retorna modelo e labels."""
    
    pipeline = build_pipeline(n_components, n_clusters)
    labels = pipeline.fit_predict(df)
    
    return pipeline, labels


if __name__ == "__main__":
    df = pd.read_csv('data/raw/CC GENERAL.csv')
    df['CREDIT_LIMIT'] = df['CREDIT_LIMIT'].fillna(
        df['CREDIT_LIMIT'].median()
    )
    df['MINIMUM_PAYMENTS'] = df['MINIMUM_PAYMENTS'].fillna(
        df['MINIMUM_PAYMENTS'].median()
    )
    
    pipeline, labels = train_pipeline(df)
    
    joblib.dump(pipeline, 'models/pipeline.pkl')
    
    print("Pipeline treinado e salvo!")
    print(f"Clusters: {pd.Series(labels).value_counts().sort_index().to_dict()}")