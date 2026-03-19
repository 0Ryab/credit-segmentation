import pytest
import numpy as np
import pandas as pd
from src.pipeline import LogTransformer, DropColumns, build_pipeline, train_pipeline


@pytest.fixture
def sample_df():
    """Dataset de exemplo para testes."""
    return pd.DataFrame({
        'CUST_ID': ['C001', 'C002', 'C003', 'C004', 'C005'],
        'BALANCE': [100.0, 0.0, 5000.0, 200.0, 1000.0],
        'PURCHASES': [500.0, 0.0, 2000.0, 100.0, 800.0],
        'ONEOFF_PURCHASES': [500.0, 0.0, 1000.0, 100.0, 400.0],
        'INSTALLMENTS_PURCHASES': [0.0, 0.0, 1000.0, 0.0, 400.0],
        'CASH_ADVANCE': [0.0, 1000.0, 0.0, 50.0, 0.0],
        'PURCHASES_FREQUENCY': [0.8, 0.0, 0.9, 0.3, 0.7],
        'ONEOFF_PURCHASES_FREQUENCY': [0.8, 0.0, 0.5, 0.3, 0.5],
        'PURCHASES_INSTALLMENTS_FREQUENCY': [0.0, 0.0, 0.5, 0.0, 0.3],
        'CASH_ADVANCE_FREQUENCY': [0.0, 0.5, 0.0, 0.1, 0.0],
        'CASH_ADVANCE_TRX': [0, 5, 0, 1, 0],
        'PURCHASES_TRX': [10, 0, 20, 2, 8],
        'CREDIT_LIMIT': [5000.0, 3000.0, 10000.0, 2000.0, 7000.0],
        'PAYMENTS': [600.0, 500.0, 2500.0, 150.0, 900.0],
        'MINIMUM_PAYMENTS': [100.0, 200.0, 300.0, 50.0, 150.0],
        'PRC_FULL_PAYMENT': [0.8, 0.0, 0.1, 0.5, 0.9],
        'TENURE': [12, 6, 24, 8, 18],
        'BALANCE_FREQUENCY': [0.9, 0.5, 1.0, 0.6, 0.8]
    })


class TestLogTransformer:
    
    def test_transforma_valores_positivos(self, sample_df):
        """Log de valores positivos deve ser maior que zero."""
        transformer = LogTransformer(cols=['BALANCE', 'PURCHASES'])
        result = transformer.transform(sample_df)
        assert result['BALANCE'].iloc[2] > 0  # 5000 → log1p > 0
    
    def test_transforma_zeros(self, sample_df):
        """Log de zero deve resultar em zero."""
        transformer = LogTransformer(cols=['PURCHASES'])
        result = transformer.transform(sample_df)
        assert result['PURCHASES'].iloc[1] == 0.0  # log1p(0) = 0
    
    def test_nao_altera_outras_colunas(self, sample_df):
        """Colunas não listadas não devem ser alteradas."""
        transformer = LogTransformer(cols=['BALANCE'])
        result = transformer.transform(sample_df)
        pd.testing.assert_series_equal(
            result['TENURE'], sample_df['TENURE']
        )
    
    def test_fit_retorna_self(self, sample_df):
        """Fit deve retornar o próprio transformer."""
        transformer = LogTransformer(cols=['BALANCE'])
        result = transformer.fit(sample_df)
        assert result is transformer


class TestDropColumns:
    
    def test_remove_coluna(self, sample_df):
        """Coluna listada deve ser removida."""
        dropper = DropColumns(cols=['CUST_ID'])
        result = dropper.transform(sample_df)
        assert 'CUST_ID' not in result.columns
    
    def test_mantem_outras_colunas(self, sample_df):
        """Colunas não listadas devem ser mantidas."""
        dropper = DropColumns(cols=['CUST_ID'])
        result = dropper.transform(sample_df)
        assert 'BALANCE' in result.columns
    
    def test_coluna_inexistente_nao_quebra(self, sample_df):
        """Coluna inexistente não deve gerar erro."""
        dropper = DropColumns(cols=['COLUNA_INEXISTENTE'])
        result = dropper.transform(sample_df)
        assert result.shape == sample_df.shape


class TestPipeline:
    
    def test_build_pipeline_retorna_pipeline(self):
        """build_pipeline deve retornar um Pipeline."""
        from sklearn.pipeline import Pipeline
        pipeline = build_pipeline()
        assert isinstance(pipeline, Pipeline)
    
    def test_pipeline_tem_etapas_corretas(self):
        """Pipeline deve ter as 5 etapas na ordem correta."""
        pipeline = build_pipeline()
        etapas = [name for name, _ in pipeline.steps]
        assert etapas == ['drop', 'log', 'scaler', 'pca', 'kmeans']
    
    def test_train_pipeline_retorna_labels(self, sample_df):
        """train_pipeline deve retornar labels para cada linha."""
        _, labels = train_pipeline(
            sample_df, n_components=2, n_clusters=2
        )
        assert len(labels) == len(sample_df)
    
    def test_labels_dentro_do_range(self, sample_df):
        """Labels devem estar entre 0 e n_clusters-1."""
        n_clusters = 2
        _, labels = train_pipeline(
            sample_df, n_components=2, n_clusters=n_clusters
        )
        assert labels.min() >= 0
        assert labels.max() <= n_clusters - 1