import pytest
import pandas as pd
import numpy as np
from kotaro import DataCleaner

# Datos de prueba (puedes crear un dataframe pequeño para probar)
@pytest.fixture
def sample_data():
    data = {
        'age': [25, 30, 35, np.nan, 40],
        'salary': [50000, 55000, 60000, 45000, np.nan],
        'department': ['HR', 'Finance', 'IT', 'HR', 'Finance']
    }
    df = pd.DataFrame(data)
    return df

# Test para verificar que los tipos de datos se devuelven correctamente
def test_get_data_types(sample_data):
    data_cleaner = DataCleaner(sample_data)
    data_types = data_cleaner.get_data_types()
    assert data_types == {'age': 'float64', 'salary': 'float64', 'department': 'object'}

# Test para verificar el conteo de valores nulos
def test_get_null_counts(sample_data):
    data_cleaner = DataCleaner(sample_data)
    null_counts = data_cleaner.get_null_counts()
    assert null_counts == {'age': 1, 'salary': 1, 'department': 0}

# Test para la imputación de valores nulos (usando media)
def test_impute_missing_values(sample_data):
    data_cleaner = DataCleaner(sample_data)
    df_imputed = data_cleaner.impute_missing_values()
    assert df_imputed['age'].isnull().sum() == 0  # Verifica que no haya valores nulos en 'age'
    assert df_imputed['salary'].isnull().sum() == 0  # Verifica que no haya valores nulos en 'salary'

# Test para eliminar filas con nulos
def test_drop_missing_values(sample_data):
    data_cleaner = DataCleaner(sample_data)
    df_dropped = data_cleaner.drop_missing_values()
    assert df_dropped.isnull().sum().sum() == 0  # Verifica que no haya valores nulos

# Test para verificar si existen duplicados
def test_get_duplicate_rows(sample_data):
    data_cleaner = DataCleaner(sample_data)
    df_duplicates = data_cleaner.get_duplicate_rows()
    assert df_duplicates.shape[0] == 0  # Verifica que no haya duplicados

# Test para eliminar duplicados
def test_drop_duplicate_rows(sample_data):
    data_cleaner = DataCleaner(sample_data)
    df_no_duplicates = data_cleaner.drop_duplicate_rows()
    assert df_no_duplicates.shape[0] == 4  # Verifica que el número de filas haya disminuido
