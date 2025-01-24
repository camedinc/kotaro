#--------------------------------------------------------------------------
# Clase que ejecuta las tareas de limpieza y normalización del set de datos
#--------------------------------------------------------------------------

# Librerías
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import List

import logging
logging.basicConfig(level=logging.INFO)

# Clase
class DataCleaner:
    """
    La clase ofrece una secuencia de métodos que permiten depurar un dataframe en bruto.

    df(DataFrame): Base de datos que ingresa como parámetro.
    """
    def __init__(self, df):
        # Valida que el obeto reciba un DataFrame
        if not isinstance(df, pd.DataFrame):
            raise TypeError("El parámetro 'df' debe ser un pandas DataFrame.")
        self.df = df


    def _get_copy(self) -> pd.DataFrame:
        """
        Devuelve una copia del DataFrame.

        Returns:
            pd.DataFrame: Un DataFrame que es una copia exacta al original.
        """
        return self.df.copy()
    

    def _get_numeric_columns(self) -> list[str]:
        """
        Captura los nombres de las columnas numéricas.

        Returns:
            list: Una lista con los nombres de las columnas de tipo
            numérico del DataFrame.
        """
        return self.df.select_dtypes(include=["number"]).columns.tolist()
    

    def get_data_types(self) -> pd.DataFrame:
        """
        Muestra los tipos de variables en el DataFrame.

        Returns:
            pd.DataFrame: Un DataFrame con dos columnas ('Columnas', 'Tipo')
            que muestran el nombre y el tipo de cada columna.
        """
        df_type = self.df.dtypes.rename_axis('Columna').reset_index(name='Tipo') 
        return df_type
    

    def get_type_counts(self) -> pd.DataFrame:
        """
        Muestra el conteo de los tipos de variables en el Dataframe.

        Returns:
            pd.DataFrame: Un DataFrame con tres columnas ('Tipo', 'Conteo', 'Porcentaje')
            que muestran la cantidad y proporción de tipos.
        """
        type_summary = self.df.dtypes.value_counts().reset_index(name='Conteo')
        type_summary.columns = ['Tipo', 'Conteo']
        total = type_summary['Conteo'].sum()
        type_summary['Porcentaje'] = (type_summary['Conteo']/total)*100
        return type_summary
    

    def describe_data(self) -> pd.DataFrame:
        """
        Genera resúmenes estadísticos de las variables numéricas y categóricas de DataFrame.

        Returns:
            pd.DataFrame: Dos DataFrames con los estadísticos principales tanto para las
            variables numéricas y categóricas, por separado.
        """
        # Genera un resumen de las variables numéricas
        numerical_summary = self.df.describe(include=["number"]).T
        categorical_summary = self.df.describe(exclude=["number"]).T
        return {"Variables numéricas" : numerical_summary, 
                "Variables categóricas": categorical_summary}
    

    def get_null_counts(self) -> pd.DataFrame:
        """
        Muestra la cantidad de valores nulos presentes en el DataFrame.

        Returns:
            pd.DataFrame: Un DataFrames con dos columnas ('Columna', 'Conteo de Nas')
            que muestran la cantidad de valores perdidos por variable.
        """
        df_na = self.df.isna().sum().reset_index(name='Nulos')
        return df_na
    

    def impute_missing_values(self, columns: list[str] = None) -> pd.DataFrame:
        """
        Imputa los valores nulos con el valor de la media de la columna.

        Returns:
            pd.DataFrame: Copia del DataFrame original con los
            valores perdidos imputados por el valor de la media de la columna respectiva.
        """
        if columns is None:
            columns = self.df.select_dtypes(include=["number"]).columns.tolist()

        # Crear una copia del DataFrame para conservar el original
        df_copy = self._get_copy()

        # Registra las columnas que fueron imputadas
        columns_imputed = []
        for col in columns:
            if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
                if df_copy[col].isnull().any():
                    df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
                    columns_imputed.append(col)

        # Verificar si no se imputaron columnas
        if not columns_imputed:
            logging.info("No se imputaron columnas.")
        else:
            logging.info("Columnas con alguna imputación: %d", len(columns_imputed))
        return df_copy
    

    def drop_missing_values(self) -> pd.DataFrame:
        """
        Elimina las filas con valores nulos en el DataFrame.

        Returns:
            pd.DataFrame: Copia del DataFrame original con los
            los registros con valores perdidos eliminados.
        """
        # Crear una copia del DataFrame para conservar el original
        df_copy = self._get_copy()

        if df_copy.isna().sum().sum() == 0:
            logging.info("No hay registros con nulos que eliminar.")
        else:
            logging.info("Registros con nulos eliminados: %d", df_copy.isna().sum().sum())
            return df_copy.dropna()
    

    def get_duplicate_rows(self):
        """
        Muestra el número de filas duplicadas presentes en el DataFrame.

        Returns:
            int: Valor entero que es la cantidad de registros 
            duplicados en el DataFrame.
        """
        return {"Total duplicados" : self.df.duplicated().sum()}
    

    def drop_duplicate_rows(self) -> pd.DataFrame:
        """
        Elimina las filas duplicadas del DataFrame.

        Returns:
            pd.DataFrame: Copia del DataFrame original con los
            registros duplicados presentes sólo una vez.
        """
        # Crear una copia del DataFrame para conservar el original
        df_copy = self._get_copy()

        if self.df.duplicated().sum() == 0:
            logging.info("No hay registros duplicados que eliminar.")
        else:
            logging.info("Duplicados eliminados: %d", self.df.duplicated().sum())
            return df_copy.drop_duplicates()
    

    def normalize(self) -> pd.DataFrame:
        """
        Normaliza las columnas numéricas del DataFrame.

        Returns:
            pd.DataFrame: Copia del DataFrame original con columnas
            numéricas normalizadas.
        """
        # Crear una copia del DataFrame para conservar el original
        df_copy = self._get_copy()

        # Selecciona el nombre de las columnas numéricas
        numerical_columns = self._get_numeric_columns()

        if not numerical_columns:
            logging.info("No se normalizaron columnas.")
        else:
            # Normaliza las columnas numéricas
            scaler = MinMaxScaler()
            df_copy[numerical_columns] = scaler.fit_transform(df_copy[numerical_columns])
            logging.info("Columnas normalizadas: %d", len(numerical_columns))
        return df_copy
    

    def standardize(self) -> pd.DataFrame:
        """
        Estandariza las columnas numéricas del DataFrame.

        Returns:
            pd.DataFrame: Copia del DataFrame original con columnas
            numéricas estandarizadas.
        """
        # Crear una copia del DataFrame para conservar el original
        df_copy = self._get_copy()

        # Selecciona el nombre de las columnas numéricas
        numerical_columns = self._get_numeric_columns()

        if not numerical_columns:
            logging.info("No se estandarizaron columnas.")
        else:
            # Estandariza las columnas numéricas
            scaler = StandardScaler()
            df_copy[numerical_columns] = scaler.fit_transform(df_copy[numerical_columns])
            logging.info("Columnas estandarizadas: %d", len(numerical_columns))
        return df_copy
    

    def apply_one_hot_encoding(self) -> pd.DataFrame:
        """
        Aplica one-hot encoding (OHE).

        Returns:
            pd.DataFrame: Copia del DataFrame original con las
            columnas categóricas en formato dummies.
        """
        # Crear una copia del DataFrame para conservar el original
        df_copy = self._get_copy()
        categorical_columns = df_copy.select_dtypes(exclude=["number"]).columns.tolist()
        df_encoded = pd.get_dummies(df_copy, columns=categorical_columns, drop_first=True)

        if not categorical_columns:
            logging.info("No hay columnas categóricas.")
        else:
            logging.info("Columnas categóricas a dummies: %d", len(categorical_columns))
        return df_encoded