import pandas as pd
import numpy as np
import sys
import os

# Ruta actual
print("Ruta actual:", os.getcwd())

# Ruta absoluta del directorio base y sube un nivel ".."
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print("Ruta absoluta (base_dir)", base_dir)

# Agregar el directorio base al PYTHONPATH
sys.path.append(base_dir)

# Ruta al archivo de datos relativo al directorio base
file_path = os.path.join(base_dir, "data", "employee_attrition_dataset.csv")
print(f"Ruta absoluta al archivo (file_path): {file_path}")

# Leer el archivo CSV
df = pd.read_csv(file_path)
print("Archivo cargado exitosamente")

# Importar la clase DataCleaner
from kotaro import DataCleaner

# Instancia de la clase DataCleaner
data = DataCleaner(df)

# Función principal del pipeline de limpieza
def data_pipeline(data_cleaner):
    
    # Consulta tipos
    df_type = data_cleaner.get_data_types()
    print(f"Tipos de datos: \n{df_type}")

    # Genera un resumen
    df_num, df_cat = data_cleaner.describe_data()
    print(f"Resumen de datos numéricos y categóricos: \n{df_num}\n{df_cat}")

    # Conteo de tipos
    type_count = data_cleaner.get_type_counts()
    print(f"Conteo de tipos: \n{type_count}")

    # Evalúa los nulos
    df_na = data_cleaner.get_null_counts()
    print(f"Conteo de valores nulos: \n{df_na}")

    # Imputar posibles valores perdidos por media
    df_impute = data_cleaner.impute_missing_values()
    print(f"Datos imputados: \n{df_impute}")

    # Eliminar los nulos
    df_drop_na = data_cleaner.drop_missing_values()
    print(f"Datos después de eliminar nulos: \n{df_drop_na}")

    # Muestra si hay duplicados
    df_duplicados = data_cleaner.get_duplicate_rows()
    print(f"Duplicados: \n{df_duplicados}")

    # Eliminar los duplicados si los hay
    df_drop_duplicados = data_cleaner.drop_duplicate_rows()
    print(f"Datos después de eliminar duplicados: \n{df_drop_duplicados}")

    # Normaliza variables numéricas
    df_normalize = data_cleaner.normalize()
    print(f"Datos normalizados: \n{df_normalize}")

    # Estandariza variables numéricas
    df_standard = data_cleaner.standardize()
    print(f"Datos estandarizados: \n{df_standard}")

    # Aplica variables dummies para categóricas (One-Hot Encoding)
    df_ohe = data_cleaner.apply_one_hot_encoding()
    print(f"Datos con One-Hot Encoding: \n{df_ohe}")

    return df_ohe  # Devolver el dataframe final

# Ejecución del pipeline y guardado del dataframe limpio
df_final = data_pipeline(data)

# Guardar el dataframe limpio como un archivo .csv en la carpeta /data
output_file_path = os.path.join(base_dir, "data", "employee_attrition_cleaned.csv")
df_final.to_csv(output_file_path, index=False)
print(f"Archivo limpio guardado en: {output_file_path}")