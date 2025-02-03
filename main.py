import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Ruta actual
print("Ruta actual:", os.getcwd())

# Ruta absoluta del directorio base y sube un nivel ".."
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
print("Ruta absoluta (base_dir)", base_dir)

# Agregar el directorio base al PYTHONPATH
sys.path.append(base_dir)

# Ruta al archivo de datos relativo al directorio base
file_path = os.path.join(base_dir, "data", "employee_attrition_dataset.csv")
print(f"Ruta absoluta al archivo (file_path): {file_path}")

# Ruta al directorio de gráficas
path_graficas = os.path.join(base_dir, "graficas")
print(f"Ruta absoluta al directorio gráficas: {path_graficas}")

# Leer el archivo CSV
df = pd.read_csv(file_path)
print("Archivo cargado exitosamente")
#print(df.head())

# Importar la clase DataCleaner
from kotaro import DataCleaner

# Instancia
data = DataCleaner(df)

# Consulta tipos
df_type = data.get_data_types()
print(df_type)

# Genera un resumen
df_num, df_cat = data.describe_data()

# Conteo de tipos
type_count = data.get_type_counts()
print(type_count)

# Evalúa los nulos
df_na = data.get_null_counts()
print(df_na)

# Imputar posibles valores perdidos por media
df_impute = data.impute_missing_values()
print(df_impute)

# Eliminar los nulos NAs
df_drop_na = data.drop_missing_values()
print(df_drop_na)

# Muestra si hay duplicados
df_duplicados = data.get_duplicate_rows()
print(df_duplicados)

# Eliminar los duplicados si los hay
df_drop_duplicados = data.drop_duplicate_rows()
print(df_drop_duplicados)

# Normaliza variables numéricas
df_normalize = data.normalize()
print(df_normalize)

# Estandariza variables numéricas
df_standard = data.standardize()
print(df_standard)

# Aplica variables dummies para categóricas ohe
df_ohe = data.apply_one_hot_encoding()
print(df_ohe)

# Prueba de funciones gráficas
# Importar la clase DataCleaner
from kotaro import DataExplorer
numeric_cols = ["Monthly_Income", 
                "Years_at_Company", 
                "Years_in_Current_Role", 
                "Years_Since_Last_Promotion", 
                "Training_Hours_Last_Year",
                "Job_Involvement", 
                "Distance_From_Home", 
                "Age"]

# Instancia
plot = DataExplorer(df=df_ohe, numeric_cols=numeric_cols)
fig, axes = plot.plot_graphics(plot_type="Histogram")
fig.savefig(os.path.join(path_graficas, 'histogramas.png'), dpi = 300, bbox_inches='tight')
plt.show()