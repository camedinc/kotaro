#--------------------------------------------------------------------------
# Clase que ejecuta las tareas de análisis exploratorio
#--------------------------------------------------------------------------

# Librerías
import pandas as pd
import matplotlib.pyplot as plt

# Graficar el conjunto de variables
class DataExplorer:
    def __init__(self, df = None, numeric_cols: list[str] = None):
        self.df = df
        self.numeric_cols = numeric_cols

    def plot_graphics(self, plot_type="Histogram"):
        """
        """
        self.plot_type = plot_type
        
        if self.numeric_cols is None:
            # Seleccionar sólo columnas numéricas
            self.numeric_cols = self.df.select_dtypes(include=["number"]).columns.tolist()
            n_cols = len(self.numeric_cols)
        else:
            n_cols = len(self.numeric_cols)

        if n_cols >= 0:
            print(f"Se grafican {n_cols} campos numéricos")

        if n_cols == 0:
            print("No hay columnas numéricas en el DataFrame.")

        # Determinar el número de filas (máx. 3 columnas por fila)
        rows = (n_cols // 3) + (n_cols % 3 > 0)

        # Crear grilla
        fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))

        # Asegurar que axes sea iterable
        axes = axes.flatten() if n_cols > 1 else [axes]

        for i, col in enumerate(self.numeric_cols):
            if self.plot_type == "Histogram":
                axes[i].hist(self.df[col], bins=20, color='skyblue', edgecolor='black')
                axes[i].set_title(f'Histograma de {col}')
                #axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frecuencia')
        
        # Ocultar los ejes sobrantes si hay menos de 3*n filas
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        # Ajustar la grilla para evitar el solapamiento de los textos
        plt.subplots_adjust(hspace=0.5, wspace=0.5)  # Ajusta el espacio horizontal y vertical
        
        # Ajustar automáticamente el layout para evitar el solapamiento de textos
        plt.tight_layout(pad=7.5)  # El parámetro 'pad' agrega espacio adicional entre las subgráficas

        # Mostrar las gráficas
        plt.show()

        return fig, axes