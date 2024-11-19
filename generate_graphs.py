# generate_graphs.py
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Cargar los datos
DATA_FILE = 'data\Data.csv'
data = pd.read_csv(DATA_FILE)

if data.empty:
    print("El archivo CSV no contiene datos o no se cargó correctamente.")
else:
    print(f"El archivo CSV contiene {len(data)} filas y {len(data.columns)} columnas.")




def generate_summary_table():
    """
    Genera una tabla interactiva con información resumen de la base de datos.
    """
    # Crear un resumen de la base de datos
    summary = data.describe(include='all').transpose()  # Información descriptiva
    summary['missing_values'] = data.isnull().sum()  # Agregar valores nulos
    summary['unique_values'] = data.nunique()  # Agregar valores únicos
    
    # Seleccionar columnas relevantes
    summary_table = summary[['mean', 'std', 'min', 'max', 'missing_values', 'unique_values']].reset_index()
    summary_table.rename(columns={
        'index': 'Column',
        'mean': 'Promedio',
        'std': 'Desviación Estándar',
        'min': 'Valor Mínimo',
        'max': 'Valor Máximo',
        'missing_values': 'Valores Nulos',
        'unique_values': 'Valores Únicos'
    }, inplace=True)
    
    # Crear tabla con Plotly
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(summary_table.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[summary_table[col] for col in summary_table.columns],
                   fill_color='lavender',
                   align='left'))
    ])
    
    fig.update_layout(title="Resumen de la Base de Datos")
    return fig.to_html(full_html=False)



def generate_histogram():
    """
    Genera un histograma interactivo de notas de admisión con una línea de densidad KDE.
    """
    import numpy as np
    from scipy.stats import gaussian_kde 

    # Datos
    x = data["Admission grade"].dropna()

    # Calcular la densidad KDE
    kde = gaussian_kde(x)
    x_kde = np.linspace(x.min(), x.max(), 100)
    y_kde = kde(x_kde)

    # Normalizar el histograma para que coincida con la escala de KDE
    hist_values, bin_edges = np.histogram(x, bins=20, density=True)

    # Crear el gráfico con Plotly
    fig = go.Figure()

    # Agregar el histograma normalizado
    fig.add_trace(go.Bar(
        x=bin_edges[:-1],  # Bins
        y=hist_values,     # Frecuencia normalizada
        name="Frecuencia",
        marker_color="blue",
        opacity=0.7
    ))

    # Agregar la línea KDE
    fig.add_trace(go.Scatter(
        x=x_kde,
        y=y_kde,
        mode="lines",
        name="Densidad",
        line=dict(color="red", width=2)
    ))

    # Configurar el diseño del gráfico
    fig.update_layout(
        title="Histograma Interactivo de Notas de Admisión con Línea KDE",
        xaxis_title="Nota de Admisión",
        yaxis_title="Densidad",
        legend_title="Elementos"
    )

    return fig.to_html(full_html=False)



def generate_pie_chart():
    """
    Genera un gráfico de pastel interactivo para becados.
    """
    scholarship_counts = data['Scholarship holder'].value_counts()
    labels = ['No Becado', 'Becado']
    values = scholarship_counts.values

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0.05, 0])])
    fig.update_layout(title="Proporción de Becados")
    return fig.to_html(full_html=False)



def generate_bar_plot():
    """
    Genera un gráfico de barras interactivo para unidades curriculares aprobadas.
    """
    fig = px.bar(
        data, 
        x='Target', 
        y='Curricular units 2nd sem (approved)', 
        title="Unidades Curriculares Aprobadas por Resultado",
        labels={'Target': 'Resultado', 'Curricular units 2nd sem (approved)': 'Unidades Aprobadas'},
        color='Target'
    )
    fig.update_layout(xaxis_title="Resultado (Dropout, Enrolled, Graduate)", yaxis_title="Promedio Unidades Aprobadas")
    return fig.to_html(full_html=False)



def generate_correlation_heatmap():
    """
    Genera un heatmap interactivo de correlación.
    """
    corr_matrix = data[['Admission grade', 'Curricular units 1st sem (grade)', 
                        'Curricular units 2nd sem (grade)', 'Unemployment rate', 
                        'GDP', 'Inflation rate']].corr()
    
    fig = px.imshow(corr_matrix,
                    text_auto=True,
                    color_continuous_scale='Viridis',  # Escala de color compatible con Plotly
                    labels=dict(color="Correlación"))
    
    # Actualiza el diseño para cambiar el tamaño
    fig.update_layout(title="Heatmap de Correlación", 
                      xaxis_title="Variables", 
                      yaxis_title="Variables",
                      width=900,  # Ancho del gráfico
                      height=700)  # Alto del gráfico
    
    return fig.to_html(full_html=False)



def generate_graphs():
    """
    Genera todos los gráficos interactivos y los devuelve en formato HTML.
    """
    graphs = {
        "histogram": generate_histogram(),
        "pie_chart": generate_pie_chart(),
        "bar_plot": generate_bar_plot(),
        "correlation_heatmap": generate_correlation_heatmap(),
        "summary_table": generate_summary_table()
    }
    return graphs



