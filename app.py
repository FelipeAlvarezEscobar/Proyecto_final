import pandas as pd
from diccionarios import DICCIONARIOS
from flask import Flask, request, render_template, jsonify
import os
import generate_graphs  # Importa el script para generar gráficos
import plotly.express as px
from models import ModelTrainer  # Importa la clase ModelTrainer



app = Flask(__name__)

# Load the dataset
DATA_FILE = 'data\Data.csv'
data = pd.read_csv(DATA_FILE)

# Number of rows per page
ROWS_PER_PAGE = 100

# Dictionary for column header translations
COLUMN_TITLES = {
    "Marital status": "Estado civil",
    "Application mode": "Modo de aplicación",
    "Application order": "Orden de aplicación",
    "Course": "Curso",
    "Daytime/evening attendance": "Asistencia diurna/nocturna",
    "Previous qualification": "Calificación previa",
    "Previous qualification (grade)": "Nota calificación previa",
    "Nacionality": "Nacionalidad",
    "Mother's qualification": "Calificación de la madre",
    "Father's qualification": "Calificación del padre",
    "Mother's occupation": "Ocupación de la madre",
    "Father's occupation": "Ocupación del padre",
    "Admission grade": "Nota de admisión",
    "Displaced": "Desplazado",
    "Educational special needs": "Necesidades educativas especiales",
    "Debtor": "Deudor",
    "Tuition fees up to date": "Matrícula al día",
    "Gender": "Género",
    "Scholarship holder": "Becado",
    "Age at enrollment": "Edad al inscribirse",
    "International": "Internacional",
    "Curricular units 1st sem (credited)": "Unidades curriculares 1er sem (aprobadas)",
    "Curricular units 1st sem (enrolled)": "Unidades curriculares 1er sem (inscritas)",
    "Curricular units 1st sem (evaluations)": "Evaluaciones 1er sem",
    "Curricular units 1st sem (approved)": "Unidades curriculares 1er sem (aprobadas)",
    "Curricular units 1st sem (grade)": "Nota media 1er sem",
    "Curricular units 1st sem (without evaluations)": "Unidades curriculares 1er sem (sin evaluación)",
    "Curricular units 2nd sem (credited)": "Unidades curriculares 2do sem (aprobadas)",
    "Curricular units 2nd sem (enrolled)": "Unidades curriculares 2do sem (inscritas)",
    "Curricular units 2nd sem (evaluations)": "Evaluaciones 2do sem",
    "Curricular units 2nd sem (approved)": "Unidades curriculares 2do sem (aprobadas)",
    "Curricular units 2nd sem (grade)": "Nota media 2do sem",
    "Curricular units 2nd sem (without evaluations)": "Unidades curriculares 2do sem (sin evaluación)",
    "Unemployment rate": "Tasa de desempleo",
    "Inflation rate": "Tasa de inflación",
    "GDP": "PIB",
    "Target": "Resultado"
}

# Function to decode categorical values using the master dictionary
def decode_row(row):
    decoded = row.copy()
    for column, mapping in DICCIONARIOS.items():
        if column in row:
            decoded[column] = mapping.get(row[column], row[column])
    return decoded

@app.route('/')
def index():
    """
    Main page displaying the menu and the "Base de datos completa" button.
    """
    return render_template('index.html')

@app.route('/table')
def table():
    """
    Page displaying the paginated table.
    """
    # Get the current page from the request arguments
    page = request.args.get('page', 1, type=int)
    
    # Calculate start and end rows for the current page
    start_row = (page - 1) * ROWS_PER_PAGE
    end_row = start_row + ROWS_PER_PAGE
    
    # Slice the data for the current page
    paginated_data = data.iloc[start_row:end_row].apply(decode_row, axis=1)
    
    # Determine total pages
    total_pages = (len(data) + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE

    return render_template(
        'table.html', 
        data=paginated_data.to_dict(orient='records'),
        page=page,
        total_pages=total_pages,
        column_titles=COLUMN_TITLES
    )
@app.route('/graphs')
def graphs():
    """
    Route to render all interactive graphs.
    """
    from generate_graphs import generate_graphs  # Importa la función
    graphs = generate_graphs()  # Obtén todos los gráficos como HTML

    return render_template('graph.html', graphs=graphs)


@app.route('/models')
def models_results():
    # Instancia y ejecuta el flujo completo de ModelTrainer
    trainer = ModelTrainer(data_path=r'data\Data.csv')
    trainer.load_data()
    trainer.preprocess_data()
    trainer.create_models()
    trainer.train_and_evaluate()

    # Obtén los resultados y el gráfico
    results_df = trainer.get_results()
    accuracy_chart = trainer.generate_accuracy_comparison_chart()

    # Renderiza la tabla y el gráfico
    return render_template(
        'models.html',
        table_data=results_df,  # DataFrame de resultados
        accuracy_chart=accuracy_chart  # Gráfico generado en Plotly
    )


if __name__ == '__main__':
    app.run(debug=True)