# Proyecto de Análisis de Estudiantes: Predicción y Visualización

Este proyecto es una aplicación web interactiva diseñada para analizar datos de estudiantes, generar visualizaciones y entrenar modelos de aprendizaje automático para predecir el estado académico de los mismos. La herramienta ofrece una interfaz intuitiva y modular, permitiendo explorar patrones en los datos y evaluar modelos de clasificación.

---

## **Objetivo**

El objetivo principal es proporcionar una plataforma que permita:
1. Analizar el rendimiento académico y las características demográficas de los estudiantes.
2. Generar gráficos interactivos para explorar patrones en los datos.
3. Entrenar y evaluar múltiples modelos de clasificación para predecir si un estudiante:
   - Desertará.
   - Se inscribirá.
   - Se graduará.

---

## **Características**

### **1. Visualización Interactiva**
- **Gráficos generados con Plotly**:
  - Histogramas para analizar distribuciones.
  - Gráficos de pastel para proporciones.
  - Mapas de calor para correlaciones entre variables.

### **2. Modelos de Clasificación**
- Implementación de varios modelos de aprendizaje automático:
  - **Regresión Logística.**
  - **Random Forest.**
  - **Gradient Boosting (LightGBM, CatBoost, XGBoost).**
  - **SVM, K-Neighbors, LDA.**
- Comparación de métricas como precisión, recall y F1-score.

### **3. Decodificación de Datos**
- Traducción de valores categóricos a términos comprensibles usando un archivo de diccionario (`diccionarios.py`).

### **4. Interfaz Web**
- Navegación sencilla a través de páginas:
  - **Inicio:** Página principal con acceso a funcionalidades clave.
  - **Tabla:** Vista tabular de los datos con paginación.
  - **Gráficos:** Exploración visual de patrones.
  - **Modelos:** Resultados y evaluación de los modelos.

---

## **Requisitos**

### **1. Instalación**
Asegúrate de tener instalado **Python 3.8 o superior**. Luego, instala las dependencias con:

```bash
pip install -r requirements.txt
pip install flask pandas
pip install scikit-learn lightgbm catboost xgboost
pip install scipy

## Ejecuta la aplicacion:
python app.py
Abre en tu navegador: Ve a http://127.0.0.1:5000/.