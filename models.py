import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

class ModelTrainer:
    def __init__(self, data_path):
        """
        Initialize the ModelTrainer with the path to the data file.
        """
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = []

    def load_data(self):
        """
        Load the dataset from the given path.
        """
        self.data = pd.read_csv(self.data_path)
        print(f"Data loaded: {self.data.shape[0]} rows, {self.data.shape[1]} columns")

    def preprocess_data(self):
        """
        Preprocess the data: encode categorical variables and scale numerical features.
        """
        self.X = self.data.drop(columns="Target")
        self.y = self.data["Target"]

        # Encode categorical variables
        le = LabelEncoder()
        for col in self.X.columns:
            if self.X[col].dtype == "object":
                self.X[col] = le.fit_transform(self.X[col])

        # Scale numerical features
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

        # Split the data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def create_models(self):
        """
        Initialize the models to be trained.
        """
        self.models = {
            "Logistic Regression": LogisticRegression(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "LightGBM": LGBMClassifier(random_state=42),
            "CatBoost": CatBoostClassifier(silent=True, random_state=42)
        }


    def train_and_evaluate(self):
        """
        Train and evaluate each model, storing the results.
        """
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(self.x_train, self.y_train)

            # Predict on test set
            y_pred = model.predict(self.x_test)

            # Evaluate performance
            accuracy = accuracy_score(self.y_test, y_pred)
            class_report = classification_report(self.y_test, y_pred, output_dict=True)

            # Store results
            self.results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': class_report['weighted avg']['precision'],
                'Recall': class_report['weighted avg']['recall'],
                'F1-Score': class_report['weighted avg']['f1-score']
            })
    def generate_accuracy_comparison_chart(self):
        """
        Genera un gráfico interactivo que compara la precisión de los modelos.
        """
        # Asegurarse de que los resultados sean un DataFrame
        if not isinstance(self.results, pd.DataFrame):
            self.results = pd.DataFrame(self.results)

        # Ordenar por precisión
        self.results = self.results.sort_values(by="Accuracy", ascending=False)

        # Crear un gráfico de barras
        color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        fig = px.bar(
            self.results,
            y="Model",
            x="Accuracy",
            color="Model",
            color_discrete_sequence=color_palette,
            title="Comparación de Precisión de Modelos",
            labels={"Accuracy": "Puntaje de Precisión", "Model": "Modelos de Aprendizaje"},
        )

        # Personalizar diseño
        fig.update_layout(
            xaxis_title="Puntaje de Precisión",
            yaxis_title="Modelos de Aprendizaje",
            showlegend=False,
            template="plotly_white",
            width=1000,
            height=600,
        )
        return fig.to_html(full_html=False)

    




    def get_results(self):
        """
        Return the evaluation results as a DataFrame.
        """
        return pd.DataFrame(self.results)

# Example usage:
# trainer = ModelTrainer(data_path=r'C:\Users\matii\OneDrive\Escritorio\Proyecto_final\data\Data.csv')
# trainer.load_data()
# trainer.preprocess_data()
# trainer.create_models()
# trainer.train_and_evaluate()
# results_df = trainer.get_results()
# print(results_df)
