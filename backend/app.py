import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify
from flask_cors import CORS
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

app = Flask(__name__)
CORS(app)

scaler = StandardScaler()
X_train_cols = None
current_model = None

def prepare_data():
    global scaler, X_train_cols
    # Chargement des données
    df = pd.read_csv('../data/bank-full.csv', sep=';')
    df.replace('unknown', np.nan, inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    y = df['y'].map({'yes': 1, 'no': 0})
    X = df.drop('y', axis=1)
    X = pd.get_dummies(X, drop_first=True)
    X_train_cols = X.columns
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    num_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = prepare_data()

@app.route('/train', methods=['POST'])
def train():
    global current_model
    data = request.json
    algo = data.get('algorithm')
    params = data.get('params', {})
    
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Bank_Marketing_Comparison")
    
    with mlflow.start_run(run_name=f"Run_{algo}"):
        # --- PERIMETRE AVANCE : Sélection des modèles ---
        if algo == "Random Forest":
            depth = int(params.get('max_depth', 10))
            current_model = RandomForestClassifier(max_depth=depth, n_estimators=100, random_state=42)
            mlflow.log_param("max_depth", depth)
            
        elif algo == "KNN":
            k = int(params.get('n_neighbors', 5))
            current_model = KNeighborsClassifier(n_neighbors=k)
            mlflow.log_param("n_neighbors", k)
            
        elif algo == "Logistic Regression":
            current_model = LogisticRegression(max_iter=1000, class_weight='balanced')
            
        elif algo == "SVM":
            # SVM peut être lent sur de gros datasets, on utilise un noyau linéaire pour la démo
            current_model = SVC(kernel='linear', class_weight='balanced', probability=True)
            mlflow.log_param("kernel", "linear")

        # Apprentissage
        current_model.fit(X_train, y_train)
        y_pred = current_model.predict(X_test)
        
        # Calcul des métriques
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

        # Log MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.sklearn.log_model(current_model, "model")

        # Matrice de Confusion
        plt.figure(figsize=(5, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens')
        plt.title(f"Confusion Matrix: {algo}")
        plot_path = os.path.join('../frontend/public', 'matrix.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(plot_path)

    return jsonify({
        "accuracy": f"{round(acc*100,2)}%",
        "f1": round(f1,3),
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "algo": algo
    })

@app.route('/predict', methods=['POST'])
def predict():
    if current_model is None: return jsonify({"error": "No model trained"}), 400
    data = request.json
    try:
        input_df = pd.DataFrame(np.zeros((1, len(X_train_cols))), columns=X_train_cols)
        input_df['age'] = float(data.get('age', 30))
        input_df['balance'] = float(data.get('balance', 0))
        input_df['duration'] = float(data.get('duration', 0))
        
        num_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
        input_df[num_cols] = scaler.transform(input_df[num_cols])
        
        pred = current_model.predict(input_df)[0]
        return jsonify({"result": "OUI (Souscrira) ✅" if pred == 1 else "NON (Ne souscrira pas) ❌"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)