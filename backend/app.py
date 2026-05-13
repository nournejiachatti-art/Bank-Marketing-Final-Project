import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

app = Flask(__name__)
CORS(app)

# Ensure static folder exists
os.makedirs('static', exist_ok=True)

scaler = StandardScaler()
X_train_cols = None
current_model = None
X_test_unscaled = None
df_original = None  # Stocke le DataFrame original complet

def prepare_data():
    global scaler, X_train_cols, X_test_unscaled, df_original
    # Chargement des données
    df = pd.read_csv('../data/bank-full.csv', sep=';')
    df_original = df.copy()  # Stocke le DataFrame original complet
    
    df.replace('unknown', np.nan, inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    y = df['y'].map({'yes': 1, 'no': 0})
    X = df.drop('y', axis=1)
    X = pd.get_dummies(X, drop_first=True)
    X_train_cols = X.columns
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test_unscaled = X_test.copy()  # Keep unscaled copy for error analysis
    num_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    return X_train, X_test, y_train, y_test, X_test_unscaled


def test_stability():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Bank_Marketing_Comparison")

    seeds = [10, 42, 100, 2024, 999]
    for seed in seeds:
        with mlflow.start_run(run_name=f"Stability_Test_Seed_{seed}"):
            model = RandomForestClassifier(max_depth=10, n_estimators=100, random_state=seed)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)

            mlflow.log_param("seed", seed)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("recall", rec)
            # Log additional stability metrics here if needed

X_train, X_test, y_train, y_test, X_test_unscaled = prepare_data()

def get_error_examples(model, X_test, X_test_unscaled, y_test, df_original, n=3):
    """Identifie des exemples concrets de mal-classement avec données originales lisibles."""
    y_pred = model.predict(X_test)  # Prediction sur données scalées
    
    # Trouver les indices des erreurs
    error_mask = y_test != y_pred
    error_indices = X_test_unscaled[error_mask].index
    
    # Récupérer les données originales (vraies valeurs) pour ces erreurs
    errors_original = df_original.loc[error_indices].copy()
    errors_original['vrai_y'] = y_test[error_mask].values
    errors_original['pred_y'] = y_pred[error_mask]
    
    # Séparer Faux Positifs (prédit OUI mais c'est NON) et Faux Négatifs (prédit NON mais c'est OUI)
    fn = errors_original[errors_original['vrai_y'] == 1].head(2)  # Réel=OUI, Prédit=NON
    fp = errors_original[errors_original['vrai_y'] == 0].head(2)  # Réel=NON, Prédit=OUI
    
    samples = pd.concat([fn, fp])
    
    # Retourner seulement les colonnes pertinentes pour l'affichage
    result = []
    for idx, row in samples.iterrows():
        result.append({
            'age': int(row['age']),
            'duration': int(row['duration']),
            'balance': int(row['balance']),
            'vrai_y': int(row['vrai_y']),
            'pred_y': int(row['pred_y']),
            'error_type': 'FN' if row['vrai_y'] == 1 else 'FP'
        })
    
    return result

@app.route('/train', methods=['POST'])
def train():
    global current_model
    data = request.json
    algo = data.get('algorithm')
    params = data.get('params', {})
    print(f"🔍 RECEIVED: algo={algo}, params={params}")
    
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Bank_Marketing_Comparison")
    
    with mlflow.start_run(run_name=f"Run_{algo}"):
        try:
            # --- PERIMETRE AVANCE : Sélection des modèles ---
            if algo == "Random Forest":
                depth = int(params.get('max_depth', 10))
                print(f"📊 Random Forest: Using max_depth = {depth}")
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

            elif algo == "AdaBoost":
                try:
                    n_estimators = int(float(params.get('n_estimators', 50)))
                    learning_rate = float(params.get('learning_rate', 1.0))
                    n_estimators = max(1, n_estimators)  # Assure au moins 1 estimateur
                    learning_rate = max(0.01, learning_rate)  # Learning rate minimum
                    
                    current_model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
                    mlflow.log_params({"n_estimators": n_estimators, "learning_rate": learning_rate})
                    print(f"🔧 AdaBoost: n_estimators={n_estimators}, learning_rate={learning_rate}")
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Paramètres AdaBoost invalides: {str(e)}")

            elif algo == "XGBoost":
                try:
                    n_estimators = int(float(params.get('n_estimators', 100)))
                    max_depth = int(float(params.get('max_depth', 6)))
                    lr = float(params.get('learning_rate', 0.1))
                    
                    # Validation des plages
                    n_estimators = max(1, n_estimators)
                    max_depth = max(1, min(max_depth, 15))  # Entre 1 et 15
                    lr = max(0.001, min(lr, 1.0))  # Entre 0.001 et 1.0
                    
                    current_model = XGBClassifier(
                        n_estimators=n_estimators, 
                        max_depth=max_depth, 
                        learning_rate=lr,
                        random_state=42,
                        use_label_encoder=False,
                        eval_metric='logloss'
                    )
                    mlflow.log_params({"n_estimators": n_estimators, "max_depth": max_depth, "learning_rate": lr})
                    print(f"⚡ XGBoost: n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={lr}")
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Paramètres XGBoost invalides: {str(e)}")

            # Apprentissage
            current_model.fit(X_train, y_train)
            y_pred = current_model.predict(X_test)
            
            # Métriques communes à tous les modèles (Tâche 3)
            acc_test = accuracy_score(y_test, y_pred)
            acc_train = accuracy_score(y_train, current_model.predict(X_train)) 
            f1 = f1_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)

            # Initialisation de la réponse
            response_data = {
                "accuracy": f"{round(acc_test*100,2)}%",
                "accuracy_train": f"{round(acc_train*100,2)}%",
                "f1": round(f1,3),
                "precision": round(prec, 3),
                "recall": round(rec, 3),
                "algo": algo,
                "show_importance": False, # Par défaut, on ne montre pas l'importance
                "error_samples": []  # Initialisation
            }

            # Matrice de Confusion (Tâche 3)
            plt.figure(figsize=(5, 4))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens')
            plt.title(f"Confusion Matrix: {algo}")
            plot_path = os.path.join('static', 'matrix.png')
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            mlflow.log_artifact(plot_path)

            # Feature Importance pour Random Forest (Tâche 4)
            if algo == "Random Forest":
                response_data["show_importance"] = True
                plt.figure(figsize=(8, 5))
                importances = current_model.feature_importances_
                feature_names = X_train.columns
                indices = np.argsort(importances)[-10:]  # Top 10
                plt.barh(range(len(indices)), importances[indices], color='steelblue')
                plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
                plt.xlabel('Importance')
                plt.title(f'Feature Importance (RF, depth={params.get("max_depth", 10)})')
                importance_path = os.path.join('static', 'importance.png')
                plt.savefig(importance_path, bbox_inches='tight')
                plt.close()
                mlflow.log_artifact(importance_path)
                
                # AJOUT : Analyse des erreurs pour Random Forest (Tâche 4)
                samples = get_error_examples(current_model, X_test, X_test_unscaled, y_test, df_original)
                response_data["error_samples"] = samples

            response_data["depth_used"] = params.get('max_depth', 10) if algo == "Random Forest" else None
            return jsonify(response_data)
        except Exception as e:
            print(f"❌ ERROR DURING TRAINING: {str(e)}")
            return jsonify({"error": str(e)}), 500

    return jsonify(response_data)

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

@app.route('/matrix.png')
def get_matrix():
    return send_from_directory('static', 'matrix.png')

@app.route('/importance.png')
def get_importance():
    return send_from_directory('static', 'importance.png')

@app.route('/stability-test', methods=['POST'])
def stability_test():
    # Paramètres fixes pour l'étude de stabilité
    depth = 10
    n_trees = 100
    seeds = [10, 42, 100, 2024, 999]
    stability_results = []

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Bank_Marketing_Stability")

    for seed in seeds:
        with mlflow.start_run(run_name=f"Stability_Seed_{seed}"):
            model = RandomForestClassifier(max_depth=depth, n_estimators=n_trees, random_state=seed)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            mlflow.log_param("seed", seed)
            mlflow.log_param("max_depth", depth)
            mlflow.log_metric("accuracy", acc)
            
            stability_results.append({"seed": seed, "accuracy": round(acc * 100, 2)})

    return jsonify({
        "status": "success",
        "message": "5 runs de stabilité envoyés vers MLflow",
        "data": stability_results
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)