import os
import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def prepare_data():
    # Chargement des données
    df = pd.read_csv('../data/bank-full.csv', sep=';')
    df.replace('unknown', np.nan, inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    y = df['y'].map({'yes': 1, 'no': 0})
    X = df.drop('y', axis=1)
    X = pd.get_dummies(X, drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    num_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    return X_train, X_test, y_train, y_test

def run_stability_study(X_train, X_test, y_train, y_test):
    # Liste de 5 graines aléatoires différentes
    seeds = [10, 42, 100, 123, 2024]
    
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Bank_Marketing_Stability")

    for seed in seeds:
        # On lance un run spécifique pour chaque graine
        with mlflow.start_run(run_name=f"Stability_Seed_{seed}"):
            # Paramétrage fixe (celui que nous avons jugé optimal)
            depth = 10
            n_trees = 100
            
            model = RandomForestClassifier(
                max_depth=depth, 
                n_estimators=n_trees, 
                random_state=seed
            )
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            
            # Log des paramètres et de la métrique
            mlflow.log_param("random_state_seed", seed)
            mlflow.log_param("max_depth", depth)
            mlflow.log_metric("accuracy", acc)
            
            print(f"Run terminé pour seed {seed} - Accuracy: {acc:.4f}")

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = prepare_data()
    run_stability_study(X_train, X_test, y_train, y_test)