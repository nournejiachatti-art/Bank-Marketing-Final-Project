import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

def train_model(X_train, y_train, X_test, y_test, algo_name, params):
    mlflow.set_experiment("Bank_Marketing_Project")
    
    with mlflow.start_run(run_name=algo_name):
        if algo_name == "Random Forest":
            model = RandomForestClassifier(**params)
        elif algo_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif algo_choice == "KNN":
            model = KNeighborsClassifier(n_neighbors=params.get('n', 5))
        elif algo_choice == "SVM":
            model = SVC(kernel='linear', probability=True)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Logs
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("f1", f1_score(y_test, y_pred))
        mlflow.sklearn.log_model(model, "model")
        
        return model