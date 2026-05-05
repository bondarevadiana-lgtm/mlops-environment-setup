import mlflow
import mlflow.sklearn
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


def get_model(model_type, **params):
    if model_type == 'nb':
        return MultinomialNB(**params)
    elif model_type == 'lr':
        return LogisticRegression(max_iter=1000, **params)
    elif model_type == 'svc':
        return LinearSVC(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def run_experiment(model_type, model_name, params):

    mlflow.set_experiment("Newsgroups_Classification_Lab1")
    
    file_path = 'data/raw/newsgroups.csv'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found!")
        return
    
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['text']) 
    
    X = df['text']
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    with mlflow.start_run(run_name=model_name):
        model = get_model(model_type, **params)
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
            ('classifier', model)
        ])
        
        mlflow.log_params(params)
        mlflow.set_tag("model_type", model_type)
        
        print(f"--- Running {model_name} ---")
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3)
        mlflow.log_metric("cv_accuracy_mean", cv_scores.mean())
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.sklearn.log_model(pipeline, "model_pipeline")
        
        os.makedirs('models', exist_ok=True)
        model_path = f'models/{model_name}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(pipeline, f)
            
        print(f"Result: Accuracy = {test_accuracy:.4f}\n")

if __name__ == "__main__":

    run_experiment('nb', 'NB_Alpha_1.0', {'alpha': 1.0})
    run_experiment('nb', 'NB_Alpha_0.5', {'alpha': 0.5})
    run_experiment('nb', 'NB_Alpha_0.1', {'alpha': 0.1})
    
    run_experiment('lr', 'LogReg_C1.0', {'C': 1.0})
    run_experiment('svc', 'LinearSVC_C0.1', {'C': 0.1})