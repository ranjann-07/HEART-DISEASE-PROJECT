import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

def train_and_evaluate_models():
    print("Loading cleaned dataset...")
    df = pd.read_csv('heart_cleaned.csv')
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for later use in prediction
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler saved as 'scaler.pkl'")
    
    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Support Vector Machine": SVC(random_state=42, probability=True)
    }
    
    best_model = None
    best_accuracy = 0
    best_model_name = ""
    
    print("\n--- Training and Evaluating Models ---")
    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"\n{name}:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        # Update best model based on accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name
            
    print(f"\nBest Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")
    
    # Save the best model
    joblib.dump(best_model, 'heart_disease_model.pkl')
    print(f"Best model ({best_model_name}) saved as 'heart_disease_model.pkl'.")

if __name__ == "__main__":
    train_and_evaluate_models()
