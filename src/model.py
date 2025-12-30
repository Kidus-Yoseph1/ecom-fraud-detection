import os
import joblib
import pandas as pd 

from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import (train_test_split, GridSearchCV, 
                                     StratifiedKFold, cross_val_score)

def split(data):
    X = data.drop(['class', 'user_id', 'device_id', 'signup_time', 'purchase_time'], axis=1)
    y = data['class']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2
                                                        , random_state=42,
                                                        stratify=y)
    print("Splitting Completed")
    return X_train, X_test, y_train, y_test

def Model(X_train, y_train):
    # rf = RandomForestClassifier(random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    mask = y_train.notna()
    X_train = X_train[mask]
    y_train = y_train[mask].astype(int)
    X_train = X_train.dropna()
    y_train = y_train[X_train.index] 
    y_train = pd.to_numeric(y_train, errors='coerce')
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    params = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [10, 20] 
    }

    grid_search = GridSearchCV(pipeline, param_grid=params,
                               cv = skf, scoring='f1',n_jobs=2, verbose=1)
    
    print("Started training Random Forest Classifier")
    print("-"* 30)
    grid_search.fit(X_train,y_train)
    print(f"Best Parameters: {grid_search.best_params_}")

    return grid_search.best_estimator_

def save_model_artifacts(model, scaler, encoder, folder='models'):
    os.makedirs(folder, exist_ok=True)

    # Save the model
    joblib.dump(model, f'{folder}/fraud_random_forest_model.pkl')
    
    # Save the scaler 
    joblib.dump(scaler, f'{folder}/scaler.pkl')

    # Save the encoder 
    joblib.dump(encoder, f'{folder}/encoder.pkl')

    print(f"All artifacts saved successfully to the '{folder}' directory.")
