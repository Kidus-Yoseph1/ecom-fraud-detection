from preprocess import preprocess, encode, scale
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (train_test_split, GridSearchCV, 
                                     StratifiedKFold, cross_val_score)

def split(data):
    X = data.drop(['class', 'user_id', 'device_id', 'signup_time', 'purchase_time'], axis=1)
    y = data['class']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2
                                                        , random_state=42,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test

def Model(X_train, y_train):
    rf = RandomForestClassifier(random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    params = {
        'n_estimators':[100,200],
        'max_depth': [10,20,None]
    }

    grid_search = GridSearchCV(estimator=rf, param_grid=params,
                               cv = skf, scoring='f1',n_jobs=-1)
    
    print("Started trainin Random Forest Classifier")
    grid_search.fit(X_train,y_train)
    print(f"Best Parameters: {grid_search.best_params_}")

    return grid_search.best_estimator_

