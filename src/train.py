import os
import pandas as pd
from preprocess import preprocess, encode, scale
from model import split, Model, save_model_artifacts
from eval import evaluate

fraud_path = pd.read_csv("../Data/Fraud_Data.csv")
ip_path = pd.read_csv("../Data/IpAddress_to_Country.csv")
final_path = "../Data/processed/"
os.makedirs(final_path, exist_ok=True)
cat_cols = ['source', 'browser', 'sex','country'] 
num_cols =  ['purchase_value', 'age', 'time_since_signup', 'hour_of_day', 'day_of_week', 'device_count', 'ip_count']


if __name__ == "__main__":
    # preprocess the data
    fraud_merged = preprocess(fraud_path,ip_path,final_path)
    # split the data
    X_train, X_test, y_train, y_test = split(fraud_merged)
    # encode the data
    X_train_encoded, X_test_encoded, encoder = encode(X_train,X_test,cat_cols)
    # scale the data
    X_train_scaled, X_test_scaled, scaler = scale(X_train_encoded,X_test_encoded,num_cols)
    # train the model
    model = Model(X_train_scaled,X_test_scaled)

    # evaluate the model performance
    evaluate(model,X_test_scaled,y_test)

    # and finally save the model artifacts (model, scale and encoder)for later use
    save_model_artifacts(model,scaler,encoder)
