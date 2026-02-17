import os
import pandas as pd
import torch
from preprocess import preprocess, encode, scale
from model import split, Model, save_model_artifacts
from autoencoder import preprocess_autoencoder, TransactionAutoencoder 
from eval import evaluate

fraud_path = "Data/Fraud_Data.csv"
ip_path = "Data/IpAddress_to_Country.csv"
final_path = "Data/processed/"
os.makedirs(final_path, exist_ok=True)
full_file_path = os.path.join(final_path, "processed_data.csv")
cat_cols = ['source', 'browser', 'sex'] 
num_cols =  ['purchase_value', 'age', 'time_since_signup', 'hour_of_day', 'day_of_week', 'device_count', 'ip_count']


if __name__ == "__main__":
    # # preprocess the data
    fraud_merged = preprocess(fraud_path,ip_path,full_file_path)
    # split the data
    X_train, X_test, y_train, y_test = split(fraud_merged)
    # encode the data
    X_train_encoded, X_test_encoded, encoder = encode(X_train,X_test,cat_cols)
    # scale the data
    X_train_scaled, X_test_scaled, scaler = scale(X_train_encoded,X_test_encoded,num_cols)
    # train the Random forest  model
    model = Model(X_train_scaled,y_train)

    # evaluate the model performance
    evaluate(model,X_test_scaled,y_test)

    # and finally save the model artifacts (model, scale and encoder)for later use
    save_model_artifacts(model,scaler,encoder)


    # train the autoencoder
    fraud_df = 'Data/processed/processed_data.csv'

    # preprocess
    X_train, X_test, y_train, y_test, autoencoder_scaler = preprocess_autoencoder(fraud_df)

    # Convert & Filter (Only train on Normal transactions)
    X_train_ts = torch.tensor(X_train.values, dtype=torch.float32)
    X_train_normal = X_train_ts[torch.tensor(y_train) == 0]

    # train model
    model = TransactionAutoencoder(input_dim=X_train_normal.shape[1])
    model.train_model(X_train_normal, num_epochs=50) 

    # evaluate the model 
    X_test_ts = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_ts = torch.tensor(y_test, dtype=torch.float32)
    model.evaluate(X_test_ts, y_test_ts, percentile=92)

    # save dict states
    model.save_fraud_model(autoencoder_scaler)


