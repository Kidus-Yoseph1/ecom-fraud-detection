import pandas as pd 
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess(credit_path, fraud_path, ip_path, final_path):
    ## load all the csv data   
    credit_df = pd.read_csv(credit_path)
    fraud_df = pd.read_csv(fraud_path)
    ip_df = pd.read_csv(ip_path)

    # clean data 
    # deal with data types and duplicates
    
    # convert timestamps to datetime objects
    fraud_df['signup_time'] = pd.to_datetime(fraud_df['signup_time'])
    fraud_df['purchase_time'] = pd.to_datetime(fraud_df['purchase_time'])

    # remove duplicates
    fraud_df = fraud_df.drop_duplicates()
    credit_df = credit_df.drop_duplicates()

    print("updated data types for fraud data")
    print(fraud_df[['purchase_time', 'signup_time']].dtypes)


    # ip to country mapping
    # ensure ip adresses are integers for comparision

    fraud_df['ip_address'] = fraud_df['ip_address'].astype(int)
    ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].astype(int)
    ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].astype(int)

    # dataframes must be  sorted by the key
    fraud_df = fraud_df.sort_values('ip_address')
    ip_df = ip_df.sort_values('lower_bound_ip_address')

    # perform range based merge
    fraud_merged  = pd.merge_asof(
        fraud_df,
        ip_df,
        left_on='ip_address',
        right_on='lower_bound_ip_address'
    ) 

    """
    check if the actually falls within the upper bound of the range,
    if its outside the range, the country should be 'unknown'
    """

    fraud_merged.loc[fraud_merged['ip_address'] > fraud_merged['upper_bound_ip_address'], 'country'] = 'unknown'
    # fill the remaining NANs with "unknown"
    fraud_merged['cuntry'] = fraud_merged['country'].fillna('Unknown')
    
    # save the merged data to a csv 
    fraud_merged.to_csv(final_path)
    return fraud_merged

def encode(X_train, X_test, cat_cols: list):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
    # fit transform training data
    X_train_encode = encoder.fit_transform(X_train[cat_cols])
    # only transform on test data 
    X_test_encode = encoder.transform(X_test[cat_cols])

    encoded_cols = encoder.get_feature_names_out(cat_cols)

    # Convert back to DataFrames
    X_train_encoded_df = pd.DataFrame(X_train_encode, columns=encoded_cols, index=X_train.index)
    X_test_encoded_df = pd.DataFrame(X_test_encode, columns=encoded_cols, index=X_test.index)

    # Drop original categorical columns and join encoded ones
    X_train_encoded = X_train.drop(columns=cat_cols).join(X_train_encoded_df)
    X_test_encoded = X_test.drop(columns=cat_cols).join(X_test_encoded_df)

    return X_train_encoded, X_test_encoded, encoder

def scale(X_train, X_test, num_cols: list):
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])

    X_test_scaled = X_test.copy()
    X_train_scaled[num_cols] = scaler.transform(X_test[num_cols])

    return X_train_scaled, X_test_scaled, scaler

