import pandas as pd
import pytest
from preprocess import encode, scale

def test_preprocess_logic():
    # Create simple dummy data
    train_data = pd.DataFrame({
        'country': ['India', 'USA', 'India', 'Canada'],
        'amount': [100, 200, 150, 300],
        'age': [25, 30, 35, 40]
    })
    
    test_data = pd.DataFrame({
        'country': ['USA', 'Brazil'], # 'Brazil' is a new category
        'amount': [120, 180],
        'age': [28, 32]
    })

    cat_cols = ['country']
    num_cols = ['amount', 'age']

    
    # Ensure it handles 'Brazil' (new category) without crashing
    X_train_enc, X_test_enc, encoder = encode(train_data, test_data, cat_cols)
    
    assert X_train_enc.shape[1] == X_test_enc.shape[1], "Column mismatch after encoding"
    assert 'country_USA' in X_train_enc.columns

    # Test Scaling
    X_train_sc, X_test_sc, scaler = scale(X_train_enc, X_test_enc, num_cols)
    
    assert not X_train_sc[num_cols].isnull().values.any(), "Scaling introduced NaNs"
    assert X_train_sc['amount'].mean() < 0.00001, "Scaling did not center the mean to 0"

    print("Simple Preprocess Test Passed!")
