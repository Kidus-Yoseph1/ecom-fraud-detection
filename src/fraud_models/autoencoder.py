import os 
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import joblib

def preprocess_autoencoder(file_path):
    # load data
    df = pd.read_csv(file_path)
    fraud_df = df.copy()
    labels = fraud_df['class'].values
    fraud_df = fraud_df.drop(['country', 'class','user_id', 'signup_time', 'purchase_time'], axis=1)

    ## columns to train with 
    cat_cols = ['source', 'browser', 'sex'] 
    num_cols =  ['purchase_value', 'age', 'time_since_signup', 'hour_of_day',
                    'day_of_week', 'device_count', 'ip_count']

    # Split
    X_train, X_test,  y_train, y_test= train_test_split(df,labels, test_size=0.2, random_state=42)
    
    # Encode Categorical
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
    train_enc_arr = encoder.fit_transform(X_train[cat_cols])
    
    # Get names AFTER fitting
    encoded_cols = encoder.get_feature_names_out(cat_cols)
    
    # Create DataFrames
    X_train_cat = pd.DataFrame(train_enc_arr, columns=encoded_cols, index=X_train.index)
    X_test_cat = pd.DataFrame(encoder.transform(X_test[cat_cols]), columns=encoded_cols, index=X_test.index)

    # Scale Numerical
    scaler = StandardScaler()
    X_train_num = pd.DataFrame(scaler.fit_transform(X_train[num_cols]), columns=num_cols, index=X_train.index)
    X_test_num = pd.DataFrame(scaler.transform(X_test[num_cols]), columns=num_cols, index=X_test.index)

    # Combine
    X_train_final = pd.concat([X_train_cat, X_train_num], axis=1)
    X_test_final = pd.concat([X_test_cat, X_test_num], axis=1)

    print("-" * 20)
    print("Encoding & Scaling Completed")
    print("-" * 20)

    return X_train_final, X_test_final,  y_train, y_test, scaler

class TransactionAutoencoder(nn.Module):
    """
    Autoencoder for Unsupervised Fraud Detection.
    
    The model learns to compress and reconstruct 'Normal' transactions.
    Fraudulent transactions will result in a higher reconstruction error (MSE),
    acting as an anomaly detection mechanism.
    """
   
    def __init__(self, input_dim):
        super(TransactionAutoencoder, self).__init__()
        
        # Encoder: Reducing dimensionality to find the latent representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8), # Latent Bottleneck
            nn.ReLU()
        )
        
        # Decoder: Attempting to reconstruct the original input from the bottleneck
        self.decoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        """
        Standard forward pass.
        Returns the reconstructed version of the input x.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

    def train_model(self, X_train_normal, num_epochs=100, batch_size=128, lr=0.001):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        num_epochs = 100
        batch_size = 128
        train_data = X_train_normal

        print(f"Starting training on {len(train_data)} normal transactions...")

        for epoch in range(num_epochs):
            self.train()
            epoch_loss = 0
            
            # Shuffle indices manually for better training stability
            permutation = torch.randperm(train_data.size(0))
            
            for i in range(0, len(train_data), batch_size):
                indices = permutation[i : i + batch_size]
                batch = train_data[indices]
                
                # Forward pass
                output = self(batch)
                loss = criterion(output, batch) # Reconstruction error
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()

            avg_loss = epoch_loss / (len(train_data) / batch_size)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Avg Reconstruction Loss: {avg_loss:.6f}')

        print("Training Complete.")


    def evaluate(self, X_test_ts, y_test_ts, percentile = 92):
        with torch.no_grad():
            # Pass the test tensor through the model
            predictions_ts = self(X_test_ts)

            # Calculate MSE per row (dim=1)
            # Higher error = higher likelihood of fraud
            mse_per_row = torch.mean((X_test_ts - predictions_ts)**2, dim=1)

            # Convert to a numpy array for classification
            reconstruction_errors = mse_per_row.cpu().numpy()
        
        # Use the 92th percentile as a threshold (can be adjusted)
        threshold = np.percentile(reconstruction_errors, percentile)
        print(f"Threshold for anomaly detection: {threshold:.6f}")

        # Any error above threshold is classified as Fraud (1), below as Normal (0)
        y_pred = (reconstruction_errors > threshold).astype(int)

        # Ensure y_test is a numpy array
        y_test_np = y_test_ts.cpu().numpy()
        
        # Print the final metrics
        print("--- Classification Report ---")
        print(classification_report(y_test_np, y_pred, target_names=['Normal', 'Fraud']))

    def save_fraud_model(self, scaler, directory='models/autoencoder', filename='autoencoder_v1.pth'):
        """
        Saves the PyTorch model state_dict to the specified directory.
        """
        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

        # Define the full path
        filepath = os.path.join(directory, filename)

        # Save the state_dict
        torch.save(self.state_dict(), filepath)
        print(f"Model successfully saved to: {filepath}")
        
        scaler_path = os.path.join(directory, 'autoencoder_scaler.joblib')
        joblib.dump(scaler, scaler_path)
        print(f"Scaler successfully saved to: {scaler_path}")


        