import joblib
import torch
import pandas as pd
import numpy as np
from .autoencoder import TransactionAutoencoder 

class InferenceEngine:
    def __init__(self, model_dir='models/autoencoder/'):
        self.input_dim = 14 
        self.threshold = 0.031416  # Replace with your actual 92nd percentile
        
        # 1. Load Autoencoder
        self.ae_model = TransactionAutoencoder(self.input_dim)
        state_dict = torch.load(f'{model_dir}autoencoder_v1.pth', map_location='cpu')
        self.ae_model.load_state_dict(state_dict)
        self.ae_model.eval()

        # 2. Load Preprocessing Artifacts
        self.scaler = joblib.load(f'{model_dir}autoencoder_scaler.joblib')
        self.encoder = joblib.load(f'{model_dir}autoencoder_encoder.joblib')

    def _preprocess(self, data_input):
        df = pd.DataFrame([data_input]) if isinstance(data_input, dict) else data_input

        # Columns lists
        cat_cols = ['source', 'browser', 'sex'] 
        num_cols = ['purchase_value', 'age', 'time_since_signup', 'hour_of_day', 
                    'day_of_week', 'device_count', 'ip_count']

  
        enc_arr = self.encoder.transform(df[cat_cols])

        # Get Numerical Array 
        num_arr = df[num_cols].values

        # Combine them into a (1, 14) array
        final_array = np.hstack([enc_arr, num_arr])


        try:
            features_scaled = self.scaler.transform(final_array)
        except ValueError: 
            # provide all 14 names in the correct order.
            df_final = pd.DataFrame(final_array, columns=self.scaler.feature_names_in_)
            features_scaled = self.scaler.transform(df_final)
        
        return torch.FloatTensor(features_scaled)

    def predict(self, data_input):
        features_tensor = self._preprocess(data_input)

        with torch.no_grad():
            reconstruction = self.ae_model(features_tensor)
            mse_values = torch.mean((features_tensor - reconstruction) ** 2, dim=1).cpu().numpy()

        results = []
        for mse in mse_values:
            if mse <= self.threshold:
                status = "Approved"
            elif mse > (self.threshold * 2.0):
                status = "Fraud"
            else:
                status = "Suspicious"
            
            results.append({
                "status": status,
                "anomaly_score": round(float(mse), 6),
                "is_anomaly": bool(mse > self.threshold)
            })

        return results[0] if len(results) == 1 else results
    
