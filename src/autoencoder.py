import torch
import torch.nn as nn

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
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8), # Latent Bottleneck
            nn.ReLU()
        )
        
        # Decoder: Attempting to reconstruct the original input from the bottleneck
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid() # Assumes input data is scaled between 0 and 1
        )

    def forward(self, x):
        """
        Standard forward pass.
        Returns the reconstructed version of the input x.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_anomaly_score(self, x):
        """
        Helper method for the API.
        Calculates the Mean Squared Error between input and reconstruction.
        """
        self.eval()
        with torch.no_grad():
            reconstruction = self.forward(x)
            # Calculate MSE per sample
            mse = torch.mean((x - reconstruction) ** 2, dim=1)
        return mse