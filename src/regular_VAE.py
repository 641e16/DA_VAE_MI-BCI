import torch
from torch import nn

class BasicVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        """Basic Variational Autoencoder (VAE)."""
        super(BasicVAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim * 2)  # Output both mu and logvar
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # Output values in [0, 1]
        )

    def forward(self, x):
        """Forward pass through the VAE."""
        # Encode to latent space
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)  # split into mean and log variance
        
        # peparameterization trick
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        
        # decode back to input space
        x_recon = self.decoder(z)
        
        return x_recon, mu, logvar

    def loss_function(self, x, x_recon, mu, logvar):
        """Compute the VAE loss function."""

        # mse reconstruction loss
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
        
        # KL Divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + kl_loss

    def train_vae(self, model, train_loader, epochs=10, lr=1e-3):
        """Train the VAE model. very basic set up."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            model.train()
            
            total_loss = 0
            for batch_idx, (x_batch, _) in enumerate(train_loader):
                x_batch = x_batch.to(device).view(x_batch.size(0), -1)  # Flatten input
                
                optimizer.zero_grad()
                
                # Forward pass
                x_recon, mu, logvar = model(x_batch)
                
                # Compute loss
                loss = model.loss_function(x_batch, x_recon, mu, logvar)
                
                # backward pass and optimization
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader.dataset):.4f}")
