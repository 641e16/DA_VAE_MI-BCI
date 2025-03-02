import torch
import torch.nn as nn
from pyriemann.utils.distance import distance_riemann  # for computing distances between SPD matrices
from pyriemann.utils.base import logm, expm  # riemannian operations
import numpy as np
from pyriemann.utils.mean import mean_riemann
import copy

def ensure_spd(matrix, epsilon=1e-6):
    """Forces a matrix to be SPD by fixing negative eigenvalues"""
    # get smallest eigenvalue
    min_eig = np.linalg.eigvalsh(matrix).min()
    # if smallest eigenvalue is too small, add to diagonal
    if min_eig < epsilon:
        matrix += (epsilon - min_eig) * np.eye(matrix.shape[0])
        #print('Corrected SPD')
    return matrix

def safe_distance_riemann(A, B, epsilon=1e-6):
    """Safely compute riemannian distance between matrices"""
    try:
        # ensure both matrices are SPD
        A = ensure_spd(A)
        B = ensure_spd(B)
        return distance_riemann(A, B)
    except:
        # return large penalty if computation fails
        return 100.0

def initialize_weights(m):
    """Initialize network weights determininistically"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
        if m.bias is not None: # for batch normalization
            nn.init.zeros_(m.bias) # set bias to zero

def wrapped_normal_sampling(model, n_samples_per_class, class_data, device='cuda', scale_factor=0.01):
    """
    Generate samples by following geodesics on the SPD manifold based on the explanation in the paper:
    "Data augmentation with variational autoencoders and manifold sampling"
    """
    model.eval()
    synthetic_data = []
    synthetic_labels = []
    torch.manual_seed(11082003) # depending on seed works better/worse
    
    with torch.no_grad():
        # for each class
        for class_label, data in enumerate(class_data):
            # generate n samples for this particular class
            for i in range(n_samples_per_class):
                # first get a random real sample as reference point
                idx = torch.randint(0, len(data), (1,))
                real_sample = data[idx].to(device)
                
                # convert SPD matrix to tangent space using log map directly -> skips the encoder step and works directly on the manifold
                tangent_vector = model.to_tangent_space(real_sample)
                
                # sample a direction in tangent space (must be symmetric for SPD)
                v_m = model.unvectorize(tangent_vector)
                noise = torch.randn_like(v_m)
                noise = 0.5 * (noise + noise.transpose(-1, -2))  # ensure symmetry!!
                
                # scale the noise
                scaled_noise = noise * scale_factor
                
                # add to tangent vector and map back to SPD manifold
                perturbed_tangent = v_m + scaled_noise
                perturbed_tangent_vec = model.vectorize(perturbed_tangent)
                
                # map back to SPD manifold using exp map directly
                synthetic = model.from_tangent_space(perturbed_tangent_vec)
                synthetic_data.append(synthetic)
                synthetic_labels.append(class_label)
    
    return torch.cat(synthetic_data, dim=0), torch.tensor(synthetic_labels, device=device)


class ImprovedRiemannianVAE(nn.Module):
    def __init__(self, n_channels, latent_dim, seed=11):
        super().__init__()
        torch.manual_seed(seed)  # set seed for reproducibility
        self.n_channels = n_channels  # number of EEG channels
        self.latent_dim = latent_dim  # dimension of latent space
        # dimension of vectorized upper triangular part of SPD matrix
        self.spd_dim = (n_channels * (n_channels + 1)) // 2 # n(n+1)/2
        self.epsilon = 1e-6  # numerical stability term
        
        # Encoder network: SPD_dim -> 128 -> 64
        self.encoder = nn.Sequential(
            nn.Linear(self.spd_dim, 128),  # first dense layer
            nn.LeakyReLU(0.2),  # activation with negative slope
            nn.BatchNorm1d(128),  # normalize activations
            nn.Dropout(0.2),  # prevent overfitting
            nn.Linear(128, 64),  # second dense layer
            nn.LeakyReLU(0.2),  # activation
            nn.BatchNorm1d(64)  # final normalization
        )
        
        # Two parallel layers for mean and log-variance
        self.fc_mu = nn.Linear(64, latent_dim)  # maps to latent mean
        self.fc_logvar = nn.Linear(64, latent_dim)  # maps to latent log-variance
        
        # Decoder network: latent_dim -> 64 -> 128 -> SPD_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),  # first dense layer
            nn.LeakyReLU(0.2),  # activation
            nn.BatchNorm1d(64),  # normalize
            nn.Dropout(0.2),  # regularize
            nn.Linear(64, 128),  # second dense layer
            nn.LeakyReLU(0.2),  # activation
            nn.BatchNorm1d(128),  # normalize
            nn.Linear(128, self.spd_dim)  # final layer to SPD dimension
        )
        
        self.reference = None  # reference point for tangent space

        self.encoder.apply(initialize_weights) # applies function element wise along an axis of a tensor
        self.fc_mu.apply(initialize_weights)
        self.fc_logvar.apply(initialize_weights)
        self.decoder.apply(initialize_weights)


    
    def set_reference_point(self, data):
        """Compute reference point as riemannian mean of data"""
        # compute riemannian mean of the data
        np.random.seed(11)
        ref = mean_riemann(data.cpu().numpy())
        #ref = ensure_spd(ref, self.epsilon)  # ensure it's SPD?
        # convert to torch tensor
        self.reference = torch.tensor(
            ref,
            device=data.device,
            dtype=torch.float32
        )
    
    def to_tangent_space(self, x):
        """Map SPD matrices to tangent space"""
        np.random.seed(11)
        if self.reference is None:
            self.set_reference_point(x)  # compute reference if needed
            print('Reference point set: ', self.reference)
            
        batch_size = x.shape[0]
        tangent_vectors = []
        
        # convert to numpy for riemannian ops
        x_np = x.cpu().numpy().astype(np.float64)
        ref_np = self.reference.cpu().numpy().astype(np.float64)
        
        for i in range(batch_size):
            x_i = ensure_spd(x_np[i], self.epsilon)  # ensure SPD
            # compute inverse of reference point
            ref_inv = np.linalg.inv(ref_np + self.epsilon * np.eye(self.n_channels))
            # compute logarithmic map
            tang = logm(np.matmul(ref_inv, x_i))
            tangent_vectors.append(tang)
            
        # convert back to torch tensors
        tangent_vectors = torch.tensor(
            np.stack(tangent_vectors), 
            device=x.device,
            dtype=torch.float32
        )
        return self.vectorize(tangent_vectors)  # vectorize upper triangular part
    
    def from_tangent_space(self, v):
        """Map vectors back to SPD manifold"""
        np.random.seed(11)
        v_mat = self.unvectorize(v)  # convert vector to matrix
        batch_size = v_mat.shape[0]
        spd_matrices = []
        
        # convert to numpy
        v_np = v_mat.cpu().detach().numpy().astype(np.float64)
        ref_np = self.reference.cpu().numpy().astype(np.float64)
        
        for i in range(batch_size):
            # compute exponential map
            spd = np.matmul(
                ref_np,
                expm(v_np[i] + self.epsilon * np.eye(self.n_channels))
            )
            spd = ensure_spd(spd, self.epsilon)  # ensure SPD
            spd_matrices.append(spd)
            
        # convert back to torch tensors
        return torch.tensor(
            np.stack(spd_matrices), 
            device=v.device,
            dtype=torch.float32
        )
    
    def vectorize(self, x):
        """Convert matrices to vectors (upper triangular part)"""
        idx = torch.triu_indices(self.n_channels, self.n_channels)
        return x[:, idx[0], idx[1]]
    
    def unvectorize(self, v):
        """Convert vectors back to symmetric matrices"""
        batch_size = v.shape[0]
        # create empty matrices
        matrix = torch.zeros(
            batch_size, 
            self.n_channels, 
            self.n_channels, 
            device=v.device,
            dtype=v.dtype
        )
        # get indices for upper triangular part
        idx = torch.triu_indices(self.n_channels, self.n_channels)
        # fill upper and lower triangular parts
        matrix[:, idx[0], idx[1]] = v
        matrix[:, idx[1], idx[0]] = v  # ensure symmetry
        return matrix
    
    def encode(self, x):
        """Encode input SPD matrices to latent space"""
        h = self.to_tangent_space(x)  # map to tangent space
        h = self.encoder(h)  # encode
        return self.fc_mu(h), self.fc_logvar(h)  # return mean and log-variance
    
    def decode(self, z):
        """Decode latent vectors back to SPD matrices"""
        h = self.decoder(z)  # decode
        return self.from_tangent_space(h)  # map back to SPD manifold
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for sampling"""
        if self.training:
            std = torch.exp(0.005 * logvar) # compute standard deviation
            eps = torch.randn_like(std)  # random normal noise
            return mu + eps * std  # sample from distribution
        return mu  # just return mean during inference
    
    def forward(self, x):
        """Forward pass through VAE"""
        mu, logvar = self.encode(x)  # encode
        z = self.reparameterize(mu, logvar)  # sample
        return self.decode(z), mu, logvar  # decode and return
    
    def loss_function(self, recon_x, x, mu, logvar, beta=0.0001):
        """Compute VAE loss with riemannian distance"""
        batch_size = x.size(0)
        
        # compute reconstruction loss using riemannian distance
        riem_dist = []
        recon_np = recon_x.detach().cpu().numpy().astype(np.float64)
        x_np = x.detach().cpu().numpy().astype(np.float64)
        
        for i in range(batch_size):
            dist = safe_distance_riemann(recon_np[i], x_np[i], self.epsilon)
            riem_dist.append(dist)
        
        riem_dist = torch.tensor(riem_dist, device=x.device, dtype=torch.float32)
        recon_loss = riem_dist.mean()
        
        # compute KL divergence loss
        kl_loss = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp()
        ) / batch_size
        
        # add L2 regularization
        l2_reg = 0.0
        for param in self.parameters():
            l2_reg += torch.norm(param)
        
        # combine losses with weights
        return recon_loss + beta * kl_loss + 1e-5 * l2_reg

# training function
def train_improved_rvae(model, train_loader, epochs, device, seed=11):
    """Train the VAE and return the best model"""
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=20,
        verbose=True
    )
    
    # early stopping setup
    best_loss = float('inf')
    best_state_dict = None
    patience_counter = 0
    max_patience = 20
    
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device).float()
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            loss = model.loss_function(
                recon_batch, 
                data, 
                mu, 
                logvar,
                beta=min(0.001 * epoch/50, 0.01) # TODO: check if this is actually worth it
            )
            
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
        
        if n_batches > 0:
            avg_loss = total_loss / n_batches
            scheduler.step(avg_loss)
            
            # save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state_dict = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= max_patience:
                print(f'Early stopping at epoch {epoch}')
                print(f'Loading best model with loss: {best_loss:.4f}')
                model.load_state_dict(best_state_dict)
                break
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Average Loss = {avg_loss:.4f}, Patience = {patience_counter}')
    
    # ensure to return the best model even if we don't trigger early stopping
    if not patience_counter >= max_patience:
        model.load_state_dict(best_state_dict)
    
    return model