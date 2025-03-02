from moabb.datasets import BNCI2014_004
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from sklearn.pipeline import make_pipeline
from pyriemann.estimation import Covariances # Riemannian covariance estimation -> convert the preprocessed EEG signals into covariance matrices (SPD matrices)
from pyriemann.tangentspace import TangentSpace # Manifold projection
from braindecode.preprocessing import exponential_moving_standardize # EEG normalization
from sklearn.model_selection import KFold
from moabb.paradigms import MotorImagery # MI paradigm definition
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader # Batch processing
from torch.utils.data import TensorDataset # Dataset wrapping tensors

from pyriemann.classification import MDM # Riemannian classifier
from sklearn.metrics import balanced_accuracy_score # Evaluation metric
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin # Custom transformer
import matplotlib.pyplot as plt
from numpy.linalg import eigvalsh
from regular_VAE import BasicVAE
from trial_2_vae import ImprovedRiemannianVAE, train_improved_rvae, safe_distance_riemann, wrapped_normal_sampling
from sklearn.manifold import TSNE
import seaborn as sns
from enum import Enum
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd



class VAEType(Enum):
    REGULAR = "regular"
    RIEMANNIAN = "riemannian"


def visualize_latent_space_tsne(model, X_cov_tensor, y_tensor, device, perplexity=7, n_components=2):
    """Visualize latent space using t-SNE to capture relationships across all dimensions"""
    model.eval()
    
    # move data to device
    X = X_cov_tensor.to(device)
    y = y_tensor.to(device)
    
    # get latent representations
    with torch.no_grad():
        mu, logvar = model.encode(X)
        z = model.reparameterize(mu, logvar)
        
        # Move to CPU for sklearn
        latent_vecs = z.cpu().numpy()
        labels = y.cpu().numpy()
    
    # apply t-SNE to reduce to either 2D or3D
    print(f"Applying t-SNE to reduce from {latent_vecs.shape[1]} to {n_components} dimensions...")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=11)
    latent_tsne = tsne.fit_transform(latent_vecs)
    
    # for easier plotting with seaborn
    df = pd.DataFrame()
    df['x'] = latent_tsne[:, 0]
    df['y'] = latent_tsne[:, 1]
    if n_components == 3:
        df['z'] = latent_tsne[:, 2]
    df['class'] = labels
    
    
    plt.figure(figsize=(10, 8))
    
    if n_components == 2:
        # 2D plot with styling
        plt.title(f"t-SNE Visualization of {latent_vecs.shape[1]}D Latent Space")
        sns.scatterplot(
            x='x', y='y', 
            hue='class', 
            #palette=['#FF9999', '#9999FF'],
            data=df,
            alpha=0.7,
            s=50,
            edgecolor='none'
        ) 
    else:  # 3D plot
        ax = plt.figure(figsize=(12, 10)).add_subplot(111, projection='3d')
        for c in np.unique(labels):
            mask = labels == c
            ax.scatter(
                latent_tsne[mask, 0], 
                latent_tsne[mask, 1], 
                latent_tsne[mask, 2],
                label=f"Class {c}",
                alpha=0.7,
                s=50
            )
        ax.set_title(f"3D t-SNE Visualization of {latent_vecs.shape[1]}D Latent Space")
        ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # try different perplexity values to ensure robustness
    if n_components == 2:
        plt.figure(figsize=(15, 10))
        perplexities = [5, 15, 30, 50]
        for i, perp in enumerate(perplexities):
            tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
            latent_tsne = tsne.fit_transform(latent_vecs)
            
            plt.subplot(2, 2, i+1)
            for c in np.unique(labels):
                mask = labels == c
                plt.scatter(
                    latent_tsne[mask, 0],
                    latent_tsne[mask, 1],
                    label=f"Class {c}",
                    alpha=0.7,
                    s=30
                )
            plt.title(f"Perplexity = {perp}")
            if i == 0:
                plt.legend()
        
        plt.suptitle("t-SNE with Different Perplexity Values")
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()


def prepare_data_for_vae(X_cov_tensor, y_tensor, train_idx, vae_type=VAEType.REGULAR):
    if vae_type == VAEType.REGULAR:
        # flatten and normalize
        X_cov_flattened = X_cov_tensor.reshape(X_cov_tensor.shape[0], -1)
        X_cov_normalized = (X_cov_flattened - X_cov_flattened.min()) / (
            X_cov_flattened.max() - X_cov_flattened.min()
        )
        train_data = X_cov_normalized[train_idx]
    else:
        # for Riemannian VAE, use the covariance matrices directly
        # and make sure we preserve the 3D structure!! (batch_size, n_channels, n_channels)
        train_data = X_cov_tensor[train_idx]
        
        print(f"Riemannian VAE input shape: {train_data.shape}")
    
    generator = torch.Generator()
    generator.manual_seed(11)
    train_dataset = TensorDataset(train_data, y_tensor[train_idx])
    return DataLoader(train_dataset, batch_size=32, shuffle=True, generator=generator)


def generate_synthetic_data(model, n_samples, latent_dim, vae_type, device='cuda'):
    with torch.no_grad():
        z_samples = torch.randn(n_samples, latent_dim).to(device)
        
        if vae_type == VAEType.REGULAR:
            synthetic_data = model.decoder(z_samples)
            # reshape back to original size
            return synthetic_data.reshape(-1, 3, 3)
        else:
            synthetic_data = model.decode(z_samples)
            return synthetic_data

def run_vae_pipeline(X_cov_tensor, y_tensor, train_idx, vae_type=VAEType.REGULAR, 
                    n_channels=3, latent_dim=16, epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # prepare data based on VAE type
    train_loader = prepare_data_for_vae(X_cov_tensor, y_tensor, train_idx, vae_type)
    
    # setup regular VAE model
    input_dim = X_cov_tensor.size(1) * X_cov_tensor.size(2) if vae_type == VAEType.REGULAR else None
    #model = setup_vae_model(vae_type, n_channels, latent_dim, input_dim, device)
    
    # train model
    if vae_type == VAEType.REGULAR:
        #model.train_vae(model=model, train_loader=train_loader, epochs=epochs)
        pass
    else:
        model = ImprovedRiemannianVAE(n_channels=3, latent_dim=16).to(device)
        train_improved_rvae(model, train_loader, epochs=800, device=device)
    return model
 
def plot_EEG(X, trial=0, channels=None, title="EEG Signal", sampling_rate=250): # TODO: change implementation since plotting y axis wrong
    """EEG plotting function with statistical properties."""

    # select the trial
    eeg_data = X[trial]  # shape: (n_channels, n_samples)

    # select specific channels if provided (otherwise plot all)
    if channels is not None:
        eeg_data = eeg_data[channels]

    #time axis for plotting
    n_samples = eeg_data.shape[1]
    time_axis = np.arange(n_samples) / sampling_rate  # convert samples to seconds

    # plot EEG data
    plt.figure(figsize=(12, 4))
    for i, channel_data in enumerate(eeg_data):
        plt.plot(time_axis, channel_data + i * 10, label=f"Channel {i+1}")  # offset for visibility
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (μV)")
    plt.legend(loc="upper right")
    plt.show()

    # statistical properties
    print(f"Trial {trial} Statistical Properties:")
    print(f"Mean: {X[trial].mean():.2f} μV")
    print(f"Std Dev: {X[trial].std():.2f} μV")
    print(f"Range: [{X[trial].min():.2f}, {X[trial].max():.2f}] μV")

def inspect_pipeline(preprocess_pipe, X, trial=0, channels=None, sampling_rate=250):
    """Inspect changes to EEG data at each stage of the pipeline."""

    # iterate through each step in the pipeline except the last one (Covariances)
    for step_name, step in preprocess_pipe.steps[:-1]:
        print(f"Stage: {step_name}")
        
        # apply the current step's transformation
        X = step.fit_transform(X)
        print(f"X shape after {step_name}: {X.shape}")
        # plot the EEG data for the specified trial and channels
        plot_EEG(X, trial=trial, channels=channels, title=f"EEG signals After {step_name}", sampling_rate=sampling_rate)
    
    # apply the final step (Covariance estimation) and return SPD matrices
    print("Stage: Covariance Estimation")
    X_cov = preprocess_pipe.steps[-1][1].fit_transform(X)
    print(f"X_cov shape: {X_cov.shape}")
    return X_cov


class MicrovoltsScaler(BaseEstimator, TransformerMixin):
    """Critical for EEG signal scaling:
    - Converts volts to microvolts (standard EEG unit)
    - Maintains numerical stability in covariance calculations?
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X * 1e6 # Multiply by 1e6 to convert V to μV

class EMSTransformer(BaseEstimator, TransformerMixin):
    """Exponential Moving Standardization:
    - Mitigates non-stationarity in EEG signals
    - Reduces baseline drift and electrode polarization effects
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.array([
            exponential_moving_standardize(
                trial.T.astype(np.float64),  # requires double precision for stability for some reason
                factor_new=0.01,  # controls adaptation rate (1% new info)
                init_block_size=1000,  # initial normalization window
                eps=1e-4  # prevents division by zero in normalization
            ).T
            for trial in X
        ])

def verify_SPD_properties(X_cov):  
    """
    Verify SPD properties for covariance matrices.
    Check for symmetry and positive definiteness.
    """
    # get how many cvariance matrices we have
    n_trials = X_cov.shape[0]
    print(f"Number of trials: {n_trials}")

    n_not_symmetric = 0
    n_not_positive_definite = 0
    for cov_trial in X_cov:
        # Symmetry check
        symm_error = np.abs(cov_trial - cov_trial.T).max()
        if symm_error > 1e-10:
            #print(f"Symmetry error: {symm_error:.2e}")
            #print(f"Covariance matrix {cov_trial} is not symmetric!")
            n_not_symmetric += 1
        
        # Eigenvalue check
        eigvals = eigvalsh(cov_trial)
        if np.any(eigvals <= 0):
            #print(f"Eigenvalues: {eigvals}")
            #print(f"Covariance matrix {cov_trial} is not SPD!")
            n_not_positive_definite += 1
    
    print(f"Number of non-symmetric matrices: {n_not_symmetric}")
    print(f"Number of not positive definite matrices: {n_not_positive_definite}")


preprocess_pipe = make_pipeline(
    MicrovoltsScaler(),       # 1. Unit conversion
    EMSTransformer(),         # 2. Temporal normalization
    Covariances(estimator='lwf')  # 3. Covariance estimation using Ledoit-Wolf shrinkage for SPD garantuee
)

dataset = BNCI2014_004()
subjects = dataset.subject_list[:1]  # Ajust to select a subset of subjects -> 1 for now


# paradigm definition controls temporal and spectral filtering
paradigm = MotorImagery(
    n_classes=2,        # binary hand movement imagery
    fmin=8, fmax=30,    # mu/beta rhythm range (motor-related)
    tmin=3.0,           # start 3s after trial start (post-cue)
    tmax=5.0            # extract 2s of active MI period (5 - 3 = 2s)
)

def generate_synthetic_data_per_class(model, X_orig, y_orig, n_samples_per_class, latent_dim, device='cuda', seed=11):
    """Generate synthetic data with balanced class distribution"""
    torch.manual_seed(seed)
    model.eval()
    
    # split original data by class
    class_0_idx = (y_orig == 0)
    class_1_idx = (y_orig == 1)
    
    X_class_0 = X_orig[class_0_idx]
    X_class_1 = X_orig[class_1_idx]
    
    synthetic_data = []
    synthetic_labels = []

    # pre-generate all random indices
    indices_class_0 = torch.randperm(len(X_class_0))[:n_samples_per_class] # random permutation of indices
    indices_class_1 = torch.randperm(len(X_class_1))[:n_samples_per_class]
    
    
    with torch.no_grad():
        # Generate for class 0
        for idx in indices_class_0:
            # sample real data point from class 0
            real_sample = X_class_0[idx:idx+1].to(device)

            
            # encode and add noise in latent space
            # logvar represent the variance of the latent distribution in log space
            mu, logvar = model.encode(real_sample)

            # use fixed random state for noise
            torch.manual_seed(idx) # different seed for each sample but reproducible
            std = torch.exp(logvar)
            z = mu + std * torch.randn_like(std)
            synthetic = model.decode(z)
            
            '''
            # just use the mean, no noise -> doesn't help
            mu, _ = model.encode(real_sample)
            synthetic = model.decode(mu)  # decode directly from mu
            '''
            synthetic_data.append(synthetic)
            synthetic_labels.append(0)
            
        # Generate for class 1
        for _ in range(n_samples_per_class):
            # sample real data point from class 1
            real_sample = X_class_1[idx:idx+1].to(device)
            
            # encode and add noise in latent space
            mu, logvar = model.encode(real_sample)

            torch.manual_seed(idx + n_samples_per_class) # different seeds for second class
            std = torch.exp(logvar)
            z = mu + std * torch.randn_like(std) # reparametrization trick

            synthetic = model.decode(z)
            
            '''
            # just use the mean, no noise -> less diverse samples
            mu, _ = model.encode(real_sample)
            synthetic = model.decode(mu)  # decode directly from mu
            '''
            synthetic_data.append(synthetic)
            synthetic_labels.append(1)
    
    # stack all synthetic data
    synthetic_data = torch.cat(synthetic_data, dim=0)
    synthetic_labels = torch.tensor(synthetic_labels, device=device)
    
    return synthetic_data, synthetic_labels


def visualize_latent_space(model, X_cov_tensor, y_tensor, device):
    """Extract and visualize latent representations"""
    model.eval()
    
    X = X_cov_tensor.to(device)
    y = y_tensor.to(device)
    
    # get latent representations
    with torch.no_grad():
        mu, logvar = model.encode(X)
        z = model.reparameterize(mu, logvar)
        
        # Move to CPU for plotting
        latent_vecs = z.cpu().numpy()
        labels = y.cpu().numpy()
    
    # plot first two dimensions
    plt.figure(figsize=(10, 8))
    
    # plot points colored by class
    for class_id in [0, 1]:
        mask = labels == class_id
        plt.scatter(
            latent_vecs[mask, 0], 
            latent_vecs[mask, 1],
            alpha=0.6,
            label=f"Class {class_id}"
        )
    
    plt.title("Latent Space Visualization")
    plt.xlabel("First Latent Dimension")
    plt.ylabel("Second Latent Dimension")
    plt.legend()
    plt.colorbar()
    plt.show()


def evaluate_with_cv(X_cov, y_numeric, device, n_splits=5, seed=11):
    """Run k-fold cross validation with both vanilla and augmented data"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # store results
    acc_no_aug_list = []
    acc_aug_list = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_cov)):
        print(f"\n=== Fold {fold+1}/{n_splits} ===")
        
        # convert to tensors
        X_cov_tensor = torch.tensor(X_cov, dtype=torch.float32)
        y_tensor = torch.tensor(y_numeric, dtype=torch.long)
        
        # train VAE
        model = run_vae_pipeline(
            X_cov_tensor=X_cov_tensor,
            y_tensor=y_tensor,
            train_idx=train_idx,
            vae_type=VAEType.RIEMANNIAN,
            n_channels=3,
            latent_dim=16,
            epochs=800
        )
        
        # baseline classification
        X_train_no_aug = X_cov[train_idx]
        X_test = X_cov[test_idx]
        y_train = y_numeric[train_idx]
        y_test = y_numeric[test_idx]
        
        clf_no_aug = MDM(metric='riemann')
        clf_no_aug.fit(X_train_no_aug, y_train)
        y_pred_no_aug = clf_no_aug.predict(X_test)
        acc_no_aug = balanced_accuracy_score(y_test, y_pred_no_aug)
        acc_no_aug_list.append(acc_no_aug)
        
        # augmented classification
        synthetic_data, synthetic_labels = generate_synthetic_data_per_class(
            model=model,
            X_orig=torch.tensor(X_train_no_aug, dtype=torch.float32).to(device),
            y_orig=torch.tensor(y_train, dtype=torch.long).to(device),
            n_samples_per_class=1000,
            latent_dim=16,
            device=device,
            seed=seed
        )
        
        X_train_aug = np.concatenate([X_train_no_aug, synthetic_data.cpu().numpy()])
        y_train_aug = np.concatenate([y_train, synthetic_labels.cpu().numpy()])
        
        clf_aug = MDM(metric='riemann')
        clf_aug.fit(X_train_aug, y_train_aug)
        y_pred_aug = clf_aug.predict(X_test)
        acc_aug = balanced_accuracy_score(y_test, y_pred_aug)
        acc_aug_list.append(acc_aug)
        
        print(f"Fold {fold+1} No Aug Accuracy: {acc_no_aug:.3f}")
        print(f"Fold {fold+1} Aug Accuracy: {acc_aug:.3f}")
    
    visualize_latent_space(model, X_cov_tensor, y_tensor, device)

    # compute statistics
    print("\n=== Final Results ===")
    print(f"No Augmentation: {np.mean(acc_no_aug_list):.3f} ± {np.std(acc_no_aug_list):.3f}")
    print(f"With Augmentation: {np.mean(acc_aug_list):.3f} ± {np.std(acc_aug_list):.3f}")
    
    return acc_no_aug_list, acc_aug_list

def evaluate_with_fixed_vae(X_cov, y_numeric, device, n_splits=5, seed=11):
    """Train VAE once, then do CV on the classifier"""
    # first, train the VAE on all data
    X_cov_tensor = torch.tensor(X_cov, dtype=torch.float32)
    y_tensor = torch.tensor(y_numeric, dtype=torch.long)
    
    model = run_vae_pipeline(
        X_cov_tensor=X_cov_tensor,
        y_tensor=y_tensor,
        train_idx=np.arange(len(X_cov)),  # use all data
        vae_type=VAEType.RIEMANNIAN,
        n_channels=3,
        latent_dim=16,
        epochs=800
    )

    test_dataset = TensorDataset(X_cov_tensor, y_tensor)
    generator = torch.Generator()
    generator.manual_seed(seed)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, generator=generator)
    
    
    # now do cv for classification
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    acc_no_aug_list = []
    acc_aug_list = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_cov)):
        print(f"\n=== Fold {fold+1}/{n_splits} ===")
        
        # baseline classification
        X_train = X_cov[train_idx]
        X_test = X_cov[test_idx]
        y_train = y_numeric[train_idx]
        y_test = y_numeric[test_idx]
        
        # no augmentation
        clf_no_aug = MDM(metric='riemann')
        clf_no_aug.fit(X_train, y_train)
        acc_no_aug = balanced_accuracy_score(y_test, clf_no_aug.predict(X_test))
        acc_no_aug_list.append(acc_no_aug)
        
        # with augmentation
                # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
        
        # Split training data by class (repeated for now)
        X_class_0 = X_train_tensor[y_train_tensor == 0]
        X_class_1 = X_train_tensor[y_train_tensor == 1]
        
        # generate synthetic training data only
        synthetic_data, synthetic_labels = wrapped_normal_sampling(
            model=model,
            n_samples_per_class=500,
            class_data=[X_class_0, X_class_1],
            device=device,
            scale_factor=0.01
        )
        
        X_train_aug = np.concatenate([X_train, synthetic_data.cpu().numpy()])
        y_train_aug = np.concatenate([y_train, synthetic_labels.cpu().numpy()])
        
        # number of samples
        print(f"Number of samples: {len(X_train)} (original) vs {len(X_train_aug)} (augmented)")

        clf_aug = MDM(metric='riemann')
        clf_aug.fit(X_train_aug, y_train_aug)
        acc_aug = balanced_accuracy_score(y_test, clf_aug.predict(X_test))
        acc_aug_list.append(acc_aug)
        
        print(f"Fold {fold+1} No Aug Accuracy: {acc_no_aug:.3f}")
        print(f"Fold {fold+1} Aug Accuracy: {acc_aug:.3f}")
    
    # visualize latent space just once since we have one VAE
    visualize_latent_space(model, X_cov_tensor, y_tensor, device)

    visualize_latent_space_tsne(model, X_cov_tensor, y_tensor, device, n_components=3)
    
    recon_stats = evaluate_reconstructions(model, X_cov_tensor, device)

    for i in range(3):  # look at first 3 examples
        original = X_test[i]
        original_tensor = torch.tensor(original, dtype=torch.float32)
        visualize_reconstruction(model, original_tensor, device)
    

    metrics = compute_reconstruction_metrics(model, test_loader, device)
    print(f"Mean Riemann distance: {metrics['mean_riem_dist']:.4f}")

    print("\n=== Final Results ===")
    print(f"No Augmentation: {np.mean(acc_no_aug_list):.3f} ± {np.std(acc_no_aug_list):.3f}")
    print(f"With Augmentation: {np.mean(acc_aug_list):.3f} ± {np.std(acc_aug_list):.3f}")
    
    # evaluate synthetic-only
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    acc_synthetic_list = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_cov)):
        print(f"\n=== Fold {fold+1}/{n_splits} ===")
        
        X_train = X_cov[train_idx]
        X_test = X_cov[test_idx]
        y_train = y_numeric[train_idx]
        y_test = y_numeric[test_idx]
        
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
        
        # Split training data by class (repeated for now)
        X_class_0 = X_train_tensor[y_train_tensor == 0]
        X_class_1 = X_train_tensor[y_train_tensor == 1]
        
        # generate synthetic training data only
        synthetic_data, synthetic_labels = wrapped_normal_sampling(
            model=model,
            n_samples_per_class=2000,
            class_data=[X_class_0, X_class_1],
            device=device,
            scale_factor=0.01
        )
        
        # train classifier on synthetic data only
        clf_synthetic = MDM(metric='riemann')
        clf_synthetic.fit(synthetic_data.cpu().numpy(), synthetic_labels.cpu().numpy())
        
        # evaluate on real test data
        acc_synthetic = balanced_accuracy_score(y_test, clf_synthetic.predict(X_test))
        acc_synthetic_list.append(acc_synthetic)
        
        print(f"Fold {fold+1} Synthetic-only Accuracy: {acc_synthetic:.3f}")
    
    print("\n=== Final Results ===")
    print(f"(Wrapped Normal Sampling) Synthetic-only: {np.mean(acc_synthetic_list):.3f} ± {np.std(acc_synthetic_list):.3f}")

    return acc_no_aug_list, acc_aug_list, acc_synthetic_list

def check_matrix_properties(original, reconstructed):
    """Compare eigenvalues and other properties of original vs reconstructed"""
    orig_np = original.cpu().numpy()
    recon_np = reconstructed.cpu().numpy()
    
    # eigenvalue comparison (mean absolute difference)
    orig_eig = np.linalg.eigvals(orig_np)
    recon_eig = np.linalg.eigvals(recon_np)
    
    # condition number comparison? (ratio of largest to smallest eigenvalue)
    orig_cond = np.linalg.cond(orig_np)
    recon_cond = np.linalg.cond(recon_np)
    
    # trace comparison (should be similar)
    orig_trace = np.trace(orig_np)
    recon_trace = np.trace(recon_np)
    
    return {
        'eig_diff': np.mean(np.abs(orig_eig - recon_eig)),
        'cond_ratio': recon_cond / orig_cond,
        'trace_diff': np.abs(orig_trace - recon_trace)
    }

def evaluate_reconstructions(model, test_data, device):
    """Analyze reconstruction quality across test data"""
    model.eval()
    property_stats = {
        'eig_diff': [],
        'cond_ratio': [],
        'trace_diff': []
    }
    
    with torch.no_grad():
        for batch_idx in range(len(test_data)):
            # get original matrix
            original = test_data[batch_idx:batch_idx+1]
            original = original.to(device)
            
            # get reconstruction
            recon, _, _ = model(original)
            
            # compute properties
            props = check_matrix_properties(original[0], recon[0])
            
            # collect stats
            for key in property_stats:
                property_stats[key].append(props[key])
    
    # compute summary statistics
    summary = {}
    for key in property_stats:
        values = property_stats[key]
        summary[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    # print report 
    print("\nReconstruction Quality Analysis:")
    print("-" * 50)
    print(f"Eigenvalue difference (closer to 0 is better):")
    print(f"  Mean: {summary['eig_diff']['mean']:.4f} ± {summary['eig_diff']['std']:.4f}")
    print(f"  Range: [{summary['eig_diff']['min']:.4f}, {summary['eig_diff']['max']:.4f}]")
    
    print(f"\nCondition number ratio (closer to 1 is better):")
    print(f"  Mean: {summary['cond_ratio']['mean']:.4f} ± {summary['cond_ratio']['std']:.4f}")
    print(f"  Range: [{summary['cond_ratio']['min']:.4f}, {summary['cond_ratio']['max']:.4f}]")
    
    print(f"\nTrace difference (closer to 0 is better):")
    print(f"  Mean: {summary['trace_diff']['mean']:.4f} ± {summary['trace_diff']['std']:.4f}")
    print(f"  Range: [{summary['trace_diff']['min']:.4f}, {summary['trace_diff']['max']:.4f}]")
    
    return summary

def compute_reconstruction_metrics(model, data_loader, device):
    model.eval()
    riem_distances = []
    
    with torch.no_grad(): 
        for batch, _ in data_loader: # ignore labels
            batch = batch.to(device)
            recon, _, _ = model(batch) # get reconstruction
            
            # compute distances
            for orig, rec in zip(batch, recon): # iterate over batch
                dist = safe_distance_riemann(
                    orig.cpu().numpy(), 
                    rec.cpu().numpy()
                )
                riem_distances.append(dist)
    
    return {
        'mean_riem_dist': np.mean(riem_distances),
        'std_riem_dist': np.std(riem_distances)
    }

def visualize_reconstruction(model, original_matrix, device):
    """A visual comparison between an original SPD matrix and its reconstruction from the VAE."""
    model.eval()

    
    with torch.no_grad():
        recon, _, _ = model(original_matrix.unsqueeze(0).to(device))
        recon = recon.squeeze(0).cpu()
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # plot original
    im1 = ax1.imshow(original_matrix.cpu(), cmap='coolwarm') # display original matrix as heatmap 
    ax1.set_title('Original')
    plt.colorbar(im1, ax=ax1)
    
    # plot reconstruction
    im2 = ax2.imshow(recon, cmap='coolwarm')
    ax2.set_title('Reconstructed')
    plt.colorbar(im2, ax=ax2)
    
    # plot difference
    diff = original_matrix.cpu() - recon
    im3 = ax3.imshow(diff, cmap='coolwarm')
    ax3.set_title('Difference')
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    plt.show()


def evaluate_synthetic_only(X_cov, y_numeric, device, n_splits=5, seed=11):
    """Train VAE once, then evaluate using only synthetic data for training"""
    # train VAE on all data first -> same set up as before
    X_cov_tensor = torch.tensor(X_cov, dtype=torch.float32)
    y_tensor = torch.tensor(y_numeric, dtype=torch.long)
    
    model = run_vae_pipeline(
        X_cov_tensor=X_cov_tensor,
        y_tensor=y_tensor,
        train_idx=np.arange(len(X_cov)),
        vae_type=VAEType.RIEMANNIAN,
        n_channels=3,
        latent_dim=16,
        epochs=800
    )

    # setup cross validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    acc_synthetic_list = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_cov)):
        print(f"\n=== Fold {fold+1}/{n_splits} ===")
        
        X_train = X_cov[train_idx]
        X_test = X_cov[test_idx]
        y_train = y_numeric[train_idx]
        y_test = y_numeric[test_idx]
        
        # generate synthetic training data only
        synthetic_data, synthetic_labels = generate_synthetic_data_per_class(
            model=model,
            X_orig=torch.tensor(X_train, dtype=torch.float32).to(device),
            y_orig=torch.tensor(y_train, dtype=torch.long).to(device),
            n_samples_per_class=1000,  # increase synthetic samples since it's our only training data
            latent_dim=16,
            device=device,
            seed=seed
        )
        
        # train classifier on synthetic data only
        clf_synthetic = MDM(metric='riemann')
        clf_synthetic.fit(synthetic_data.cpu().numpy(), synthetic_labels.cpu().numpy())
        
        # evaluate on real test data
        acc_synthetic = balanced_accuracy_score(y_test, clf_synthetic.predict(X_test))
        acc_synthetic_list.append(acc_synthetic)
        
        print(f"Fold {fold+1} Synthetic-only Accuracy: {acc_synthetic:.3f}")
    
    print("\n=== Final Results ===")
    print(f"Synthetic-only: {np.mean(acc_synthetic_list):.3f} ± {np.std(acc_synthetic_list):.3f}")
    
    return acc_synthetic_list


def main():

    vae_type = VAEType.RIEMANNIAN  # VAEType.REGULAR or VAEType.RIEMANNIAN

    # Data retrieval with proper paradigm constraints
    X, y, meta = paradigm.get_data(
        dataset=dataset,
        subjects=subjects, # only 1 subject for testing
        return_epochs=False  # get raw data for custom covariance processing
    )

    print(f"Initial X shape: {X.shape}") # (trials, channels, time)
    plot_EEG(X, trial=5)  # visualize first trial
    X_cov = inspect_pipeline(preprocess_pipe, X, trial=5, channels=[0, 1, 2], sampling_rate=250) # TODO: skip the visualization part
    verify_SPD_properties(X_cov)

    # convert string labels to integers -> label encoding
    label_map = {'right_hand': 0, 'left_hand': 1}
    y_numeric = np.array([label_map[label] for label in y])

    # data splitting with subject stratification
    train_idx, test_idx = train_test_split(
        range(len(X_cov)),
        test_size=0.2,
        stratify=meta['subject'], # preserve equal subject distribution
        random_state=11
    )

    print(f"train_idx length: {len(train_idx)}")
    print(f"y_numeric length: {len(y_numeric)}")

    print(f"X_cov[train_idx]: {X_cov[train_idx]}")
    print(f"y_numeric[train_idx]: {y_numeric[train_idx]}")

    # convert to tensors and create datasets/loaders
    X_cov_tensor = torch.tensor(X_cov, dtype=torch.float32)
    y_tensor = torch.tensor(y_numeric, dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #n_channels = 3  # BNCI2014_004 has 3 channels
    #latent_dim = 16

    
    # # for SPD matrices with shape (3x3), flatten to size 9
    #X_cov_flattened = X_cov_tensor.reshape(X_cov_tensor.shape[0], -1)
    
    #X_cov_normalized = (X_cov_flattened - X_cov_flattened.min()) / (X_cov_flattened.max() - X_cov_flattened.min())
    #train_dataset = TensorDataset(X_cov_normalized, y_tensor)

    #generator = torch.Generator()
    #generator.manual_seed(11)

    #train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=generator)

    #print(f"X_cov_flattened shape: {X_cov_flattened.shape}")

    #input_dim = X_cov_tensor.size(1) * X_cov_tensor.size(2)  # Flattened input size (e.g., 9 for 3x3 matrices)
    '''
    model = run_vae_pipeline(
        X_cov_tensor=X_cov_tensor,
        y_tensor=y_tensor,
        train_idx=train_idx,
        vae_type=vae_type,
        n_channels=3,
        latent_dim=16,
        epochs=300
    )
    '''
    """
    synthetic_cov = generate_synthetic_data(
        model=model,
        n_samples=1000,
        latent_dim=16,
        vae_type=vae_type,
        device=device
    )"""
    '''
    acc_no_aug_list, acc_aug_list = evaluate_with_cv(X_cov, y_numeric, device=device)

    
    # plot results
    plt.figure(figsize=(10, 6))
    plt.boxplot([acc_no_aug_list, acc_aug_list], labels=['No Augmentation', 'With Augmentation'])
    plt.title('Classification Performance Across Folds')
    plt.ylabel('Balanced Accuracy')
    plt.show()
    '''
    acc_no_aug_list, acc_aug_list, acc_synthetic_list = evaluate_with_fixed_vae(X_cov, y_numeric, device=device)

    # plot results
    plt.figure(figsize=(10, 6))
    plt.boxplot([acc_no_aug_list, acc_aug_list], labels=['No Augmentation', 'With Augmentation'])
    plt.title('Classification Performance Across Folds with the same VAE')
    plt.ylabel('Balanced Accuracy')
    plt.show()

    #acc_synthetic_list = evaluate_synthetic_only(X_cov, y_numeric, device=device)

    plt.figure(figsize=(10, 6))
    plt.boxplot([acc_synthetic_list], labels=['Synthetic-only'])
    plt.ylabel('Balanced Accuracy')
    plt.title('Classification Performance with Synthetic Training Data')
    plt.show()
    
if __name__ == "__main__":
    main()