from torch import nn
import torch
import torch
import torch.nn.functional as F


class CustomMSELoss(nn.Module):
    """
    Custom Mean Squared Error Loss implementation for PyTorch.
    This implementation allows for optional averaging across batch dimension
    and custom weighting of samples.
    
    Args:
        reduction (str): Specifies the reduction method to apply to the output:
            'none': no reduction will be applied
            'mean': the sum of the output will be divided by the number of elements
            'sum': the output will be summed
            Default: 'mean'
        weight (torch.Tensor, optional): a manual rescaling weight given to each
            sample. If given, it has to be a Tensor of size (N,) where N is the
            batch size. Default: None
    """
    
    def __init__(self, reduction='mean', weight=None):
        super(CustomMSELoss, self).__init__()
        self.reduction = reduction
        self.weight = weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the loss calculation.
        
        Args:
            predictions (torch.Tensor): Model predictions
            targets (torch.Tensor): Ground truth values
            
        Returns:
            torch.Tensor: Computed loss value
            
        Raises:
            ValueError: If predictions and targets have different shapes
            ValueError: If weight tensor shape doesn't match batch size
        """
        if predictions.shape != targets.shape:
            raise ValueError(f"Shape mismatch: predictions shape {predictions.shape} "
                           f"!= targets shape {targets.shape}")
        
        squared_diff = (predictions - targets) ** 2
        
        # Apply sample weights if provided
        if self.weight is not None:
            if self.weight.shape[0] != predictions.shape[0]:
                raise ValueError(f"Weight tensor size {self.weight.shape[0]} "
                               f"!= batch size {predictions.shape[0]}")
            squared_diff = squared_diff * self.weight.view(-1, 1)
        
        # Apply reduction method
        if self.reduction == 'none':
            return squared_diff
        elif self.reduction == 'sum':
            return torch.sum(squared_diff)
        elif self.reduction == 'mean':
            return torch.mean(squared_diff)
        else:
            raise ValueError(f"Invalid reduction method: {self.reduction}")

class VAE_Loss(torch.nn.Module):
    def __init__(self, beta = 1e-5):
        super(VAE_Loss, self).__init__()
        self.mse_loss = CustomMSELoss()
        self.beta = beta

    def forward(self, reconstructed_mu_var : tuple, original, beta = 1e-5):
        reconstructed, mu, log_var = reconstructed_mu_var
        batch_size = reconstructed.size(0)
        # Reconstruction loss (mean squared error) / BCE
        reconstruction_loss = self.mse_loss(reconstructed, original)  # Sum over the batch
        # print(batch_size)
        # KL Divergence loss
        # Formula: 0.5 * (mu^2 + exp(log_var) - log_var - 1)
        kl_divergence = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()),axis=0)
        kl_loss = kl_divergence 
        # Total VAE loss
        # print("KL", kl_loss)
        # print("KL scaled", kl_loss * 1e-3)
        # print("REC", reconstruction_loss)
        total_loss = reconstruction_loss + kl_loss  * beta
        # print(total_loss)
        return total_loss


class VAE(nn.Module):
    def __init__(self, latent_dim=64, dropout_prob=0.3):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1), # 1x28x28 -> 64x14x14
        nn.ReLU(),
        nn.Dropout(dropout_prob),
        nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1), # 16x14x14 -> 16x7x7
        nn.ReLU(),
        nn.Dropout(dropout_prob),
        nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1), # 16x7x7 -> 16x7x7
        nn.ReLU(),
        nn.Dropout(dropout_prob),
        nn.Flatten(),
        nn.Linear(784, 784), # Intermediate projection
        nn.ReLU(),
        nn.Dropout(dropout_prob)
        )
        self.fc_mu = nn.Linear(784, latent_dim)
        self.fc_logvar = nn.Linear(784, latent_dim)

        self.decoder = nn.Sequential(
        nn.Linear(latent_dim, 784), # Match intermediate projection
        nn.ReLU(),
        nn.Dropout(dropout_prob),
        nn.Unflatten(1, (16, 7, 7)), # Reshape to 1024x1x1
        nn.ConvTranspose2d(16, 16, kernel_size=3, stride=1, padding=1), # 16x7x7 -> 16x7x7
        nn.ReLU(),
        nn.Dropout(dropout_prob),
        nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1), # 16x7x7 -> 16x14x14
        nn.ReLU(),
        nn.Dropout(dropout_prob),
        nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1), # 16x14x14 -> 1x28x28
        )


    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def generate(self, num_samples, bs=64, device=None):
        """
        Generate samples from the VAE by sampling from the latent distribution.
        
        Args:
            num_samples (int): Number of samples to generate
            bs (int): Batch size for generation
            device (torch.device): Device to generate samples on
            temperature (float): Temperature parameter for sampling (higher = more diverse)
                
        Returns:
            torch.Tensor: Generated samples of shape (num_samples, C, H, W)
        """
        if device is None:
            device = next(self.parameters()).device
            
        # Initialize list to store generated samples
        generated_samples = []
        
        # Calculate number of batches needed
        num_batches = (num_samples + bs - 1) // bs
        
        with torch.no_grad():
            for i in range(num_batches):
                # Calculate batch size for last batch
                current_bs = min(bs, num_samples - i * bs)
                
                # Sample from standard normal distribution
                z = torch.randn(current_bs, self.latent_dim, device=device)
                
                # Generate samples using decoder
                samples = self.decoder(z)
                
                # Append to list
                generated_samples.append(samples.cpu())
        
        # Concatenate all batches
        generated_samples = torch.cat(generated_samples, dim=0)
        
        # Ensure we return exactly num_samples
        return generated_samples[:num_samples]
