import numpy as np
import torch

class LinearScheduler:
    def __init__(self, beta_start, beta_end, timesteps):
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        
    def forward(self, x_prev, t):
        """
        x_prev: previous timestep image
        t: current timestep index
        """
        beta_t = self.betas[t]
        noise = torch.randn_like(x_prev)
        x_t = torch.sqrt(1 - beta_t) * x_prev + noise * torch.sqrt(beta_t)
        return x_t, noise

class CosineScheduler:
    def __init__(self, beta_start, beta_end, timesteps):
        # Pre-compute schedule
        self.betas = torch.zeros(timesteps)
        for t in range(timesteps):
            beta_t = beta_end + 0.5 * (beta_start - beta_end) * (1 + np.cos(np.pi * t / timesteps))
            self.betas[t] = beta_t

    def forward(self, x_prev, t):
        """
        x_prev: previous timestep image
        t: current timestep index
        """
        beta_t = self.betas[t]
        noise = torch.randn_like(x_prev)
        x_t = torch.sqrt(1 - beta_t) * x_prev + torch.sqrt(beta_t) * noise
        return x_t, noise
    




class DDPM:
    """Basically some noise scheduler :)"""

    def __init__(self, beta_start, beta_end, timesteps):
        self.betas = torch.zeros(timesteps)
        for t in range(timesteps):
            beta_t = beta_end + 0.5 * (beta_start - beta_end) * (1 + np.cos(np.pi * t / timesteps))
            self.betas[t] = beta_t
        
        # Pre-compute alpha_bars
        self.alphas = 1 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.max_timestep = timesteps

    def forward(self, x_prev, t):
        """
        x_prev: previous timestep image
        t: current timestep index
        """
        beta_t = self.betas[t]
        noise = torch.randn_like(x_prev)
        x_t = torch.sqrt(1 - beta_t) * x_prev + torch.sqrt(beta_t) * noise
        return x_t, noise

    
    def add_noise(self, original, timesteps : list[int], noise):
        """
        Adds noise to the original batch of images, each image has separate timestamp and noise.
        original: batch of images (x_0) [B, C, H, W]  
        timesteps: list of timesteps [B]
        noise: batch of noise [B, C, H, W]

        returns : batch of changed images [B, C, H, W]
        """
        if noise is None:
            noise = torch.randn_like(original)

        device = original.device
        if self.alphas_bar.device != device:
            self.alphas_bar = self.alphas_bar.to(device)

        alpha_bars = self.alphas_bar[timesteps].view(-1, 1, 1, 1)  # [B, 1, 1, 1]

        x_t = torch.sqrt(alpha_bars) * original + torch.sqrt(1 - alpha_bars) * noise
        return x_t

class DDIMSampler:
    def __init__(self,beta_start, beta_end, timesteps=1000):
        # self.num_timesteps = num_timesteps
        # Create beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        # Calculate alphas
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.max_timestep = timesteps
        self.eps = 1e-8

        
    def sample(self, model, n_samples, img_size, device, steps=100, condition = None):
        # Start from pure noise
        x = torch.randn(n_samples, *img_size).to(device)
        if condition is None:
            condition = torch.zeros(n_samples).to(device).to(torch.long) -1 # -1 is for null condition
        else: 
            if condition.device != device:
                condition = condition.to(device)
            assert condition.shape == (n_samples,) and condition.dtype == torch.long, "Condition must be a tensor of shape (n_samples,) and dtype torch.long"
        # Create sampling timestep sequence
        timesteps = np.linspace(self.max_timestep - 1, 0, steps).astype(np.int64)
        
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_prev = timesteps[i + 1]
            
            # Get current alphas
            at = self.alphas_cumprod[t] + self.eps
            at_prev = self.alphas_cumprod[t_prev] + self.eps
            
            # Model prediction of noise
            with torch.no_grad():
                noise_pred = model(x, torch.tensor([t]).to(device), condition)
            
            # Predict x0 (clean image)
            pred_x0 = (x - torch.sqrt(1 - at) * noise_pred) / torch.sqrt(at)
            
            # DDIM deterministic formula
            x = torch.sqrt(at_prev) * pred_x0 + \
                torch.sqrt(1 - at_prev) * noise_pred
            
        return x
