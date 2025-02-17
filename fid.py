import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from scipy.linalg import sqrtm
import numpy as np

class FIDScore:
    def __init__(self):
        # Load the pre-trained InceptionV3 model
        self.inception_model = models.inception_v3(pretrained=True, transform_input=False)
        # Remove the final classification layer to get feature embeddings
        self.inception_model.fc = nn.Identity()
        self.inception_model.eval()

        # Define a transform to preprocess the 28x28 images for InceptionV3
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),  # Resize to InceptionV3 input size
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_images(self, images):
        # Convert 28x28 grayscale images to 3-channel RGB and preprocess
        if images.shape[1] == 1:  # If grayscale, convert to RGB
            images = images.repeat(1, 3, 1, 1)
        processed_images = torch.stack([self.transform(img) for img in images])
        return processed_images

    def get_features(self, images):
        # Preprocess images and extract features using InceptionV3
        processed_images = self.preprocess_images(images)
        with torch.no_grad():
            features = self.inception_model(processed_images)
        return features.cpu().numpy()

    def calculate_fid(self, real_images, generated_images):
        # Get features for real and generated images
        real_features = self.get_features(real_images)
        generated_features = self.get_features(generated_images)

        # Calculate mean and covariance of the features
        mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        mu_gen, sigma_gen = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)

        # Compute the squared trace of the covariance matrices
        covmean, _ = sqrtm(sigma_real.dot(sigma_gen), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        # Calculate the FID score
        fid_score = np.linalg.norm(mu_real - mu_gen)**2 + np.trace(sigma_real + sigma_gen - 2 * covmean)
        return fid_score

# Example usage
if __name__ == "__main__":
    # Create two random batches of 28x28 images (e.g., MNIST-like data)
    batch_size = 64
    real_images = torch.rand(batch_size, 1, 28, 28)  # Real images (grayscale)
    generated_images = torch.rand(batch_size, 1, 28, 28)  # Generated images (grayscale)

    # Compute FID score
    fid_calculator = FIDScore()
    fid_score = fid_calculator.calculate_fid(real_images, generated_images)
    print(f"FID Score: {fid_score}")