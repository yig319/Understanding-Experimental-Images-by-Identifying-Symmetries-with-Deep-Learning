import torch
import torch.nn as nn
import torchvision.models as models

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder (ResNet-50)
        self.encoder = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])  # Remove classification head
        
        # Latent space
        self.latent_dim = latent_dim
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Use Sigmoid activation for image reconstruction
        )
    
    def forward(self, x):
        # Encoder
        features = self.encoder(x)
        
        # Decoder
        reconstructed = self.decoder(features)
        
        return features, reconstructed

# Create an instance of the model
latent_dim = 256  # You can adjust the latent space dimension
model = Autoencoder(latent_dim)

# Print the model architecture
print(model)

# Define the loss functions (classification loss and autoencoder loss)
classification_criterion = nn.CrossEntropyLoss()
autoencoder_criterion = nn.MSELoss()

# You can set up your data loading, optimizer, and training loop according to your specific dataset and requirements.
