from models.generator import Generator
from models.discriminator import Discriminator
import torch

# Initialize models
G = Generator()
D = Discriminator()

# Dummy input image tensor (batch=1, RGB=3, 256x256)
x = torch.randn(1, 3, 256, 256)

# Run forward pass
print("Generator output shape:", G(x).shape)  # expected [1, 3, 256, 256]
print("Discriminator output shape:", D(x).shape)  # expected [1, 1, ~30, ~30]
