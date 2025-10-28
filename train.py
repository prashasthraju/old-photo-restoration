import torch
import itertools
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataset import UnpairedDataset
from models.generator import Generator
from models.discriminator import Discriminator

# ======================
# CONFIG
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 5
batch_size = 1
lr = 0.0002
lambda_cycle = 10.0
lambda_identity = 5.0

# ======================
# DATASET
# ======================
dataset = UnpairedDataset("data/old_preprocessed", "data/clean_preprocessed")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ======================
# MODELS
# ======================
G_AB = Generator().to(device)  # old -> clean
G_BA = Generator().to(device)  # clean -> old
D_A = Discriminator().to(device)  # real vs fake old
D_B = Discriminator().to(device)  # real vs fake clean

# ======================
# LOSS FUNCTIONS
# ======================
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

# ======================
# OPTIMIZERS
# ======================
optimizer_G = optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()),
    lr=lr, betas=(0.5, 0.999)
)
optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))

# ======================
# TRAINING LOOP
# ======================
for epoch in range(epochs):
    loop = tqdm(dataloader, leave=True)
    for i, data in enumerate(loop):
        real_A = data["old"].to(device)
        real_B = data["clean"].to(device)

        # ------------------
        #  Train Generators
        # ------------------
        optimizer_G.zero_grad()

        # Identity losses
        id_A = G_BA(real_A)
        id_B = G_AB(real_B)
        loss_id_A = criterion_identity(id_A, real_A)
        loss_id_B = criterion_identity(id_B, real_B)

        # GAN losses
        fake_B = G_AB(real_A)
        pred_fake_B = D_B(fake_B)
        loss_GAN_AB = criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B, device=device))

        fake_A = G_BA(real_B)
        pred_fake_A = D_A(fake_A)
        loss_GAN_BA = criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A, device=device))

        # Cycle losses
        rec_A = G_BA(fake_B)
        rec_B = G_AB(fake_A)
        loss_cycle_A = criterion_cycle(rec_A, real_A)
        loss_cycle_B = criterion_cycle(rec_B, real_B)

        # Total generator loss
        loss_G = (
            loss_GAN_AB + loss_GAN_BA +
            lambda_cycle * (loss_cycle_A + loss_cycle_B) +
            lambda_identity * (loss_id_A + loss_id_B)
        )
        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------
        optimizer_D_A.zero_grad()
        pred_real = D_A(real_A)
        loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real, device=device))
        pred_fake = D_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake, device=device))
        loss_D_A_total = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A_total.backward()
        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------
        optimizer_D_B.zero_grad()
        pred_real = D_B(real_B)
        loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real, device=device))
        pred_fake = D_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake, device=device))
        loss_D_B_total = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B_total.backward()
        optimizer_D_B.step()

        loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
        loop.set_postfix({
            "loss_G": loss_G.item(),
            "loss_D_A": loss_D_A_total.item(),
            "loss_D_B": loss_D_B_total.item()
        })

    # Save model checkpoints
    torch.save(G_AB.state_dict(), f"checkpoints/G_AB_epoch{epoch+1}.pth")
    torch.save(G_BA.state_dict(), f"checkpoints/G_BA_epoch{epoch+1}.pth")
    torch.save(D_A.state_dict(), f"checkpoints/D_A_epoch{epoch+1}.pth")
    torch.save(D_B.state_dict(), f"checkpoints/D_B_epoch{epoch+1}.pth")

print("Training finished ✅")
