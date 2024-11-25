import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

# Generator Model (Latent Space to 1024D Fingerprint)
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024)  # Output size matches Morgan fingerprint bit vector
        )
    
    def forward(self, z):
        return self.fc(z)

# Discriminator Model (1024D Fingerprint to 1 output)
class Discriminator(nn.Module):
    def __init__(self, input_dim=1024):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),  # Match input dimension to 1024
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x)

# Custom Dataset for Molecular Fingerprints (without labels)
class MoleculeDataset(Dataset):
    def __init__(self, molecules):
        self.molecules = molecules

    def __len__(self):
        return len(self.molecules)

    def __getitem__(self, idx):
        mol = Chem.MolFromSmiles(self.molecules[idx])
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fp = np.array(fp)
        return torch.tensor(fp, dtype=torch.float32)

# Training the GAN
def train_gan(generator, discriminator, dataloader, num_epochs=10):
    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        for real_data in dataloader:
            batch_size = real_data.size(0)
            real_data = real_data.view(batch_size, -1)  # Flatten the input data

            # Train Discriminator with real data
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            d_optimizer.zero_grad()

            # Real data
            real_output = discriminator(real_data)
            d_loss_real = criterion(real_output, real_labels)

            # Generate fake data
            z = torch.randn(batch_size, 100)  # Latent vector
            fake_data = generator(z)

            # Fake data
            fake_output = discriminator(fake_data)
            d_loss_fake = criterion(fake_output, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()

            # Generator tries to fool the discriminator
            fake_data = generator(z)
            fake_output = discriminator(fake_data)
            g_loss = criterion(fake_output, real_labels)  # Generator wants discriminator to believe it's real

            g_loss.backward()
            g_optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

# Read SMILES data from file (smiles_data.txt)
def read_smiles_from_file(file_path):
    smiles = []
    with open(file_path, 'r') as f:
        for line in f:
            smiles.append(line.strip())  # Assuming the file contains one SMILES per line
    return smiles

# Load data from smiles_data.txt
smiles_file_path = 'smiles_data.txt'  # Update with your file path
smiles = read_smiles_from_file(smiles_file_path)

# Create dataset and dataloader
dataset = MoleculeDataset(smiles)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Initialize Generator and Discriminator
generator = Generator(latent_dim=100)
discriminator = Discriminator(input_dim=1024)

# Train the GAN
train_gan(generator, discriminator, dataloader, num_epochs=10)
