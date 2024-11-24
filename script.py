import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Define the Generator and Discriminator architectures for the GAN

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
            nn.Linear(1024, 1)
        )
    
    def forward(self, z):
        return self.fc(z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x)

# Preprocess dataset (SMILES strings) into a numerical representation for the GAN

def smiles_to_vec(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
    return None

# Read SMILES data from a file
def load_smiles_data(file_path):
    with open(file_path, 'r') as f:
        smiles_data = [line.strip() for line in f.readlines()]
    return smiles_data

# Load your SMILES dataset (replace with your file path)
smiles_data = load_smiles_data("smiles_data.txt")  # Ensure this file contains SMILES strings

# Convert SMILES to numerical vectors
data = [smiles_to_vec(smiles) for smiles in smiles_data if smiles_to_vec(smiles) is not None]

# Convert list to a PyTorch Dataset
class MoleculeDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)

dataset = MoleculeDataset(data)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Initialize the generator and discriminator models
generator = Generator(latent_dim=100)
discriminator = Discriminator()

# Optimizers
lr = 0.0002
optimizer_g = optim.Adam(generator.parameters(), lr=lr)
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

# Loss function
criterion = nn.BCELoss()

# Training loop for the GAN
epochs = 10000
for epoch in range(epochs):
    for real_data in dataloader:
        batch_size = real_data.size(0)

        # Train discriminator
        discriminator.zero_grad()
        
        # Real data
        real_labels = torch.ones(batch_size, 1)
        output = discriminator(real_data)
        loss_real = criterion(output, real_labels)

        # Fake data
        z = torch.randn(batch_size, 100)
        fake_data = generator(z)
        fake_labels = torch.zeros(batch_size, 1)
        output = discriminator(fake_data)
        loss_fake = criterion(output, fake_labels)

        # Backprop and optimize discriminator
        loss_d = loss_real + loss_fake
        loss_d.backward()
        optimizer_d.step()

        # Train generator
        generator.zero_grad()
        z = torch.randn(batch_size, 100)
        fake_data = generator(z)
        output = discriminator(fake_data)
        loss_g = criterion(output, real_labels)  # We want the generator to fool the discriminator
        loss_g.backward()
        optimizer_g.step()

    # Print the progress
    if epoch % 100 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss D: {loss_d.item()}, Loss G: {loss_g.item()}')

# Generate new molecules using the trained generator
def generate_molecule(generator, num_samples=1):
    z = torch.randn(num_samples, 100)
    generated_data = generator(z)
    # Convert generated data back to SMILES string
    generated_smiles = []
    for data in generated_data:
        mol = Chem.MolFromSmiles(str(data.numpy()))
        if mol:
            generated_smiles.append(Chem.MolToSmiles(mol))
    return generated_smiles

generated_molecules = generate_molecule(generator, num_samples=5)
print("Generated Molecules:", generated_molecules)
