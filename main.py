import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

# Define the Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z).view(-1, 1, 28, 28)

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)

# Initialize models
generator = Generator()
discriminator = Discriminator()

# Optimizers
lr = 0.0002
optimizer_g = optim.Adam(generator.parameters(), lr=lr)
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

# Loss function
criterion = nn.BCELoss()

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Training
epochs = 50
for epoch in range(epochs):
    for i, (imgs, _) in enumerate(dataloader):
        # Prepare real and fake labels
        real = torch.ones(imgs.size(0), 1)
        fake = torch.zeros(imgs.size(0), 1)

        # Train Discriminator
        optimizer_d.zero_grad()
        
        # Real images
        outputs_real = discriminator(imgs)
        loss_real = criterion(outputs_real, real)
        
        # Fake images
        z = torch.randn(imgs.size(0), 100)
        fake_images = generator(z)
        outputs_fake = discriminator(fake_images.detach())
        loss_fake = criterion(outputs_fake, fake)

        loss_d = (loss_real + loss_fake) / 2
        loss_d.backward()
        optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()
        
        z = torch.randn(imgs.size(0), 100)
        fake_images = generator(z)
        outputs = discriminator(fake_images)
        loss_g = criterion(outputs, real)
        
        loss_g.backward()
        optimizer_g.step()
        
    print(f'Epoch [{epoch+1}/{epochs}], Loss_D: {loss_d.item()}, Loss_G: {loss_g.item()}')

    # Save generated images for visualization
    if (epoch + 1) % 10 == 0:
        save_image(fake_images, f'fake_images_{epoch+1}.png')

print("Training finished.")
