
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import os
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset


class TextImageDataset(Dataset):
    def __init__(self, csv_path, image_dir, train=True):
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.image_paths = [f for f in os.listdir(self.image_dir) if f.lower().endswith('.jpg')]

        # image augmentation for training
        if train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomResizedCrop(100, scale=(0.8, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            # apply only normalization for validation/testing
            self.transform = transforms.Compose([
                transforms.Resize((100, 100)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        
        image_path = os.path.join(self.image_dir, self.image_paths[idx])
        image = Image.open(image_path).convert('RGB')

        image = self.transform(image)

        return image




class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # [100, 100, 3] -> [50, 50, 32]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [50, 50, 32] -> [25, 25, 64]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [25, 25, 64] -> [13, 13, 128]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # [13, 13, 128] -> [7, 7, 256]
            nn.ReLU(),
        )

        # flatten to fully connected layer
        self.flatten = nn.Flatten()
        self.fc_encoder = nn.Sequential(
            nn.Linear(7 * 7 * 256, 2000),  # latent representation
            nn.ReLU()
        )

        # decoder
        self.fc_decoder = nn.Sequential(
            nn.Linear(2000, 7 * 7 * 256),
            nn.ReLU()
        )

        # decoder upsampling layers
        self.upconv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.upconv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.upconv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.final_conv = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x, add_noise=False):
        if add_noise:
            noise = torch.randn_like(x) * 0.05
            x = x + noise
        # encoding
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.fc_encoder(x)

        # decoding
        x = self.fc_decoder(x)
        x = x.view(x.size(0), 256, 7, 7)  # reshape to [batch_size, 256, 7, 7]

        # upsampling steps with interpolate + convolution
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # [7, 7, 256] -> [14, 14, 256]
        x = self.upconv1(x)  # [14, 14, 256] -> [14, 14, 128]
        x = F.relu(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # [14, 14, 128] -> [28, 28, 128]
        x = self.upconv2(x)  # [28, 28, 128] -> [28, 28, 64]
        x = F.relu(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # [28, 28, 64] -> [56, 56, 64]
        x = self.upconv3(x)  # [56, 56, 64] -> [56, 56, 32]
        x = F.relu(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # [56, 56, 32] -> [112, 112, 32]
        x = self.final_conv(x)  # [112, 112, 32] -> [112, 112, 3]

        x = F.interpolate(x, size=(100, 100), mode='bilinear', align_corners=False)
        return x

    def encode(self, x):
        # encoding
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.fc_encoder(x)
        return x





train_csv_path = '/kaggle/input/dl-project/isp-match-dl-2024_v2/train.csv'
val_csv_path = '/kaggle/input/dl-project/isp-match-dl-2024_v2/val.csv'
test_csv_path = '/kaggle/input/dl-project/isp-match-dl-2024_v2/test.csv'

train_image_dir = '/kaggle/input/dl-project/isp-match-dl-2024_v2/train_images'
val_image_dir = '/kaggle/input/dl-project/isp-match-dl-2024_v2/val_images'
test_image_dir = '/kaggle/input/dl-project/isp-match-dl-2024_v2/test_images'

train_dataset = TextImageDataset(train_csv_path, train_image_dir)
val_dataset = TextImageDataset(val_csv_path, val_image_dir, train=False)
test_dataset = TextImageDataset(test_csv_path, test_image_dir, train=False)


combined_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
combined_loader = DataLoader(combined_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.MSELoss()
model = Autoencoder().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

num_epochs = 20

train_losses = []

best_train_loss = float('inf')
best_model_path = 'best_model_autoencoder.pth'

# training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images in combined_loader:
        images = images.to(device)

        # forward pass
        outputs = model(images, add_noise=True)
        loss = criterion(outputs.squeeze(), images)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(combined_loader)
    train_losses.append(avg_train_loss)

    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}')

    # save the model if training loss improves
    if avg_train_loss < best_train_loss:
        best_train_loss = avg_train_loss
        torch.save(model.state_dict(), best_model_path)
        print(f'Best model saved with training loss: {best_train_loss:.4f}')


epochs = range(1, num_epochs + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.tight_layout()
plt.show()


# inference
def extract_and_save_images(dataset, save_path, batch_size=1):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    images = []
    
    for batch in dataloader:
        batch = batch.to(device)
        batch_images = model.encode(batch).detach().cpu().numpy()
        images.append(batch_images)
    
    images_array = np.concatenate(images, axis=0)
    np.save(save_path, images_array)
    print(f"Saved images to {save_path}")


extract_and_save_images(train_dataset, 'train_images_cnn.npy')
extract_and_save_images(val_dataset, 'val_images_cnn.npy')
extract_and_save_images(test_dataset, 'test_images_cnn.npy')