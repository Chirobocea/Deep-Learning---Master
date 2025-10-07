
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import os
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from torchvision import transforms

class TextImageDataset(Dataset):
    def __init__(self, csv_path, image_dir, text_npy_path, train=True):
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.text_vectors = np.load(text_npy_path).astype(np.int64) 

        # define image augmentation for training
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
        return len(self.data)

    def __getitem__(self, idx):
        image_name = str(self.data.iloc[idx, 2]) + '.jpg' 
        text_vector = self.text_vectors[idx]  

        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')

        image = self.transform(image)

        label = torch.tensor(self.data.iloc[idx, -1], dtype=torch.float32)

        return image, torch.tensor(text_vector, dtype=torch.int64), label 


class InfertenceTextImageDataset(Dataset):
    def __init__(self, csv_path, image_dir, text_npy_path):
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.text_vectors = np.load(text_npy_path).astype(np.int64) 

        # apply only normalization for validation/testing
        self.transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = str(self.data.iloc[idx, 2]) + '.jpg'  
        text_vector = self.text_vectors[idx]

        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')

        image = self.transform(image)

        return image, torch.tensor(text_vector, dtype=torch.int64)



class ImageTextModel(nn.Module):
    def __init__(self):
        super(ImageTextModel, self).__init__()
        
        # image backbone 2D convolutional layers
        self.conv_layers = nn.Sequential(
            weight_norm(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)),
            nn.GELU(),
            weight_norm(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)),
            nn.GELU(),
            weight_norm(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)),
            nn.GELU()
        )
        
        # image fully connected layers
        self.fc_img = nn.Sequential(
            nn.Flatten(),
            weight_norm(nn.Linear(256 * 25 * 25, 512)),
            nn.GELU(),
            weight_norm(nn.Linear(512, 256)),
            nn.GELU(),
        )
        
        # text backbone embedding + 1D convolutional layers
        self.embedding = nn.Embedding(1500, 32)  # embedding layer for tokens
        
        self.text_conv = nn.Sequential(
            weight_norm(nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3)),
            nn.GELU(),
            weight_norm(nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3)),
            nn.GELU(),
            weight_norm(nn.Conv1d(128, 256, kernel_size=7, stride=1, padding=3)),
            nn.GELU()
        )
        
        self.fc_text = nn.Sequential(
            nn.Flatten(),
            weight_norm(nn.Linear(256 * 32, 512)),
            nn.GELU(),
            weight_norm(nn.Linear(512, 256))
        )
        
        # combined input layer
        self.fc_combined = nn.Sequential(
            weight_norm(nn.Linear(512, 256)),
            nn.GELU(),
            weight_norm(nn.Linear(256, 128)),
            nn.GELU(),
            weight_norm(nn.Linear(128, 32)),
            nn.GELU(),
            weight_norm(nn.Linear(32, 1))
        )

    def forward(self, image, text_vector):
        # image processing
        x_img = self.conv_layers(image)
        x_img = self.fc_img(x_img)
        
        # text processing
        x_text = self.embedding(text_vector) 
        x_text = x_text.permute(0, 2, 1) 
        x_text = self.text_conv(x_text)
        x_text = self.fc_text(x_text)
        
        # combine image and text features
        combined = torch.cat((x_img, x_text), dim=1)
        
        # final output layer
        output = self.fc_combined(combined)
        
        return output


# paths to the data
train_csv_path = '/kaggle/input/dl-project/isp-match-dl-2024_v2/train.csv'
val_csv_path = '/kaggle/input/dl-project/isp-match-dl-2024_v2/val.csv'
test_csv_path = '/kaggle/input/dl-project/isp-match-dl-2024_v2/test.csv'

train_image_dir = '/kaggle/input/dl-project/isp-match-dl-2024_v2/train_images'
val_image_dir = '/kaggle/input/dl-project/isp-match-dl-2024_v2/val_images'
test_image_dir = '/kaggle/input/dl-project/isp-match-dl-2024_v2/test_images'

train_text_npy_path = '/kaggle/input/dl-v2/train_tokens.npy'
val_text_npy_path = '/kaggle/input/dl-v2/val_tokens.npy'
test_text_npy_path = '/kaggle/input/dl-v2/test_tokens.npy'

# create datasets and data loaders
train_dataset = TextImageDataset(train_csv_path, train_image_dir, train_text_npy_path)
val_dataset = TextImageDataset(val_csv_path, val_image_dir, val_text_npy_path, train=False)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

# create the model and set up training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.BCEWithLogitsLoss()
model = ImageTextModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

# total number of epochs
num_epochs = 10
num_batches_per_epoch = len(train_loader)
total_steps = num_epochs * num_batches_per_epoch 

# number of steps for the warm-up phase (20% of total steps)
num_warmup_steps = int(0.2 * total_steps)
num_warmup_epochs = int(0.2 * num_epochs)

# create learning rate schedulers based on steps
warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=num_warmup_steps)
cosine_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - num_warmup_steps, eta_min=0)

# Lists for tracking metrics
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
train_f1_scores = []
val_f1_scores = []
learning_rates = [] 

# save the best model based on validation accuracy
best_val_accuracy = 0.0
best_model_path = 'best_model.pth'

# training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    y_true_train = []
    y_pred_train = []

    for images, text_vectors, labels in train_loader:
        images, text_vectors, labels = images.to(device), text_vectors.to(device), labels.to(device)
    
        # forward pass
        outputs = model(images, text_vectors)
        loss = criterion(outputs.squeeze(), labels)
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # track learning rate
        learning_rates.append(optimizer.param_groups[0]['lr'])

        running_loss += loss.item()

        # collect predictions and true labels for metrics
        y_true_train.extend(labels.cpu().numpy())
        y_pred_train.extend(torch.sigmoid(outputs).cpu().detach().numpy())

        # apply warmup scheduler during the first num_warmup_steps
        if epoch < num_warmup_epochs:
            warmup_lr_scheduler.step()
        else:
            cosine_lr_scheduler.step()


    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # calculate training metrics from the classification report
    y_true_train = np.array(y_true_train)
    y_pred_train_binary = (np.array(y_pred_train) > 0.5).astype(int)
    train_report = classification_report(y_true_train, y_pred_train_binary, target_names=['Class 0', 'Class 1'], output_dict=True)

    train_accuracies.append(train_report['accuracy'])
    train_f1_scores.append(train_report['macro avg']['f1-score'])

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_train_loss:.4f}, Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
    print("Training Classification Report:")
    print(pd.DataFrame(train_report).transpose().round(3))

    # validation
    model.eval()
    val_loss = 0.0
    y_true_val = []
    y_pred_val = []
    with torch.no_grad():
        for images, text_vectors, labels in val_loader:
            images, text_vectors, labels = images.to(device), text_vectors.to(device), labels.to(device)
            outputs = model(images, text_vectors)
            loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item()

            y_true_val.extend(labels.cpu().numpy())
            y_pred_val.extend(torch.sigmoid(outputs).cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # calculate validation metrics
    y_true_val = np.array(y_true_val)
    y_pred_val_binary = (np.array(y_pred_val) > 0.5).astype(int)
    val_report = classification_report(y_true_val, y_pred_val_binary, target_names=['Class 0', 'Class 1'], output_dict=True)

    val_accuracies.append(val_report['accuracy'])
    val_f1_scores.append(val_report['macro avg']['f1-score'])

    print("Validation Classification Report:")
    print(pd.DataFrame(val_report).transpose().round(3))
    print(f'Validation Loss: {avg_val_loss:.4f}')

    # save the model if validation accuracy improves
    if val_report['accuracy'] > best_val_accuracy:
        best_val_accuracy = val_report['accuracy']
        torch.save(model.state_dict(), best_model_path)
        print(f'Best model saved with accuracy: {best_val_accuracy:.4f}')

epochs = range(1, num_epochs + 1)

plt.figure(figsize=(18, 12))

plt.subplot(3, 1, 1)
plt.plot(epochs, train_losses, label='Training Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(epochs, train_accuracies, label='Training Accuracy')
plt.plot(epochs, val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(epochs, train_f1_scores, label='Training F1-score')
plt.plot(epochs, val_f1_scores, label='Validation F1-score')
plt.xlabel('Epoch')
plt.ylabel('F1-score')
plt.title('Training and Validation F1-score')
plt.legend()

plt.figure(figsize=(10, 5))
plt.plot(learning_rates, label='Learning Rate')
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Across Steps')
plt.legend()

plt.tight_layout()
plt.show()



model.load_state_dict(torch.load(best_model_path))
model.eval()

# inference on the test set
test_dataset = InfertenceTextImageDataset(test_csv_path, test_image_dir, test_text_npy_path)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

predictions = []

# extract all IDs from the dataset
ids = test_dataset.data['id'].values

with torch.no_grad():
    for images, text_vectors in test_loader:
        images, text_vectors = images.to(device), text_vectors.to(device)
        outputs = model(images, text_vectors)
        probs = torch.sigmoid(outputs).cpu().numpy()
        predicted_labels = (probs > 0.5).astype(int).squeeze()
        predictions.extend(predicted_labels)

output_df = pd.DataFrame({'id': ids, 'label': predictions})
output_csv_path = 'test_predictions.csv'
output_df.to_csv(output_csv_path, index=False)
print(f'Test predictions saved to {output_csv_path}')


