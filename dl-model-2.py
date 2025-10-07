import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from sklearn.metrics import classification_report
from torch.nn.utils.parametrizations import weight_norm
import matplotlib.pyplot as plt
import random


class TextImageDataset(Dataset):
    def __init__(self, csv_path, image_dir, image_cnn_path, image_hog_path, text_w2v_path, text_tfidf_path, 
                 cnn_mean, hog_mean, w2v_mean, tfidf_mean, cnn_std, hog_std, w2v_std, tfidf_std):
        # data
        self.data = pd.read_csv(csv_path)
        self.image_paths = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]

        # features
        self.image_cnn = np.load(image_cnn_path).astype(np.float32)
        self.image_hog = np.load(image_hog_path).astype(np.float32)
        self.text_w2v = np.load(text_w2v_path).astype(np.float32) 
        self.text_tfidf = np.load(text_tfidf_path).astype(np.float32)
        

        # mean and std values for normalization
        self.cnn_mean = cnn_mean
        self.hog_mean = hog_mean
        self.w2v_mean = w2v_mean
        self.tfidf_mean = tfidf_mean
        self.cnn_std = cnn_std
        self.hog_std = hog_std
        self.w2v_std = w2v_std
        self.tfidf_std = tfidf_std
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = str(self.data.iloc[idx, -2])
        image_idx = [k for k, image_path in enumerate(self.image_paths) if image_name in image_path][0]

        image_cnn = self.image_cnn[image_idx]
        image_hog = self.image_hog[idx]
        text_w2v = self.text_w2v[idx] 
        text_tfidf = self.text_tfidf[idx]

        # normalization
        image_cnn = (image_cnn - self.cnn_mean) / self.cnn_std
        image_hog = (image_hog - self.hog_mean) / self.hog_std
        text_w2v = (text_w2v - self.w2v_mean) / self.w2v_std
        text_tfidf = (text_tfidf - self.tfidf_mean) / self.tfidf_std

        label = torch.tensor(self.data.iloc[idx, -1], dtype=torch.float32)
        
        image_vector = np.concatenate((image_cnn, image_hog), axis=0)
        repeated_w2v = np.tile(text_w2v, 20)
        text_vector = np.concatenate((repeated_w2v, text_tfidf), axis=0)
        
        return torch.tensor(image_vector, dtype=torch.float32), torch.tensor(text_vector, dtype=torch.float32), label  


class InfertenceTextImageDataset(Dataset):
    def __init__(self, csv_path, image_dir, image_cnn_path, image_hog_path, text_w2v_path, text_tfidf_path,
                cnn_mean, hog_mean, w2v_mean, tfidf_mean, cnn_std, hog_std, w2v_std, tfidf_std):
        # data
        self.data = pd.read_csv(csv_path)
        self.image_paths = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]

        #features
        self.image_cnn = np.load(image_cnn_path).astype(np.float32)
        self.image_hog = np.load(image_hog_path).astype(np.float32)
        self.text_w2v = np.load(text_w2v_path).astype(np.float32) 
        self.text_tfidf = np.load(text_tfidf_path).astype(np.float32)
        
        # mean and std values for normalization
        self.cnn_mean = cnn_mean
        self.hog_mean = hog_mean
        self.w2v_mean = w2v_mean
        self.tfidf_mean = tfidf_mean
        self.cnn_std = cnn_std
        self.hog_std = hog_std
        self.w2v_std = w2v_std
        self.tfidf_std = tfidf_std
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = str(self.data.iloc[idx, -1])
        image_idx = [k for k, image_path in enumerate(self.image_paths) if image_name in image_path][0]
        
        image_cnn = self.image_cnn[image_idx]
        image_hog = self.image_hog[idx]
        text_w2v = self.text_w2v[idx] 
        text_tfidf = self.text_tfidf[idx]

        # normalization
        image_cnn = (image_cnn - self.cnn_mean) / self.cnn_std
        image_hog = (image_hog - self.hog_mean) / self.hog_std
        text_w2v = (text_w2v - self.w2v_mean) / self.w2v_std
        text_tfidf = (text_tfidf - self.tfidf_mean) / self.tfidf_std

        image_vector = np.concatenate((image_cnn, image_hog), axis=0)
        repeated_w2v = np.tile(text_w2v, 20)
        text_vector = np.concatenate((repeated_w2v, text_tfidf), axis=0)

        return torch.tensor(image_vector, dtype=torch.float32), torch.tensor(text_vector, dtype=torch.float32)


class ImageTextModel(nn.Module):
    def __init__(self):
        super(ImageTextModel, self).__init__()

        # combined FC layers
        combined_input_dim = 2000 + 1944 + 2000 + 2000 

        self.fc_combined = nn.Sequential(
            nn.Dropout(0.2),
            weight_norm(nn.Linear(combined_input_dim, 512)),
            nn.GELU(),
            nn.Dropout(0.25),
            weight_norm(nn.Linear(512, 128)),
            nn.GELU(),
            nn.Dropout(0.25),
            weight_norm(nn.Linear(128, 32)),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, image, text_vector):

        # combine image and text features
        combined = torch.cat((image, text_vector), dim=1)

        output = self.fc_combined(combined)
        return output


# region Load Data
np_train_image_cnn = np.load('/kaggle/input/dl-v2/train_images_cnn.npy')
np_val_image_cnn = np.load('/kaggle/input/dl-v2/val_images_cnn.npy')
np_test_image_cnn = np.load('/kaggle/input/dl-v2/test_images_cnn.npy')

np_train_hog = np.load("/kaggle/input/dl-v2/train_hog.npy")
np_val_hog = np.load("/kaggle/input/dl-v2/val_hog.npy")
np_test_hog = np.load("/kaggle/input/dl-v2/test_hog.npy")

np_train_text_w2v = np.load('/kaggle/input/dl-v2/train_embeddings_w2v.npy')
np_val_text_w2v = np.load('/kaggle/input/dl-v2/val_embeddings_w2v.npy')
np_test_text_w2v = np.load('/kaggle/input/dl-v2/test_embeddings_w2v.npy')

np_train_text_tfidf = np.load('/kaggle/input/dl-v2/train_tfidf.npy')
np_val_text_tfidf = np.load('/kaggle/input/dl-v2/val_tfidf.npy')
np_test_text_tfidf = np.load('/kaggle/input/dl-v2/test_tfidf.npy')
# endregion

# combine train, validation, and test sets for each feature group
np_cnn_data = np.concatenate((np_train_image_cnn, np_val_image_cnn, np_test_image_cnn), axis=0)
np_hog_data = np.concatenate((np_train_hog, np_val_hog, np_test_hog), axis=0)
np_w2v_data = np.concatenate((np_train_text_w2v, np_val_text_w2v, np_test_text_w2v), axis=0)
np_tfidf_data = np.concatenate((np_train_text_tfidf, np_val_text_tfidf, np_test_text_tfidf), axis=0)

# compute mean and std for each group
cnn_mean, cnn_std = np_cnn_data.mean(), np_cnn_data.std()
hog_mean, hog_std = np_hog_data.mean(), np_hog_data.std()
w2v_mean, w2v_std = np_w2v_data.mean(), np_w2v_data.std()
tfidf_mean, tfidf_std = np_tfidf_data.mean(), np_tfidf_data.std()

print(f"Mean and STD for CNN Features: Mean = {cnn_mean}, STD = {cnn_std}")
print(f"Mean and STD for HOG Features: Mean = {hog_mean}, STD = {hog_std}")
print(f"Mean and STD for W2V Features: Mean = {w2v_mean}, STD = {w2v_std}")
print(f"Mean and STD for TF-IDF Features: Mean = {tfidf_mean}, STD = {tfidf_std}")

#region Paths
train_csv_path = '/kaggle/input/dl-project/isp-match-dl-2024_v2/train.csv'
val_csv_path = '/kaggle/input/dl-project/isp-match-dl-2024_v2/val.csv'
test_csv_path = '/kaggle/input/dl-project/isp-match-dl-2024_v2/test.csv'

train_image_dir = '/kaggle/input/dl-project/isp-match-dl-2024_v2/train_images'
val_image_dir = '/kaggle/input/dl-project/isp-match-dl-2024_v2/val_images'
test_image_dir = '/kaggle/input/dl-project/isp-match-dl-2024_v2/test_images'

train_image_cnn = '/kaggle/input/dl-v2/train_images_cnn.npy'
val_image_cnn = '/kaggle/input/dl-v2/val_images_cnn.npy'
test_image_cnn = '/kaggle/input/dl-v2/test_images_cnn.npy'

train_hog = "/kaggle/input/dl-v2/train_hog.npy"
val_hog = "/kaggle/input/dl-v2/val_hog.npy"
test_hog = "/kaggle/input/dl-v2/test_hog.npy"

train_text_w2v_path = '/kaggle/input/dl-v2/train_embeddings_w2v.npy'
val_text_w2v_path = '/kaggle/input/dl-v2/val_embeddings_w2v.npy'
test_text_w2v_path = '/kaggle/input/dl-v2/test_embeddings_w2v.npy'

train_text_tfidf_path = '/kaggle/input/dl-v2/train_tfidf.npy'
val_text_tfidf_path = '/kaggle/input/dl-v2/val_tfidf.npy'
test_text_tfidf_path = '/kaggle/input/dl-v2/test_tfidf.npy'
#endregion

train_dataset = TextImageDataset(train_csv_path, train_image_dir, train_image_cnn, train_hog, train_text_w2v_path, train_text_tfidf_path, 
                                 cnn_mean, hog_mean, w2v_mean, tfidf_mean, cnn_std, hog_std, w2v_std, tfidf_std)
val_dataset = TextImageDataset(val_csv_path, val_image_dir, val_image_cnn, val_hog, val_text_w2v_path, val_text_tfidf_path,
                              cnn_mean, hog_mean, w2v_mean, tfidf_mean, cnn_std, hog_std, w2v_std, tfidf_std)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.BCEWithLogitsLoss()


# hyperparameter search space 
lr_min, lr_max = 1e-6, 1e-2  # learning rate interval
batch_size_min, batch_size_max = 16, 512  # batch size interval

# random search configuration
num_search_iterations = 30 
num_epochs = 10

best_model_path = 'best_model.pth'
best_val_accuracy = 0.0
best_params = {}

# lists to store training metrics
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
train_f1_scores = []
val_f1_scores = []
learning_rates_list = []


for iteration in range(num_search_iterations):
    # randomly select hyperparameters
    lr = random.uniform(lr_min, lr_max)
    batch_size = random.randint(batch_size_min, batch_size_max)
    
    print(f"Random Search Iteration {iteration+1}/{num_search_iterations}")
    print(f"Training with lr={lr} and batch_size={batch_size}")
    
    # create data loaders for the selected batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    model = ImageTextModel().to(device)
    # create optimizer for the selected lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-1)
    
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

            running_loss += loss.item()

            # collect predictions and true labels for metrics
            y_true_train.extend(labels.cpu().numpy())
            y_pred_train.extend(torch.sigmoid(outputs).cpu().detach().numpy())

        avg_train_loss = running_loss / len(train_loader)
        
        # calculate training metrics for the classification report
        y_true_train = np.array(y_true_train)
        y_pred_train_binary = (np.array(y_pred_train) > 0.5).astype(int)
        train_report = classification_report(y_true_train, y_pred_train_binary, target_names=['Class 0', 'Class 1'], output_dict=True)
        train_accuracy = train_report['accuracy']
        train_f1 = train_report['macro avg']['f1-score']
        
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        train_f1_scores.append(train_f1)
        learning_rates_list.append(lr)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, F1-Score: {train_f1:.4f}')

        # validation loop
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

                # collect predictions and true labels for validation metrics
                y_true_val.extend(labels.cpu().numpy())
                y_pred_val.extend(torch.sigmoid(outputs).cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)

        # calculate validation metrics
        y_true_val = np.array(y_true_val)
        y_pred_val_binary = (np.array(y_pred_val) > 0.5).astype(int)
        val_report = classification_report(y_true_val, y_pred_val_binary, target_names=['Class 0', 'Class 1'], output_dict=True)
        val_accuracy = val_report['accuracy']
        val_f1 = val_report['macro avg']['f1-score']

        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        val_f1_scores.append(val_f1)

        print(f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, F1-Score: {val_f1:.4f}')

        # save the best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_params = {'lr': lr, 'batch_size': batch_size}
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Best model saved with accuracy: {best_val_accuracy:.4f}')

    print(f"Finished training for iteration {iteration+1}")
    print(f'Best model saved with accuracy: {best_val_accuracy:.4f}')


print("Best Hyperparameters:", best_params)
print("Best Validation Accuracy:", best_val_accuracy)


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
plt.plot(learning_rates_list, label='Learning Rate')
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Across Steps')
plt.legend()

plt.tight_layout()
plt.show()



# inference
model.load_state_dict(torch.load(best_model_path))
model.eval()


test_dataset = InfertenceTextImageDataset(test_csv_path, test_image_dir, test_image_cnn, test_hog, test_text_w2v_path, test_text_tfidf_path,
                                         cnn_mean, hog_mean, w2v_mean, tfidf_mean, cnn_std, hog_std, w2v_std, tfidf_std)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

predictions = []


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