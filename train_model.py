import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns




# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set hyperparameters
num_epochs = 50
batch_size = 64
learning_rate = 0.001

# Data augmentation and normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def create_val_split(data_dir, val_split=0.2):     
    for plant_type in os.listdir(data_dir):
        plant_type_path = os.path.join(data_dir, plant_type)
        if os.path.isdir(plant_type_path):
            train_plant_type_path = os.path.join(data_dir, 'train', plant_type)
            val_plant_type_path = os.path.join(data_dir, 'val', plant_type)

            os.makedirs(train_plant_type_path, exist_ok=True)
            os.makedirs(val_plant_type_path, exist_ok=True)

            images = [img for img in os.listdir(plant_type_path) if img.endswith(('.jpg', '.png', '.jpeg'))]
            train_images, val_images = train_test_split(images, test_size=val_split)

            for img in train_images:
                shutil.move(os.path.join(plant_type_path, img), os.path.join(train_plant_type_path, img))
            for img in val_images:
                shutil.move(os.path.join(plant_type_path, img), os.path.join(val_plant_type_path, img))

            shutil.rmtree(plant_type_path)

# Check if data already split
data_dir = './train'

train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

# Check if 'train' and 'val' folders already exist
if os.path.exists(train_dir) and os.path.exists(val_dir):
    print("Pre-existing data split")
else:
    create_val_split(data_dir)
    

# Load dataset
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# Define the CNN
class PlantClassifier(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.25):
        super(PlantClassifier, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(14 * 14 * 128, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


# Initialize and train the network
model = PlantClassifier(len(class_names)).to(device)
criterion = nn.CrossEntropyLoss()  # Loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Optimizer
#scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training loop
train_accuracies = []
max_accuracy = 0
for epoch in range(num_epochs):
    for phase in ['train', 'val']:
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
          
    epoch_loss = running_loss / dataset_sizes[phase]
    epoch_acc = running_corrects.double() / dataset_sizes[phase]
    if epoch_acc > max_accuracy:
        max_accuracy = epoch_acc
        torch.save(model.state_dict(), 'plant_classifier.pth')
        print("Saved new highscore model, acc=", max_accuracy)
        
    #scheduler.step()

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
    train_accuracies.append(epoch_acc)
    

# Plot the accuracies against the epoch number
plt.plot(train_accuracies, label='Training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluation function
def evaluate_model(model, dataloader, phase):
    model.eval()
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader[phase]:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())

    return np.array(true_labels), np.array(predicted_labels)

# Load the best saved model
model.load_state_dict(torch.load('plant_classifier.pth'))

# Evaluate the model on the validation dataset
true_labels, predicted_labels = evaluate_model(model, dataloaders, 'val')

# Calculate precision, recall, and F1 score
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 score: {f1:.4f}')

# Plot the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(12, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

