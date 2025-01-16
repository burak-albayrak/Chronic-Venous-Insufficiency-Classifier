import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# GPU utilization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Advanced Albumentations Data Augmentation
def get_train_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
        ], p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_test_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# PyTorch Dataset Wrapper for Albumentations
class AlbumentationsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = np.array(img)

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        return img, label

# Advanced Model Architecture
class EnsembleModel(nn.Module):
    def __init__(self, num_classes=3):
        super(EnsembleModel, self).__init__()

        # Backbone models
        self.efficientnet = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Feature extraction layers
        self.efficientnet_features = nn.Sequential(*list(self.efficientnet.features))
        self.resnet_features = nn.Sequential(*list(self.resnet.children())[:-2])

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Fusion and classification layers
        feature_dim = 1280 + 2048  # EfficientNet + ResNet feature dimensions
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Feature extraction
        eff_features = self.gap(self.efficientnet_features(x)).flatten(1)
        resnet_features = self.gap(self.resnet_features(x)).flatten(1)

        # Feature fusion
        combined_features = torch.cat([eff_features, resnet_features], dim=1)

        # Classification
        return self.classifier(combined_features)

# Cross-validation function
def cross_validation(dataset, n_splits=5):
    labels = [label for _, label in dataset]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), labels), 1):
        print(f"\n--- {fold}. Fold ---")

        # Datasets
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # Albumentations transforms
        train_transform = get_train_transforms()
        test_transform = get_test_transforms()

        train_dataset = AlbumentationsDataset(train_subset, transform=train_transform)
        val_dataset = AlbumentationsDataset(val_subset, transform=test_transform)

        # DataLoader
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

        # Model and optimizer
        model = EnsembleModel().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

        best_accuracy = 0
        best_predictions = None

        # Training
        for epoch in range(20):
            model.train()
            total_loss = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Evalution
            model.eval()
            correct = 0
            total = 0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                val_loss = 0
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            accuracy = correct / total
            print(f"Fold {fold}, Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}, "
                  f"Val Loss: {val_loss/len(val_loader):.4f}, Accuracy: {accuracy:.4f}")

            scheduler.step(val_loss)

            # Save the best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_predictions = (all_preds, all_labels)

        # Confusion matrix and report
        cm = confusion_matrix(best_predictions[1], best_predictions[0])
        report = classification_report(
            best_predictions[1],
            best_predictions[0],
            target_names=dataset.classes
        )
        print("\nClassification Report:")
        print(report)

        fold_results.append({
            'accuracy': best_accuracy,
            'confusion_matrix': cm,
            'classification_report': report
        })

    return fold_results

# Load dataset
data_dir = '/content/drive/MyDrive/SENG445_Project/Project/dataset'
dataset = datasets.ImageFolder(data_dir)

# Run Performance analysis
results = cross_validation(dataset)

# Overall performance summary
accuracies = [result['accuracy'] for result in results]
print("\n--- Performance Summary ---")
print(f"Average Accuracy: {np.mean(accuracies):.4f}")
print("Fold Accuracies:", [f"{acc:.4f}" for acc in accuracies])