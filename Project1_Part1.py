# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 13:04:45 2026

@author: ianmh
"""

#All necessary packages
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CelebA
from torchvision.models import resnet18, ResNet18_Weights
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score

# Setting seed for reproducability
torch.manual_seed(42)

# Device configuration - will choose CPU for my setup
device = torch.device("cpu")
print(f"Active execution device: {device}")

# Global Hyperparameters
BATCH_SIZE = 64        # Number of training samples passed through the neural network on each forward/backward pass
LEARNING_RATE = 1e-4    # Use smaller learning rate as ResNet is pretrained. Default range for Adam Opt
NUM_EPOCHS = 5          # Number of training passes per part
DATA_DIR = "./data"     

# Target label
TARGET_ATTR = "Smiling"

# 10 Facial Attributes of Interest
CONCEPT_ATTRS = [
    "Mouth_Slightly_Open",
    "High_Cheekbones",
    "Chubby",
    "Narrow_Eyes",
    "Bags_Under_Eyes",
    "Big_Lips",
    "Big_Nose",
    "Pointy_Nose",
    "Bushy_Eyebrows",
    "Arched_Eyebrows",
]

# variable for concept quantity
NUM_CONCEPTS = len(CONCEPT_ATTRS) 

# Crops the image to reduce pixels for compute time and makes it square
transform = transforms.Compose([
    transforms.CenterCrop(178), #Crop first so PyTorch doesn't squish rectangular image into a square with Resize
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Load our datasets
full_train_ds = CelebA(root=DATA_DIR, split="train", target_type="attr", download=False, transform=transform)
full_test_ds  = CelebA(root=DATA_DIR, split="test",  target_type="attr", download=False, transform=transform)

# Subsampling dataset since I am using CPU
train_dataset = Subset(full_train_ds, list(range(5000)))
test_dataset  = Subset(full_test_ds, list(range(1000)))

# Grabbing the indices for the target and the concepts
target_y_idx = full_train_ds.attr_names.index(TARGET_ATTR)
concept_indices = [full_train_ds.attr_names.index(attr) for attr in CONCEPT_ATTRS]

print(f"Train subset size: {len(train_dataset)} | Test subset size: {len(test_dataset)}")

# Load in a pre-trained ResNet-18 Neural network and remove the final layer.
# Instead, final layer will output a vector of features, which we flatten
# to whatever dimension we need for our problem. Ex. 1 for "Smiling", 10 
# for all features of interest. 
class SharedResNetBackbone(nn.Module):
    """
    Base feature extractor class wrapping an ImageNet-pretrained ResNet-18 backbone.
    Strips off the default 1000-class fc layer, producing a 512-dimensional vector.
    """
    def __init__(self, pretrained=True):
        super(SharedResNetBackbone, self).__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        resnet = resnet18(weights=weights)
        
        self.in_features = resnet.fc.in_features  # 512 features
        
        # Remove original fc layer by slicing up to AvgPool2d
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        features = self.backbone(x)
        return torch.flatten(features, 1)  # Output shape: (N, 512)
    
    
#Part 1

# Custom collate function to extract 'Smiling' target vector
# This is used because CelebA includes large multidimensional images
# so we define a custom collate_fn to avoid returning all 40 attributes per 
# image and just the specific Smiling target label.
# Returns a tuple of the images and their corresponding target attribute
def baseline_collate_fn(batch):
    images, target_attrs = zip(*batch)
    images = torch.stack(images, dim=0)
    
    # Extract single target 'Smiling' and convert (-1, 1) -> (0.0, 1.0)
    raw_targets = torch.stack([attr[target_y_idx] for attr in target_attrs])
    targets = (raw_targets == 1).float().unsqueeze(1)  # Shape: (N, 1)
    
    return images, targets

# DataLoaders which automate feeding the data into the model.
# Creates a stack of images (size of which is batch_size = 64), where each image is 64x64 pixels with 3 color channels
train_loader_y = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=baseline_collate_fn)
test_loader_y  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=baseline_collate_fn)

# Going to use our pre-defined backbone class so we can 
# re-use the already trained model. Adds on a linear layer 
# so we can use it for our Smiling classification problem
# and produce a single output per image. This is not a probability yet
# because this is done later with BCEWithLogitsLoss() which applies the 
# sigmoid function.
class BaselineClassifier(nn.Module):
    def __init__(self, backbone, num_classes=1):
        super(BaselineClassifier, self).__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.in_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# Instantiate Shared Backbone and Model
shared_backbone_1 = SharedResNetBackbone(pretrained=True)
baseline_model = BaselineClassifier(backbone=shared_backbone_1, num_classes=1).to(device)

criterion_y = nn.BCEWithLogitsLoss()  #Combines both the sigmoid function wtih the binary cross-entropy function
optimizer_y = optim.Adam(baseline_model.parameters(), lr=LEARNING_RATE)  #Instead of SGD, Adam computes individual learning rate per parameter

# Training Loop
print("\n--- Training Baseline Classifier (x -> y) ---")
for epoch in range(NUM_EPOCHS):
    #Initialize starting point for this epoch
    baseline_model.train()
    running_loss = 0.0

    #Iterate through the batches of images
    for images, targets in train_loader_y:
        images, targets = images.to(device), targets.to(device)

        #Clearing gradient from the previous batch
        optimizer_y.zero_grad()
        
        logits = baseline_model(images) # Forward pass
        loss = criterion_y(logits, targets) # Loss function
        loss.backward() # Backwards pass
        optimizer_y.step() # Update model weights

        # Tracking total loss for this epoch
        running_loss += loss.item() * images.size(0)

    #Finding average loss for this epoch across the dataset
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - Loss: {epoch_loss:.4f}")

# Model Evaluation (Accuracy & AUROC)

# Can use these two functions to track metrics internally rather than doing
# so by hand with a list
test_acc_metric = BinaryAccuracy().to(device) # % of total test images correctly classified as Smiling
test_auroc_metric = BinaryAUROC().to(device)  # Measures how confident the model is in a correct/incorrect answer

# Switch to eval mode so no training behaviors are triggered
baseline_model.eval()

# Iterate through our test images, pass them through the model, apply sigmoid
# function to get a probability of smiling, and assess accuracy across
# entire test dataset
with torch.no_grad(): #Disables gradient calculations since we are evaluating
    for images, targets in test_loader_y:
        images, targets = images.to(device), targets.to(device)
        logits = baseline_model(images)
        probs = torch.sigmoid(logits)
        
        test_acc_metric.update(probs, targets.int())
        test_auroc_metric.update(probs, targets.int())

#Compute and print our total accuracies
print("\n=== Baseline Classifier (x -> y) Report ===")
print(f"Test Accuracy: {test_acc_metric.compute().item() * 100:.2f}%")
print(f"Test AUROC:    {test_auroc_metric.compute().item():.4f}")