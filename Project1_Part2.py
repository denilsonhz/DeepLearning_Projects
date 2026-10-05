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
    
    
#Part 2

#Similar to previous collate function but instead Extracts the 10 attributes of interest
def concept_collate_fn(batch):
    images, target_attrs = zip(*batch)
    images = torch.stack(images, dim=0)
    
    # Extract all 10 concept values
    raw_concepts = torch.stack([
        torch.tensor([attr[idx] for idx in concept_indices]) 
        for attr in target_attrs
    ])
    concepts = (raw_concepts == 1).float()
    
    return images, concepts

# Instantiate DataLoaders with new collate function
train_loader_c = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=concept_collate_fn)
test_loader_c  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=concept_collate_fn)

# Similar class as previous but now the linear head has 10 attributes instead of 1
class ConceptPredictor(nn.Module):
    def __init__(self, backbone, num_concepts=10):
        super(ConceptPredictor, self).__init__()
        self.backbone = backbone
        self.concept_head = nn.Linear(backbone.in_features, num_concepts)

    def forward(self, x):
        features = self.backbone(x)
        return self.concept_head(features)  # Shape: (N, 10)

# Initialize the model with the shared backbone and 10 attributes of interest
shared_backbone_2 = SharedResNetBackbone(pretrained=True)
concept_model = ConceptPredictor(backbone=shared_backbone_2, num_concepts=NUM_CONCEPTS).to(device)

# Defining the criterion and omptimizer as discussed before 
criterion_c = nn.BCEWithLogitsLoss() # Able to handle multi-label targets across all 10 outputs
optimizer_c = optim.Adam(concept_model.parameters(), lr=LEARNING_RATE)

print("\n--- Training Concept Predictor (x -> c) ---")

#Same steps for the epochs as before, except calculating loss across all 10 attributes
for epoch in range(NUM_EPOCHS):
    concept_model.train()
    running_loss = 0.0
    
    for images, concept_targets in train_loader_c:
        images, concept_targets = images.to(device), concept_targets.to(device)
        
        optimizer_c.zero_grad()
        logits = concept_model(images)
        loss = criterion_c(logits, concept_targets)
        loss.backward()
        optimizer_c.step()
        
        running_loss += loss.item() * images.size(0)
        
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - Loss: {epoch_loss:.4f}")


# Setup metrics as before, but will use lists to evaluate each concept individually
acc_metric = BinaryAccuracy().to(device)
f1_metric = BinaryF1Score().to(device)

# Lists to collect predictions and targets per-attribute
all_preds, all_targets = [], []

# Same eval set up as before
concept_model.eval()
with torch.no_grad():
    for images, concept_targets in test_loader_c:
        images, concept_targets = images.to(device), concept_targets.to(device)
        logits = concept_model(images)
        probs = torch.sigmoid(logits)

        #Append the 10 attribute performances to the lists
        all_preds.append(probs)
        all_targets.append(concept_targets.int())

# Concatenate all the batches together so that we have performance across attributes for all batches in matrix form
all_preds = torch.cat(all_preds, dim=0)
all_targets = torch.cat(all_targets, dim=0)

# Now to iterate through each attribute of interest, look at the predict performances, and calculate
# the metrics so we get performance across every attribute for all of the data.
per_concept_accs, per_concept_f1s = [], []

print("\n--- Per-Concept Performance ---")
for i, concept_name in enumerate(CONCEPT_ATTRS):
    c_preds, c_targets = all_preds[:, i], all_targets[:, i]
    
    c_acc = acc_metric(c_preds, c_targets).item()
    c_f1 = f1_metric(c_preds, c_targets).item()
    
    per_concept_accs.append(c_acc)
    per_concept_f1s.append(c_f1)
    
    print(f"{concept_name:<22} | Acc: {c_acc * 100:.2f}% | F1: {c_f1:.4f}")

# Compute macro-average across all 10 attributes
mean_acc = sum(per_concept_accs) / NUM_CONCEPTS
mean_f1  = sum(per_concept_f1s) / NUM_CONCEPTS

print("\n=== Concept Predictor (x -> c) Report ===")
print(f"Mean Concept Accuracy (Macro): {mean_acc * 100:.2f}%")
print(f"Mean Concept F1 Score (Macro): {mean_f1:.4f}")
