# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 13:02:44 2026

@author: ianmh
"""

'''
Packages
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CelebA
from torchvision.models import resnet18, ResNet18_Weights
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score
import gc
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, mean_squared_error
from scipy.stats import pearsonr
import random
import matplotlib.pyplot as plt


'''
Helper Functions
'''

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def evaluate_predictions(y_true, y_pred_probs, threshold=0.5):
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred_binary = (y_pred_probs >= threshold).astype(int)

    metrics = {}
    
    metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred_probs))
    
    try:
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            metrics['auroc'] = roc_auc_score(y_true, y_pred_probs, average='macro')
        else:
            metrics['auroc'] = roc_auc_score(y_true, y_pred_probs)
    except ValueError:
        metrics['auroc'] = float('nan')
        
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        metrics['f1'] = f1_score(y_true, y_pred_binary, average='macro', zero_division=0)
    else:
        metrics['f1'] = f1_score(y_true, y_pred_binary, zero_division=0)
        
    corrs = []
    if y_true.ndim > 1:
        for i in range(y_true.shape[1]):
            if np.std(y_true[:, i]) > 0 and np.std(y_pred_probs[:, i]) > 0:
                r, _ = pearsonr(y_true[:, i], y_pred_probs[:, i])
                corrs.append(r)
        metrics['avg_pearson'] = np.mean(corrs) if corrs else 0.0
    else:
        if np.std(y_true) > 0 and np.std(y_pred_probs) > 0:
            metrics['avg_pearson'], _ = pearsonr(y_true, y_pred_probs)
        else:
            metrics['avg_pearson'] = 0.0
            
    metrics['sparsity_pct'] = float(np.mean(y_pred_binary) * 100)
    
    return metrics

'''
Project Set up
'''

seed_everything(42)
device = torch.device("cpu")
print(f"Active execution device: {device}")

# Global Hyperparameters
BATCH_SIZE = 64        
LEARNING_RATE = 1e-3
REG_PARAM = 2e-5
FINAL_LAYER_PEN = 1e-4
NUM_EPOCHS = 5          # Number of training passes per part
DATA_DIR = "./data"
ELASTIC_NET = True
if ELASTIC_NET: EL_WEIGHT, L1_LAMBDA = (0.99, 1e-4)

DATA_DIR = "./data"     
TARGET_ATTR = "Smiling"
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
NUM_CONCEPTS = len(CONCEPT_ATTRS) 

transform = transforms.Compose([
    transforms.CenterCrop(178), 
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
full_train_ds = CelebA(root=DATA_DIR, split="train", target_type="attr", download=False, transform=transform)
full_test_ds  = CelebA(root=DATA_DIR, split="test",  target_type="attr", download=False, transform=transform)
train_dataset = Subset(full_train_ds, list(range(5000)))
test_dataset  = Subset(full_test_ds, list(range(1000)))
target_y_idx = full_train_ds.attr_names.index(TARGET_ATTR)
concept_indices = [full_train_ds.attr_names.index(attr) for attr in CONCEPT_ATTRS]

print(f"Train subset size: {len(train_dataset)} | Test subset size: {len(test_dataset)}")


class SharedResNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(SharedResNetBackbone, self).__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        resnet = resnet18(weights=weights)
        
        self.in_features = resnet.fc.in_features
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        features = self.backbone(x)
        return torch.flatten(features, 1)
    

'''
Part 1 Code
'''

def baseline_collate_fn(batch):
    images, target_attrs = zip(*batch)
    images = torch.stack(images, dim=0)
    
    raw_targets = torch.stack([attr[target_y_idx] for attr in target_attrs])
    targets = (raw_targets == 1).float().unsqueeze(1)
    
    return images, targets

train_loader_y = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=baseline_collate_fn)
test_loader_y  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=baseline_collate_fn)


class BaselineClassifier(nn.Module):
    def __init__(self, backbone, num_classes=1):
        super(BaselineClassifier, self).__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.in_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

shared_backbone_1 = SharedResNetBackbone(pretrained=True)
baseline_model = BaselineClassifier(backbone=shared_backbone_1, num_classes=1).to(device)

criterion_y = nn.BCEWithLogitsLoss()
optimizer_y = optim.Adam(baseline_model.parameters(), lr=LEARNING_RATE, weight_decay = REG_PARAM)  

print("\n=== Part 1: Baseline Classifier (x -> y) ===")
print("--- Training Model ---")
for epoch in range(NUM_EPOCHS):
    baseline_model.train()
    running_loss = 0.0

    for images, targets in train_loader_y:
        images, targets = images.to(device), targets.to(device)

        optimizer_y.zero_grad()
        
        logits = baseline_model(images) 
        loss = criterion_y(logits, targets)
        
        if ELASTIC_NET:
            target_params = list(baseline_model.classifier.parameters())
            l1_penalty = sum(p.abs().sum() for p in target_params)
            l2_penalty = sum((p**2).sum() for p in target_params)
            elastic_net_reg = L1_LAMBDA * (
                EL_WEIGHT * l1_penalty + ((1 - EL_WEIGHT) / 2) * l2_penalty
            )
            
            loss = loss + elastic_net_reg
        
        loss.backward()
        optimizer_y.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - Loss: {epoch_loss:.4f}")


baseline_model.eval()
all_preds, all_labels = [], [] 

with torch.no_grad():
    for images, targets in test_loader_y:
        images, targets = images.to(device), targets.to(device)
        logits = baseline_model(images)
        probs = torch.sigmoid(logits)

        all_preds.extend(probs.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())

print("\n--- Metric Report ---")
baseline_metrics = evaluate_predictions(all_labels, all_preds)
print(f"[Evaluation] Acc: {baseline_metrics['accuracy']:.4f} | AUROC: {baseline_metrics['auroc']:.4f} | F1: {baseline_metrics['f1']:.4f} | RMSE: {baseline_metrics['rmse']:.4f}")
print("=*60")
'''
Part 2 Code
'''

def concept_collate_fn(batch):
    images, target_attrs = zip(*batch)
    images = torch.stack(images, dim=0)
    
    raw_concepts = torch.stack([
        torch.tensor([attr[idx] for idx in concept_indices]) 
        for attr in target_attrs
    ])
    concepts = (raw_concepts == 1).float()
    
    return images, concepts

train_loader_c = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=concept_collate_fn)
test_loader_c  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=concept_collate_fn)

class ConceptPredictor(nn.Module):
    def __init__(self, backbone, num_concepts=10):
        super(ConceptPredictor, self).__init__()
        self.backbone = backbone
        self.concept_head = nn.Linear(backbone.in_features, num_concepts)

    def forward(self, x):
        features = self.backbone(x)
        return self.concept_head(features)

shared_backbone_2 = SharedResNetBackbone(pretrained=True)
concept_model = ConceptPredictor(backbone=shared_backbone_2, num_concepts=NUM_CONCEPTS).to(device)

criterion_c = nn.BCEWithLogitsLoss()
optimizer_c = optim.Adam(concept_model.parameters(), lr=LEARNING_RATE, weight_decay=REG_PARAM)

print("\n=== Part 2: Concept Predictor (x -> c) ==")
print("--- Training Model ---")
for epoch in range(NUM_EPOCHS):
    concept_model.train()
    running_loss = 0.0
    
    for images, concept_targets in train_loader_c:
        images, concept_targets = images.to(device), concept_targets.to(device)
        
        optimizer_c.zero_grad()
        logits = concept_model(images)
        loss = criterion_c(logits, concept_targets)
        
        if ELASTIC_NET:
            target_params = list(concept_model.concept_head.parameters())
            l1_penalty = sum(p.abs().sum() for p in target_params)
            l2_penalty = sum((p**2).sum() for p in target_params)
            elastic_net_reg = L1_LAMBDA * (
                EL_WEIGHT * l1_penalty + ((1 - EL_WEIGHT) / 2) * l2_penalty
            )
            loss = loss + elastic_net_reg   
        
        
        loss.backward()
        optimizer_c.step()
        
        running_loss += loss.item() * images.size(0)
        
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - Loss: {epoch_loss:.4f}")


acc_metric = BinaryAccuracy().to(device)
f1_metric = BinaryF1Score().to(device)

all_preds, all_targets = [], []

concept_model.eval()
with torch.no_grad():
    for images, concept_targets in test_loader_c:
        images, concept_targets = images.to(device), concept_targets.to(device)
        logits = concept_model(images)
        probs = torch.sigmoid(logits)

        all_preds.append(probs)
        all_targets.append(concept_targets.int())

all_preds = torch.cat(all_preds, dim=0)
all_targets = torch.cat(all_targets, dim=0)

per_concept_accs, per_concept_f1s = [], []

print("\n--- Per-Concept Performance ---")
for i, concept_name in enumerate(CONCEPT_ATTRS):
    c_preds, c_targets = all_preds[:, i], all_targets[:, i]
    
    c_acc = acc_metric(c_preds, c_targets).item()
    c_f1 = f1_metric(c_preds, c_targets).item()
    
    per_concept_accs.append(c_acc)
    per_concept_f1s.append(c_f1)
    
    print(f"{concept_name:<22} | Acc: {c_acc * 100:.2f}% | F1: {c_f1:.4f}")

mean_acc = sum(per_concept_accs) / NUM_CONCEPTS
mean_f1  = sum(per_concept_f1s) / NUM_CONCEPTS

print("\n--- Metric Report ---")
print(f"Mean Concept Accuracy (Macro): {mean_acc * 100:.2f}%")
print(f"Mean Concept F1 Score (Macro): {mean_f1:.4f}")
print("="*60)

'''
Part 3 Code
'''


class IndependentCBM(nn.Module):
    def __init__(self, concept_predictor, num_concepts=10):
        super(IndependentCBM, self).__init__()
        self.concept_predictor = concept_predictor
        
        for param in self.concept_predictor.parameters():
            param.requires_grad = False
            
        self.c_to_y = nn.Linear(num_concepts, 1)

    def forward(self, x):
        concept_logits = self.concept_predictor(x)      
        concept_probs = torch.sigmoid(concept_logits)
        
        target_logit = self.c_to_y(concept_probs)
        return target_logit

cbm_model = IndependentCBM(concept_predictor=concept_model, num_concepts=10).to(device)
optimizer_cbm = torch.optim.Adam(cbm_model.c_to_y.parameters(), lr=LEARNING_RATE, weight_decay=FINAL_LAYER_PEN)
criterion_cbm = nn.BCEWithLogitsLoss()

train_loader_cbm = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=baseline_collate_fn
)
test_loader_cbm = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=baseline_collate_fn
)

print("=== Part 3: Independent CBM (c -> y Head) ===")
print("--- Model Training ---")
for epoch in range(NUM_EPOCHS):
    cbm_model.train()
    cbm_model.concept_predictor.eval()

    running_loss = 0.0
    
    for images, targets in train_loader_cbm:
        images, targets = images.to(device), targets.to(device)     
        optimizer_cbm.zero_grad()
        logits = cbm_model(images)
        loss = criterion_cbm(logits, targets)
        if ELASTIC_NET:      
            target_params = list(cbm_model.c_to_y.parameters())
            l1_penalty = sum(p.abs().sum() for p in target_params)
            l2_penalty = sum((p**2).sum() for p in target_params)
            elastic_net_reg = L1_LAMBDA * (
                EL_WEIGHT * l1_penalty + ((1 - EL_WEIGHT) / 2) * l2_penalty
            )
            loss = loss + elastic_net_reg
        
        
        loss.backward()
        optimizer_cbm.step()   
        running_loss += loss.item() * images.size(0)
        
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - CBM Loss: {epoch_loss:.4f}")


all_preds, all_labels = [], []

cbm_model.eval()
with torch.no_grad():
    for images, targets in test_loader_cbm:
        images, targets = images.to(device), targets.to(device)

        logits = cbm_model(images)
        probs = torch.sigmoid(logits)
        
        all_preds.extend(probs.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())

bottleneck_metrics = evaluate_predictions(all_labels, all_preds)
print("--- Independent CBM Metrics ---")
print(f"  • Accuracy:        {bottleneck_metrics['accuracy']:.4f}")
print(f"  • AUROC:           {bottleneck_metrics['auroc']:.4f}")
print(f"  • F1 Score:        {bottleneck_metrics['f1']:.4f}")
print(f"  • RMSE:            {bottleneck_metrics['rmse']:.4f}")
print(f"  • Avg Pearson (r): {bottleneck_metrics['avg_pearson']:.4f}")
print(f"  • Sparsity:        {bottleneck_metrics['sparsity_pct']:.1f}% active concepts")
print("=" * 60)

print("--- Baseline Metrics ---")
print(f"[Evaluation] Acc: {baseline_metrics['accuracy']:.4f} | AUROC: {baseline_metrics['auroc']:.4f} | F1: {baseline_metrics['f1']:.4f} | RMSE: {baseline_metrics['rmse']:.4f}")
print("="*60)
'''
Part 4 Code
'''

class HybridCBM(nn.Module):
    def __init__(self, concept_predictor, num_concepts=10, feature_dim=512):
        super(HybridCBM, self).__init__()
        self.concept_predictor = concept_predictor
        
        for param in self.concept_predictor.parameters():
            param.requires_grad = False
            
        self.f_c = nn.Linear(num_concepts, 1)
        self.s_x = nn.Linear(feature_dim, 1)
        self.extracted_features = None
        
        def hook_fn(module, input, output):
            self.extracted_features = torch.flatten(output, 1)

        hook_registered = False
        for name, module in self.concept_predictor.named_modules():
            if isinstance(module, nn.AdaptiveAvgPool2d) or 'avgpool' in name.lower():
                module.register_forward_hook(hook_fn)
                hook_registered = True
                print(f"[HybridCBM] Successfully hooked into backbone layer: '{name}'")
                break
                
        if not hook_registered:
            raise RuntimeError("Could not automatically locate 'avgpool' in concept_predictor submodules.")

    def forward(self, x):
        concept_logits = self.concept_predictor(x)
        concept_probs = torch.sigmoid(concept_logits)    
        features = self.extracted_features     
        f_c_logit = self.f_c(concept_probs)
        s_x_logit = self.s_x(features)
        
        return f_c_logit + s_x_logit
hybrid_cbm = HybridCBM(concept_predictor=concept_model, num_concepts=10, feature_dim=512).to(device)

trainable_params = list(hybrid_cbm.f_c.parameters()) + list(hybrid_cbm.s_x.parameters())
optimizer_hybrid = torch.optim.Adam(trainable_params, lr=LEARNING_RATE, weight_decay=FINAL_LAYER_PEN)
criterion_hybrid = nn.BCEWithLogitsLoss()

train_loader_hybrid = train_loader_cbm
test_loader_hybrid = test_loader_cbm

print("=== Part 4: Hybrid CBM (Concept Head + Side Channel) ===")
print("\n--- Training Model ---")
for epoch in range(NUM_EPOCHS):
    hybrid_cbm.train()
    hybrid_cbm.concept_predictor.eval()
    
    running_loss = 0.0
    for images, targets in train_loader_hybrid:
        images = images.to(device)
        targets = targets.to(device).float().unsqueeze(1) if targets.ndim == 1 else targets.to(device).float()
        
        optimizer_hybrid.zero_grad()
        logits = hybrid_cbm(images)
        loss = criterion_hybrid(logits, targets)
        if ELASTIC_NET:
          target_params = list(hybrid_cbm.f_c.parameters()) + list(
              hybrid_cbm.s_x.parameters()
          )
          l1_penalty = sum(p.abs().sum() for p in target_params)
          l2_penalty = sum((p ** 2).sum() for p in target_params)
          elastic_net_reg = L1_LAMBDA * (
              EL_WEIGHT * l1_penalty + ((1 - EL_WEIGHT) / 2) * l2_penalty
          )
        
          loss = loss + elastic_net_reg
        
        
        loss.backward()
        optimizer_hybrid.step()
        
        running_loss += loss.item() * images.size(0)
        
    epoch_loss = running_loss / len(train_loader_hybrid.dataset)
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - Hybrid CBM Loss: {epoch_loss:.4f}")

all_preds, all_labels = [], []

hybrid_cbm.eval()
with torch.no_grad():
    for images, targets in test_loader_hybrid:
        images = images.to(device)
        targets = targets.to(device).int()
        
        logits = hybrid_cbm(images)
        probs = torch.sigmoid(logits)
        
        all_preds.extend(probs.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())


hybrid_metrics = evaluate_predictions(all_labels, all_preds)
print("--- Hybrid CBM w/ Side Channel Metrics ---")
print("PART 4 (HYBRID CBM) CONCEPT EVALUATION METRICS:")
print(f"  • Accuracy:        {hybrid_metrics['accuracy']:.4f}")
print(f"  • AUROC:           {hybrid_metrics['auroc']:.4f}")
print(f"  • F1 Score:        {hybrid_metrics['f1']:.4f}")
print(f"  • RMSE:            {hybrid_metrics['rmse']:.4f}")
print(f"  • Avg Pearson (r): {hybrid_metrics['avg_pearson']:.4f}")
print(f"  • Sparsity:        {hybrid_metrics['sparsity_pct']:.1f}% active concepts")

print("--- Independent CBM Metrics ---")
print(f"  • Accuracy:        {bottleneck_metrics['accuracy']:.4f}")
print(f"  • AUROC:           {bottleneck_metrics['auroc']:.4f}")
print(f"  • F1 Score:        {bottleneck_metrics['f1']:.4f}")
print(f"  • RMSE:            {bottleneck_metrics['rmse']:.4f}")
print(f"  • Avg Pearson (r): {bottleneck_metrics['avg_pearson']:.4f}")
print(f"  • Sparsity:        {bottleneck_metrics['sparsity_pct']:.1f}% active concepts")

print("--- Baseline Metrics ---")
print(f"[Evaluation] Acc: {baseline_metrics['accuracy']:.4f} | AUROC: {baseline_metrics['auroc']:.4f} | F1: {baseline_metrics['f1']:.4f} | RMSE: {baseline_metrics['rmse']:.4f}")
print("="*60)
'''
Part 4 With Drop Out
'''


dropout_probs = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
dropout_models = {}
results_summary = {
    'dropout': [],
    'accuracy': [],
    'auroc': [],
    'concept_f1': []
}

class HybridCBMWithDropout(nn.Module):
    def __init__(self, concept_predictor, num_concepts=10, feature_dim=512, dropout_p=0.0):
        super(HybridCBMWithDropout, self).__init__()
        self.concept_predictor = concept_predictor
        
        for param in self.concept_predictor.parameters():
            param.requires_grad = False
            
        self.f_c = nn.Linear(num_concepts, 1)
        self.s_x = nn.Linear(feature_dim, 1)
        self.dropout = nn.Dropout(p=dropout_p)
        self.extracted_features = None

        def hook_fn(module, input, output):
            self.extracted_features = torch.flatten(output, 1)

        hook_registered = False
        for name, module in self.concept_predictor.named_modules():
            if isinstance(module, nn.AdaptiveAvgPool2d) or 'avgpool' in name.lower():
                module.register_forward_hook(hook_fn)
                hook_registered = True
                break
                
        if not hook_registered:
            raise RuntimeError("Could not automatically locate 'avgpool' in concept_predictor submodules.")
            
    def forward(self, x):
        concept_logits = self.concept_predictor(x)
        concept_probs = torch.sigmoid(concept_logits)    
        features = self.extracted_features       
        f_c_logit = self.f_c(concept_probs)
       
        dropped_features = self.dropout(features)
        s_x_logit = self.s_x(dropped_features)
        final_logits = f_c_logit + s_x_logit
        return final_logits, concept_probs

train_loader_hybrid = train_loader_cbm
test_loader_hybrid = test_loader_cbm

print("--- Part 4 Continued: Hybrid CBM with Side-Channel Dropout ---")
for p in dropout_probs:
    print(f"Training Hybrid CBM with Side-Channel Dropout: p = {p}")

    hybrid_cbm_d = HybridCBMWithDropout(
        concept_predictor=concept_model, 
        num_concepts=10, 
        feature_dim=512, 
        dropout_p=p
    ).to(device)

    trainable_params = list(hybrid_cbm_d.f_c.parameters()) + list(hybrid_cbm_d.s_x.parameters())
    optimizer_hybrid = torch.optim.Adam(trainable_params, lr=LEARNING_RATE, weight_decay=FINAL_LAYER_PEN)
    criterion_hybrid = nn.BCEWithLogitsLoss()

    for epoch in range(NUM_EPOCHS):
        hybrid_cbm_d.train()
        for param in hybrid_cbm_d.concept_predictor.parameters():
            param.requires_grad = False
            
        running_loss = 0.0
        for images, targets in train_loader_hybrid:
            images = images.to(device)
            targets = targets.to(device).float().unsqueeze(1) if targets.ndim == 1 else targets.to(device).float()

            optimizer_hybrid.zero_grad()
            
            logits, _ = hybrid_cbm_d(images)
            loss = criterion_hybrid(logits, targets)
            if ELASTIC_NET:
              target_params = list(hybrid_cbm_d.f_c.parameters()) + list(
                  hybrid_cbm_d.s_x.parameters()
              )
              l1_penalty = sum(p.abs().sum() for p in target_params)
              l2_penalty = sum((p ** 2).sum() for p in target_params)
              elastic_net_reg = L1_LAMBDA * (
                  EL_WEIGHT * l1_penalty + ((1 - EL_WEIGHT) / 2) * l2_penalty
              )
              loss = loss + elastic_net_reg
            loss.backward()
            optimizer_hybrid.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader_hybrid.dataset)
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - Loss: {epoch_loss:.4f}")


    hybrid_cbm_d.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, targets in test_loader_hybrid:
            images = images.to(device)
            targets = targets.to(device).int()
            
            logits, c_probs = hybrid_cbm_d(images)
            probs = torch.sigmoid(logits)
            
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    dropout_metrics = evaluate_predictions(all_labels, all_preds)
    dropout_models[p] = hybrid_cbm_d
    results_summary['dropout'].append(p)
    results_summary['accuracy'].append(dropout_metrics['accuracy'])
    results_summary['auroc'].append(dropout_metrics['auroc'])
    results_summary['concept_f1'].append(dropout_metrics['f1'])
    

    print(f"Dropout Rate = {p}")
    print(f"  • Accuracy:        {dropout_metrics['accuracy']:.4f}")
    print(f"  • AUROC:           {dropout_metrics['auroc']:.4f}")
    print(f"  • F1 Score:        {dropout_metrics['f1']:.4f}")
    print(f"  • RMSE:            {dropout_metrics['rmse']:.4f}")
    print(f"  • Avg Pearson (r): {dropout_metrics['avg_pearson']:.4f}")
    print(f"  • Sparsity:        {dropout_metrics['sparsity_pct']:.1f}% active concepts")
    print("\n")
print("=" * 60)
    
plt.figure(figsize=(8, 5))
plt.plot(results_summary['dropout'], results_summary['accuracy'], marker='o', label='Task Accuracy', linewidth=2)
plt.plot(results_summary['dropout'], results_summary['auroc'], marker='s', label='Task AUROC', linewidth=2)
plt.plot(results_summary['dropout'], results_summary['concept_f1'], marker='^', label='Concept Prediction F1', linestyle='--', linewidth=2)

plt.title("Hybrid CBM Accuracy vs. Side-Channel Dropout")
plt.xlabel("Dropout Probability (p) on Side Channel")
plt.ylabel("Metric Score")
plt.ylim(0.0, 1.05)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()


'''
Steerability
'''


import pandas as pd

models_to_compare = {
    "Independent CBM": cbm_model, 
    "Hybrid CBM (Standard)": hybrid_cbm,
}
for p, model_instance in dropout_models.items():
    models_to_compare[f"Hybrid + Dropout (p={p})"] = model_instance

comparative_summary = {}

print("--- Comparing Steerability Across Models ---")
for model_name, model in models_to_compare.items():
    if model is None:
        print(f"Skipping {model_name} (not found in workspace).")
        continue
        
    print(f"\nEvaluating Interventions for: {model_name}")
    model.eval()
    
    intervention_results = {}
    concept_sums = {j: 0.0 for j in range(10)}
    flip_sums = {j: 0.0 for j in range(10)}
    total_batches = 0
    
    with torch.no_grad():
        for images, targets in test_loader_hybrid:
            images = images.to(device)
            total_batches += 1
            
            out = model(images)
            if isinstance(out, tuple):
              final_logits, c_probs = out
            else:
              final_logits = out
              c_probs = torch.sigmoid(model.concept_predictor(images))           
            batch_features = (
                model.extracted_features.clone()
                if not isinstance(model, IndependentCBM)
                and model.extracted_features is not None
                else None
            )
                
            orig_task_probs = torch.sigmoid(final_logits)
            orig_preds = (orig_task_probs >= 0.5)
            
            for j in range(10):
                intervened_concepts = c_probs.clone()
                intervened_concepts[:, j] = 1.0 - intervened_concepts[:, j]
                
                if isinstance(model, IndependentCBM):
                    new_logits = model.c_to_y(intervened_concepts.to(device))
                    new_task_probs = torch.sigmoid(new_logits)
                elif isinstance(model, HybridCBM) or isinstance(model, HybridCBMWithDropout):
                    f_c_logits = model.f_c(intervened_concepts.to(device))
                    if batch_features is not None:
                        s_x_logits = model.s_x(batch_features)
                        new_logits = f_c_logits + s_x_logits
                    else:
                        new_logits = f_c_logits
                    new_task_probs = torch.sigmoid(new_logits)
                
                prob_changes = torch.abs(new_task_probs.cpu() - orig_task_probs.cpu())
                concept_sums[j] += prob_changes.mean().item()
                
                new_preds = (new_task_probs.cpu() >= 0.5)
                fraction_flipped = (orig_preds.cpu() != new_preds).float().mean().item()
                flip_sums[j] += fraction_flipped

    for j in range(10):
        avg_prob_change = concept_sums[j] / total_batches
        fraction_flipped = flip_sums[j] / total_batches
        intervention_results[f"Concept_{j}"] = {
            "avg_prob_change": avg_prob_change,
            "fraction_flipped": fraction_flipped
        }
        
    ranked_concepts = sorted(intervention_results.items(), key=lambda x: x[1]["avg_prob_change"], reverse=True)
    overall_steerability = np.mean([m["avg_prob_change"] for _, m in ranked_concepts])
    overall_flip_rate = np.mean([m["fraction_flipped"] for _, m in ranked_concepts])
    
    comparative_summary[model_name] = {
        "Mean Prob Change (Steerability)": overall_steerability,
        "Mean Flip Rate (%)": overall_flip_rate * 100,
        "Top Concept": ranked_concepts[0][0],
        "Top Effect": ranked_concepts[0][1]["avg_prob_change"]
    }
    
    print(f"--> Done [{model_name}] | Mean Prob Change: {overall_steerability:.4f} | Flip Rate: {overall_flip_rate*100:.1f}%")

summary_df = pd.DataFrame(comparative_summary).T
print(summary_df.to_string())
